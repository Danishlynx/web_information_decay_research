"""
Visual Correspondence Score (VCS) Metric
Compares visual similarity between live and archived pages
Based on Reyes Ayala, 2019 research using SSIM algorithm

Improvements:
- Compute SSIM in grayscale to emphasize structure over color.
- Use consistent timeouts and a deterministic browser context (locale, timezone, UA).
- Remove overlays (cookie banners, etc.) and disable animations prior to screenshots.
- Treat "no measurement" as None instead of 0.0 to avoid bias; separate availability stats.
- Safer Wayback URL normalization to get "id_/" screenshot-friendly URLs.
- Hash filenames to avoid Windows path length / invalid character issues.
- Basic retry for screenshot capture to reduce transient failures.
"""

import asyncio
import csv
import json
import logging
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageOps
from skimage.metrics import structural_similarity as ssim
import io
import requests
from urllib.parse import quote
import hashlib
import time
import random

# Windows event loop policy
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vcs_metric.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Optional predefined archive URLs
ARCHIVE_URLS: Dict[str, str] = {}

# Constants
VIEWPORT = {'width': 1280, 'height': 720}
NAV_TIMEOUT_MS = 45000  # consistent timeout for all categories
NETWORK_IDLE_TIMEOUT_MS = 20000
POST_NAV_SETTLE_MS = 1500
RETRY_ATTEMPTS = 2
RETRY_BACKOFF_BASE = 0.8  # seconds (jittered)

class VisualCorrespondenceAnalyzer:
    def __init__(self):
        self.output_dir = Path("outputs/vcs")
        self.results_file = self.output_dir / "vcs_results.csv"
        self.graphs_dir = self.output_dir / "graphs"
        self.screenshots_dir = self.output_dir / "vcs_screenshots"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.graphs_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[Dict] = []

    @staticmethod
    def _safe_name(url: str, suffix: str) -> str:
        h = hashlib.md5(url.encode('utf-8')).hexdigest()
        return f"{h}_{suffix}.png"

    async def setup_browser(self):
        from playwright.async_api import async_playwright
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport=VIEWPORT,
            ignore_https_errors=True,
            locale='en-US',
            timezone_id='UTC',
            user_agent=('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                        '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')),
        return playwright, browser, context[0]

    @staticmethod
    def _normalize_wayback_url(archive_url: str) -> str:
        # Ensure /web/id_/ is used once to request original asset URLs for screenshots
        if '/web/' in archive_url and '/web/id_/' not in archive_url:
            return archive_url.replace('/web/', '/web/id_/', 1)
        return archive_url

    @staticmethod
    def _disable_animations_and_overlays_script() -> str:
        return """
        (() => {
          try {
            const style = document.createElement('style');
            style.innerHTML = "*,*::before,*::after{animation:none!important;transition:none!important}";
            document.head.appendChild(style);
          } catch (e) {}
          const sels = [
            "[id*='cookie']","[class*='cookie']",".consent",".cc-banner",".gdpr",
            ".osano-cm-dialog",".truste",".optanon-alert-box-wrapper",".qc-cmp2-container",
            ".js-consent-banner",".cookie-consent",".cmp-container",".consent-banner",
            ".appconsent", ".sp-message", ".sp_veil"
          ];
          try {
            sels.forEach(s => document.querySelectorAll(s).forEach(n => n.remove()));
          } catch (e) {}
        })();
        """

    def calculate_ssim(self, img1_bytes: bytes, img2_bytes: bytes) -> float:
        """Calculate SSIM between two images in grayscale to reduce color bias."""
        try:
            img1 = Image.open(io.BytesIO(img1_bytes))
            img2 = Image.open(io.BytesIO(img2_bytes))

            # Convert to grayscale
            img1 = ImageOps.grayscale(img1)
            img2 = ImageOps.grayscale(img2)

            # Resize to common minimum dimensions
            width = min(img1.width, img2.width)
            height = min(img1.height, img2.height)
            if width < 10 or height < 10:
                return 0.0  # guard against tiny images

            img1 = img1.resize((width, height), Image.Resampling.LANCZOS)
            img2 = img2.resize((width, height), Image.Resampling.LANCZOS)

            arr1 = np.array(img1, dtype=np.uint8)
            arr2 = np.array(img2, dtype=np.uint8)

            return float(ssim(arr1, arr2, data_range=255))
        except Exception as e:
            logger.error(f"SSIM calculation error: {e}")
            return 0.0

    async def _safe_goto(self, page, url: str, timeout_ms: int) -> None:
        # Two-phase load to avoid SPA hangs on networkidle
        try:
            await page.goto(url, wait_until='domcontentloaded', timeout=timeout_ms)
        except Exception as e:
            logger.warning(f"domcontentloaded timeout for {url}: {e}")
        # Try to reach a quieter network state, but don't fail the run solely on this
        try:
            await page.wait_for_load_state('networkidle', timeout=min(timeout_ms, NETWORK_IDLE_TIMEOUT_MS))
        except Exception:
            pass  # acceptable for SPAs

    async def _prepare_page(self, context):
        page = await context.new_page()
        page.set_default_timeout(NAV_TIMEOUT_MS)
        return page

    async def _pre_screenshot_sanitization(self, page):
        # Disable animations and remove common overlays
        try:
            await page.add_style_tag(content="*,*::before,*::after{animation:none!important;transition:none!important}")
        except Exception:
            pass
        try:
            await page.evaluate(self._disable_animations_and_overlays_script())
        except Exception:
            pass
        await page.wait_for_timeout(300)  # allow reflow

    async def _capture_page_screenshot(self, context, url: str, *, full_page: bool = False) -> Optional[bytes]:
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            page = None
            try:
                page = await self._prepare_page(context)
                await self._safe_goto(page, url, NAV_TIMEOUT_MS)
                await self._pre_screenshot_sanitization(page)
                await page.wait_for_timeout(POST_NAV_SETTLE_MS)
                shot = await page.screenshot(full_page=full_page)
                return shot
            except Exception as e:
                logger.warning(f"Screenshot attempt {attempt}/{RETRY_ATTEMPTS} failed for {url}: {e}")
                if attempt < RETRY_ATTEMPTS:
                    # jittered backoff
                    time.sleep(RETRY_BACKOFF_BASE + random.random() * RETRY_BACKOFF_BASE)
            finally:
                if page:
                    await page.close()
        return None

    async def capture_screenshots(self, url: str, archive_url: str, context, *, full_page: bool = False) -> Tuple[Optional[bytes], Optional[bytes]]:
        """Capture screenshots of live and archived pages with retries and sanitization."""
        live_screenshot = None
        archive_screenshot = None

        # Archive first
        try:
            print(f"  Capturing archived page screenshot...")
            archive_screenshot = await self._capture_page_screenshot(context, archive_url, full_page=full_page)
        except Exception as e:
            logger.error(f"Archive screenshot error: {e}")

        # Live page
        try:
            print(f"  Capturing live page screenshot...")
            live_screenshot = await self._capture_page_screenshot(context, url, full_page=full_page)
        except Exception as e:
            logger.error(f"Live screenshot error: {e}")

        return live_screenshot, archive_screenshot

    async def measure_vcs(self, url: str, archive_url: str, category: str, context, *, full_page: bool = False) -> Dict:
        """Measure Visual Correspondence Score"""
        result: Dict = {
            'url': url,
            'category': category,
            'vcs_score': None,          # None means "no measurement" vs 0.0 meaning "measured but dissimilar"
            'has_live': False,
            'has_archive': False,
            'error': None,
            'timestamp': datetime.now().isoformat()
        }

        try:
            print(f"  Measuring visual correspondence for {category.upper()}: {url}")
            live_screenshot, archive_screenshot = await self.capture_screenshots(url, archive_url, context, full_page=full_page)

            if live_screenshot:
                result['has_live'] = True
                live_path = self.screenshots_dir / self._safe_name(url, "live")
                with open(live_path, 'wb') as f:
                    f.write(live_screenshot)

            if archive_screenshot:
                result['has_archive'] = True
                archive_path = self.screenshots_dir / self._safe_name(url, "archive")
                with open(archive_path, 'wb') as f:
                    f.write(archive_screenshot)

            if live_screenshot and archive_screenshot:
                score = self.calculate_ssim(live_screenshot, archive_screenshot)
                result['vcs_score'] = score
                if score > 0.8:
                    print(f"  [EXCELLENT] High similarity: {score:.3f}")
                elif score > 0.5:
                    print(f"  [GOOD] Moderate similarity: {score:.3f}")
                else:
                    print(f"  [POOR] Low similarity: {score:.3f}")
            else:
                result['error'] = "Missing screenshots"
                print(f"  [ERROR] Could not capture both screenshots")
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"VCS measurement error: {e}")

        return result

    def _lookup_archive_url(self, url: str) -> Optional[str]:
        # 1) Predefined mapping
        archive_url = ARCHIVE_URLS.get(url)
        if archive_url:
            return self._normalize_wayback_url(archive_url)

        # 2) Wayback API lookup
        try:
            api_url = f"https://archive.org/wayback/available?url={quote(url)}"
            resp = requests.get(api_url, timeout=10)
            data = resp.json()
            closest = data.get('archived_snapshots', {}).get('closest', {})
            archive_url = closest.get('url')
            if archive_url:
                return self._normalize_wayback_url(archive_url)
        except Exception as e:
            logger.warning(f"Wayback lookup failed for {url}: {e}")
        return None

    async def analyze_urls(self, urls: List[Dict], *, full_page: bool = False) -> List[Dict]:
        playwright, browser, context = await self.setup_browser()
        try:
            for idx, url_data in enumerate(urls, 1):
                url = url_data.get('url', '').strip()
                category = url_data.get('category', 'unknown').strip() or 'unknown'
                if not url:
                    continue

                print(f"\n[{idx}/{len(urls)}] Processing {url}")

                archive_url = self._lookup_archive_url(url)
                if not archive_url:
                    print(f"  No archive found - keeping score as None and tracking availability")
                    self.results.append({
                        'url': url,
                        'category': category,
                        'vcs_score': None,
                        'has_live': False,
                        'has_archive': False,
                        'error': 'No archive found',
                        'timestamp': datetime.now().isoformat()
                    })
                    continue

                result = await self.measure_vcs(url, archive_url, category, context, full_page=full_page)
                self.results.append(result)
                await asyncio.sleep(1.2)  # gentle pacing
        finally:
            await context.close()
            await browser.close()
            await playwright.stop()

        self.save_results()
        return self.results

    def save_results(self):
        with open(self.results_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['url', 'category', 'vcs_score', 'has_live', 'has_archive', 'error', 'timestamp']
            )
            writer.writeheader()
            # CSV will serialize None as empty; that's fine
            writer.writerows(self.results)
        print(f"\nResults saved to {self.results_file}")

    def calculate_and_visualize(self):
        # Separate availability and similarity
        categories_all = ['html', 'spa', 'social', 'api']
        stats = {}

        for cat in categories_all:
            cat_results = [r for r in self.results if r.get('category') == cat]
            measured_scores = [r['vcs_score'] for r in cat_results if r.get('vcs_score') is not None]
            has_archive_count = sum(1 for r in cat_results if r.get('has_archive'))
            has_live_count = sum(1 for r in cat_results if r.get('has_live'))
            total = len(cat_results)
            no_archive_count = total - has_archive_count

            if measured_scores:
                arr = np.array(measured_scores, dtype=float)
                stats[cat] = {
                    'avg_vcs': float(np.mean(arr)),
                    'min_vcs': float(np.min(arr)),
                    'max_vcs': float(np.max(arr)),
                    'std_vcs': float(np.std(arr)),
                    'measured_count': int(len(measured_scores)),
                    'total_urls': int(total),
                    'no_archive_count': int(no_archive_count),
                    'has_live_count': int(has_live_count),
                    'has_archive_count': int(has_archive_count),
                }
            else:
                stats[cat] = {
                    'avg_vcs': 0.0,
                    'min_vcs': 0.0,
                    'max_vcs': 0.0,
                    'std_vcs': 0.0,
                    'measured_count': 0,
                    'total_urls': int(total),
                    'no_archive_count': int(no_archive_count),
                    'has_live_count': int(has_live_count),
                    'has_archive_count': int(has_archive_count),
                }

        # Console summary
        print("\n" + "="*60)
        print("VISUAL CORRESPONDENCE SCORE RESULTS")
        print("="*60)
        for cat, s in stats.items():
            print(f"\n{cat.upper()}:")
            print(f"  Average VCS (measured): {s['avg_vcs']:.3f}  "
                  f"(n={s['measured_count']}/{s['total_urls']})")
            print(f"  Range: {s['min_vcs']:.3f} - {s['max_vcs']:.3f}")
            print(f"  Std Dev: {s['std_vcs']:.3f}")
            print(f"  Has archive: {s['has_archive_count']} | No archive: {s['no_archive_count']}")
            print(f"  Has live: {s['has_live_count']}")

        # Bar chart of average VCS by category (measured only)
        categories = list(stats.keys())
        vcs_scores = [stats[c]['avg_vcs'] for c in categories]
        # Keep the same color mapping idea, though this is only for visuals
        colors = ['green' if s > 0.7 else 'orange' if s > 0.4 else 'red' for s in vcs_scores]

        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(categories, vcs_scores, color=colors)
        ax.set_ylabel('VCS Score')
        ax.set_title('Average Visual Correspondence by Category (Measured Only)')
        ax.set_ylim(0, 1)
        for bar, score in zip(bars, vcs_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', fontweight='bold')

        plt.tight_layout()
        graph_path = self.graphs_dir / 'vcs_analysis.png'
        plt.savefig(graph_path, dpi=150)
        print(f"\nGraph saved to {graph_path}")

        stats_file = self.output_dir / "vcs_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to {stats_file}")


async def main():
    print("\n" + "="*60)
    print("VISUAL CORRESPONDENCE SCORE (VCS) ANALYSIS")
    print("="*60)

    urls_file = Path("../data/urls.csv")
    if urls_file.exists():
        print(f"Loading URLs from {urls_file}")
        with open(urls_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            test_urls = list(reader)
    else:
        print("ERROR: data/urls.csv not found!")
        return

    analyzer = VisualCorrespondenceAnalyzer()
    # Set full_page=True if you want entire page screenshots (may increase noise on very long pages)
    await analyzer.analyze_urls(test_urls, full_page=False)
    analyzer.calculate_and_visualize()

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    try:
        if sys.platform == 'win32':
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(main())
        else:
            asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
