#!/usr/bin/env python3
# bld_metrics_robust.py
"""
Broken Link Density (BLD) Metric — Robust Version
- Preserves original logging to file + console
- Preserves original CSV/graph/JSON outputs
- Robust Wayback resolution: /available with retries + CDX newest-first fallback
- Small host alias fix (e.g., theguardian.com -> www)
- Filters out Wayback _static / analytics requests
- N/A handling for missing data
"""

import asyncio
import csv
import json
import logging
import sys
import requests
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urlparse

# ---------- Windows asyncio policy (same as original) ----------
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# ---------- Logging: file + console ----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bld_metric.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# ---------- Small host alias fixes ----------
ALIASES = {
    'https://theguardian.com': 'https://www.theguardian.com',
}

# ---------- Hardcoded archive URLs (normalized keys) ----------
ARCHIVE_URLS = {
    'https://w3.org': 'https://web.archive.org/web/20250806070354id_/https://www.w3.org/',
    'https://example.com': 'https://web.archive.org/web/20250809095118id_/https://example.com/',
    'https://wikipedia.org': 'https://web.archive.org/web/20250809053232id_/https://www.wikipedia.org/',
    'https://bbc.com/news': 'https://web.archive.org/web/20250809010155id_/https://www.bbc.com/news',
    'https://archive.org': 'https://web.archive.org/web/20250730042019id_/https://archive.org/',
    'https://github.com': 'https://web.archive.org/web/20250809065555id_/https://github.com/',
    'https://discord.com': 'https://web.archive.org/web/20250808001155id_/https://discord.com/',
    'https://notion.so': 'https://web.archive.org/web/20250728204801id_/https://www.notion.so/',
    'https://twitter.com': 'https://web.archive.org/web/20250729210226id_/https://twitter.com/',
    'https://reddit.com': 'https://web.archive.org/web/20250809031420id_/https://www.reddit.com/',
    'https://facebook.com': 'https://web.archive.org/web/20250809011211id_/https://www.facebook.com/',
    'https://instagram.com': 'https://web.archive.org/web/20250809075255id_/https://www.instagram.com/',
}

# ---------- Helpers ----------
from urllib.parse import urlparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

USER_AGENT = "BLD-Research/1.0 (+https://example.org)"

def norm_url(u: str) -> str:
    """Normalize URL for mapping lookups (like your original)."""
    u = u.strip().lower().rstrip('/')
    u = u.replace('://www.', '://')
    return u

def _norm_origin(u: str) -> str:
    """Normalize to scheme://host[:port] for Wayback queries."""
    p = urlparse(u.strip())
    scheme = (p.scheme or "https").lower()
    host = (p.hostname or "")
    if not host:
        return u.strip().lower()
    if p.port:
        host = f"{host}:{p.port}"
    return f"{scheme}://{host}".lower()

def _sanitize_original_for_replay(original: str) -> str:
    """Make Wayback 'original' safe for replay."""
    p = urlparse(original)
    scheme = (p.scheme or "http").lower()
    host = (p.hostname or "")
    if host.endswith("."):
        host = host[:-1]
    if p.port:
        host = f"{host}:{p.port}"
    return f"{scheme}://{host}/"

def _build_replay(ts: str, original: str) -> str:
    clean = _sanitize_original_for_replay(original)
    return f"https://web.archive.org/web/{ts}id_/{clean}"

def _session_with_retries() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.5,                 # 0s, 1.5s, 3s, 4.5s, 6s...
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    s.headers.update({"User-Agent": USER_AGENT})
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s

def get_wayback_url(url: str) -> Optional[str]:
    """Resolve a snapshot with retries + CDX fallback; returns id_/ replay URL or None."""
    sess = _session_with_retries()
    timeout = 25

    origin = _norm_origin(url)
    origin = ALIASES.get(origin, origin)

    # 1) /available (usually newest)
    try:
        r = sess.get(
            "https://archive.org/wayback/available",
            params={"url": origin},
            timeout=timeout,
        )
        data = r.json()
        closest = data.get("archived_snapshots", {}).get("closest")
        if closest and closest.get("available") and closest.get("url"):
            u = closest["url"]
            if "id_/" not in u:
                u = u.replace("/http", "id_/http").replace("/https", "id_/https")
            return u
    except Exception as e:
        logger.warning(f"/available failed for {origin}: {e}")

    # 2) CDX newest-first fallback
    try:
        r = sess.get(
            "https://web.archive.org/cdx/search/cdx",
            params={
                "url": origin,
                "output": "json",
                "fl": "timestamp,original,statuscode",
                "filter": "statuscode:200",
                "limit": "1",
                "reverse": "true",
                "collapse": "digest",
            },
            timeout=timeout,
        )
        data = r.json()
        if isinstance(data, list) and len(data) >= 2 and isinstance(data[1], list):
            ts, orig, *_ = data[1]
            return _build_replay(ts, orig)
    except Exception as e:
        logger.warning(f"CDX fallback failed for {origin}: {e}")

    return None

# ---------- Analyzer ----------
class BrokenLinkDensityAnalyzer:
    def __init__(self):
        self.output_dir = Path("outputs/bld")
        self.results_file = self.output_dir / "bld_results.csv"
        self.graphs_dir = self.output_dir / "graphs"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.graphs_dir.mkdir(exist_ok=True, parents=True)
        self.results = []

    async def setup_browser(self):
        from playwright.async_api import async_playwright
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            ignore_https_errors=True
        )
        return playwright, browser, context

    def is_relevant_resource(self, original_url: str, resource_url: str) -> bool:
        """Filter out Wayback static/analytics resources."""
        parsed = urlparse(resource_url)
        if 'web.archive.org' in parsed.netloc and '/_static/' in parsed.path:
            return False
        if 'analytics' in parsed.path or 'googletagmanager' in parsed.netloc:
            return False
        return True

    async def measure_bld(self, url: str, archive_url: str, category: str, context) -> Dict:
        result = {
            'url': url,
            'category': category,
            'total_resources': 0,
            'failed_resources': 0,
            'bld_score': None,  # None = no data
            'timestamp': datetime.now().isoformat()
        }
        page = None
        resources_data = []
        try:
            page = await context.new_page()

            # Track responses / failures
            async def handle_response(response):
                try:
                    r_url = response.url
                    if self.is_relevant_resource(url, r_url):
                        resources_data.append({
                            'url': r_url,
                            'status': response.status,
                            'failed': response.status >= 400 or response.status == 0
                        })
                except Exception:
                    pass

            async def handle_request_failed(request):
                try:
                    r_url = request.url
                    if self.is_relevant_resource(url, r_url):
                        resources_data.append({
                            'url': r_url,
                            'status': 0,
                            'failed': True
                        })
                except Exception:
                    pass

            page.on('response', handle_response)
            page.on('requestfailed', handle_request_failed)

            timeout = 30000
            page.set_default_timeout(timeout)

            logger.info(f"Loading archived page for BLD: {url} -> {archive_url}")
            print(f"  Loading {category.upper()}: {url}")
            try:
                # 'load' can complete before subresources; 'networkidle' is safer but slower.
                await page.goto(archive_url, wait_until='load', timeout=timeout)
                await page.wait_for_timeout(5000)
            except Exception as e:
                logger.warning(f"Navigation issue for {url}: {e}")

            for resource in resources_data:
                result['total_resources'] += 1
                if resource['failed']:
                    result['failed_resources'] += 1

            if result['total_resources'] > 0:
                result['bld_score'] = result['failed_resources'] / result['total_resources']
                if result['bld_score'] < 0.1:
                    print(f"  [GOOD] Low BLD: {result['bld_score']:.3f}")
                elif result['bld_score'] < 0.3:
                    print(f"  [MEDIUM] Moderate BLD: {result['bld_score']:.3f}")
                else:
                    print(f"  [HIGH] High BLD: {result['bld_score']:.3f}")
            else:
                print("  [NO DATA] No resources counted.")
                logger.info(f"No resources counted for {url}")

        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
        finally:
            if page:
                await page.close()
        return result

    async def analyze_urls(self, urls: List[Dict]) -> List[Dict]:
        # Defensive: accept CSVs with at least a 'url' column.
        cleaned = []
        for r in urls:
            u = (r.get('url') or '').strip()
            if not u:
                continue
            cleaned.append({
                'url': u,
                'category': (r.get('category') or 'unknown').strip()
            })

        playwright, browser, context = await self.setup_browser()
        try:
            for idx, url_data in enumerate(cleaned, 1):
                url = url_data['url']
                category = url_data['category']

                print(f"\n[{idx}/{len(cleaned)}] Processing {url}")
                logger.info(f"[BLD] Processing {url} (category={category})")

                # 1) normalized hard-coded key
                key = norm_url(url)
                archive_url = ARCHIVE_URLS.get(key)

                # 2) robust resolver if not in map
                if not archive_url:
                    # apply alias for origin when resolving
                    archive_url = get_wayback_url(url)
                    if archive_url:
                        logger.info(f"Wayback resolved: {url} -> {archive_url}")

                if not archive_url:
                    print("  [NO DATA] No archive URL available.")
                    logger.warning(f"No archive snapshot found for {url}")
                    self.results.append({
                        'url': url,
                        'category': category,
                        'total_resources': 0,
                        'failed_resources': 0,
                        'bld_score': None,
                        'timestamp': datetime.now().isoformat()
                    })
                    continue

                result = await self.measure_bld(url, archive_url, category, context)
                self.results.append(result)

                # gentle pacing
                await asyncio.sleep(1)
        finally:
            await context.close()
            await browser.close()
            await playwright.stop()

        self.save_results()
        return self.results

    def save_results(self):
        if not self.results:
            return
        self.output_dir.mkdir(exist_ok=True, parents=True)
        with open(self.results_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['url', 'category', 'total_resources', 'failed_resources', 'bld_score', 'timestamp']
            )
            writer.writeheader()
            for r in self.results:
                row = dict(r)
                if row['bld_score'] is None:
                    row['bld_score'] = "N/A"
                writer.writerow(row)
        print(f"\nResults saved to {self.results_file}")
        logger.info(f"Results CSV written: {self.results_file}")

    def calculate_and_visualize(self):
        if not self.results:
            return

        stats: Dict[str, Dict] = {}
        categories = sorted(set(r['category'] for r in self.results))
        for cat in categories:
            cat_results = [r for r in self.results if r['category'] == cat]
            valid = [r for r in cat_results if isinstance(r['bld_score'], float)]
            if valid:
                stats[cat] = {
                    'avg_bld': sum(r['bld_score'] for r in valid) / len(valid),
                    'avg_resources': sum(r['total_resources'] for r in valid) / len(valid),
                    'avg_failures': sum(r['failed_resources'] for r in valid) / len(valid),
                    'sites_analyzed': len(valid),
                    'error_rate': 100 * (len(cat_results) - len(valid)) / max(1, len(cat_results))
                }
            else:
                stats[cat] = {
                    'avg_bld': None,
                    'sites_analyzed': 0,
                    'error_rate': 100.0
                }

        print("\n" + "="*60)
        print("BLD RESULTS BY CATEGORY")
        print("="*60)
        for cat, s in stats.items():
            print(f"\n{cat.upper()}:")
            if s['avg_bld'] is not None:
                print(f"  Average BLD Score: {s['avg_bld']:.3f}")
                print(f"  Avg Resources/Page: {s['avg_resources']:.0f}")
                print(f"  Avg Failures/Page: {s['avg_failures']:.0f}")
                print(f"  Sites analyzed (valid): {s['sites_analyzed']}")
                print(f"  Error rate: {s['error_rate']:.1f}%")
            else:
                print("  No valid data.")
                print(f"  Error rate: {s['error_rate']:.1f}%")

        # bar chart if any valid data
        if any(s['avg_bld'] is not None for s in stats.values()):
            fig, ax = plt.subplots(figsize=(10, 6))
            cats_for_plot = [c for c in stats if stats[c]['avg_bld'] is not None]
            bld_scores = [stats[c]['avg_bld'] for c in cats_for_plot]
            colors = ['green' if s < 0.3 else 'orange' if s < 0.6 else 'red' for s in bld_scores]
            bars = ax.bar(cats_for_plot, bld_scores, color=colors)
            ax.set_ylabel('BLD Score (lower is better)')
            ax.set_title('Broken Link Density by Website Category')
            ax.set_ylim(0, 1)
            for bar, score in zip(bars, bld_scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', fontweight='bold')
            plt.tight_layout()
            graph_path = self.graphs_dir / 'bld_analysis.png'
            plt.savefig(graph_path, dpi=150)
            print(f"\nGraph saved to {graph_path}")
            logger.info(f"Graph written: {graph_path}")

        stats_file = self.output_dir / "bld_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to {stats_file}")
        logger.info(f"Stats JSON written: {stats_file}")

# ---------- Main ----------
async def main():
    print("\n" + "="*60)
    print("BROKEN LINK DENSITY (BLD) ANALYSIS")
    print("="*60)

    urls_file = Path("../data/urls.csv")
    if urls_file.exists():
        print(f"Loading URLs from {urls_file}")
        with open(urls_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            test_urls = list(reader)
            if not test_urls or 'url' not in reader.fieldnames:
                print("ERROR: CSV must contain a 'url' column.")
                logger.error("CSV missing 'url' column.")
                return
    else:
        print("ERROR: data/urls.csv not found!")
        logger.error("CSV not found at ../data/urls.csv")
        return

    analyzer = BrokenLinkDensityAnalyzer()
    await analyzer.analyze_urls(test_urls)
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
