"""
Visual Correspondence Score (VCS)
Compares visual similarity between LIVE and ARCHIVED pages via SSIM.

Bias fixes:
- Availability is NOT mixed into similarity: missing shots -> vcs_score=None.
- Fresh browser context per URL (prevents cookie/storage leakage).
- Robust Wayback 'available' resolution that always returns id_/ form.
- Light stabilization to reduce animation/AB-test noise.
"""

import asyncio
import csv
import io
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote, urlparse

import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# Windows-specific asyncio configuration
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("vcs_metric.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("VCS")

# Output locations
OUT_DIR = Path("outputs/vcs")
SS_DIR = OUT_DIR / "screenshots"
GRAPH_DIR = OUT_DIR / "graphs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SS_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# Optional pre-mapped archives (leave empty in most runs)
ARCHIVE_URLS: Dict[str, str] = {}

VIEWPORT = {"width": 1280, "height": 720}
TIMEOUT_MS = 45_000
EXTRA_WAIT_MS = 3_000


def _wayback_available_id(url: str, session: Optional[requests.Session] = None) -> Optional[str]:
    """Resolve an archived URL via 'available' API, returning id_/ form if found."""
    sess = session or requests.Session()
    api = f"https://web.archive.org/wayback/available?url={quote(url)}"
    try:
        r = sess.get(api, timeout=20)
        if not r.ok:
            return None
        data = r.json()
        closest = data.get("archived_snapshots", {}).get("closest")
        if not closest or not closest.get("available"):
            return None
        # Build the id_/ form robustly
        snap_url = closest.get("url", "")
        if "/web/" not in snap_url:
            return None
        ts = snap_url.split("/web/")[1].split("/")[0]
        orig = "/".join(snap_url.split("/web/")[1].split("/")[1:])
        return f"https://web.archive.org/web/{ts}id_/{orig}"
    except Exception:
        return None


def _normalize_filename(u: str, suffix: str) -> Path:
    safe = u.replace("://", "_").replace("/", "_").replace("?", "_")[:200]
    return SS_DIR / f"{safe}_{suffix}.png"


def _stabilize_page_js() -> str:
    """CSS/JS to reduce motion/late layout changes before screenshot."""
    return """
        try {
            const s = document.createElement('style');
            s.innerHTML = `
                * { animation: none !important; transition: none !important; }
                html::-webkit-scrollbar, *::-webkit-scrollbar { display: none !important; }
            `;
            document.head.appendChild(s);
        } catch(e) {}
    """


def _calc_ssim(img1_bytes: bytes, img2_bytes: bytes) -> float:
    try:
        im1 = Image.open(io.BytesIO(img1_bytes)).convert("RGB")
        im2 = Image.open(io.BytesIO(img2_bytes)).convert("RGB")
        w = min(im1.width, im2.width)
        h = min(im1.height, im2.height)
        im1 = im1.resize((w, h), Image.Resampling.LANCZOS)
        im2 = im2.resize((w, h), Image.Resampling.LANCZOS)
        a1 = np.array(im1)
        a2 = np.array(im2)
        return float(ssim(a1, a2, channel_axis=2, data_range=255))
    except Exception as e:
        logger.error(f"SSIM error: {e}")
        return 0.0


async def _setup_browser():
    from playwright.async_api import async_playwright
    pw = await async_playwright().start()
    browser = await pw.chromium.launch(headless=True, args=["--disable-dev-shm-usage", "--no-sandbox"])
    return pw, browser


async def _capture_pair(url: str, archive_url: str, browser) -> Tuple[Optional[bytes], Optional[bytes]]:
    """Capture (live, archive) screenshots using a fresh context (no state leakage)."""
    live_shot = None
    arch_shot = None

    # ARCHIVE
    context = await browser.new_context(viewport=VIEWPORT, ignore_https_errors=True, java_script_enabled=True,
                                        extra_http_headers={"Accept-Encoding": "identity"})
    try:
        page = await context.new_page()
        page.set_default_timeout(TIMEOUT_MS)
        await page.goto(archive_url, wait_until="networkidle", timeout=TIMEOUT_MS)
        await page.wait_for_timeout(EXTRA_WAIT_MS)
        await page.add_init_script(_stabilize_page_js())
        arch_shot = await page.screenshot(full_page=False)
    except Exception as e:
        logger.warning(f"Archive shot failed for {url}: {e}")
    finally:
        await context.close()

    # LIVE
    context = await browser.new_context(viewport=VIEWPORT, ignore_https_errors=True)
    try:
        page = await context.new_page()
        page.set_default_timeout(TIMEOUT_MS)
        await page.goto(url, wait_until="networkidle", timeout=TIMEOUT_MS)
        await page.wait_for_timeout(EXTRA_WAIT_MS)
        await page.add_init_script(_stabilize_page_js())
        live_shot = await page.screenshot(full_page=False)
    except Exception as e:
        logger.warning(f"Live shot failed for {url}: {e}")
    finally:
        await context.close()

    return live_shot, arch_shot


class VisualCorrespondenceAnalyzer:
    def __init__(self):
        self.results: List[Dict] = []

    async def analyze_urls(self, urls: List[Dict]) -> List[Dict]:
        pw, browser = await _setup_browser()
        session = requests.Session()
        try:
            for i, row in enumerate(urls, 1):
                url = row.get("url", "").strip()
                category = row.get("category", "unknown").strip().lower()
                if not url:
                    continue

                print(f"\n[{i}/{len(urls)}] {url} ({category})")

                archive_url = ARCHIVE_URLS.get(url) or _wayback_available_id(url, session=session)
                if not archive_url:
                    logger.info("No archive available (VCS recorded as None).")
                    self.results.append({
                        "url": url, "category": category, "archive_url": None,
                        "vcs_score": None, "has_live": False, "has_archive": False,
                        "timestamp": datetime.utcnow().isoformat(), "error": "no-archive"
                    })
                    continue

                live_shot, arch_shot = await _capture_pair(url, archive_url, browser)
                has_live = live_shot is not None
                has_arch = arch_shot is not None

                if has_live:
                    _normalize_filename(url, "live").write_bytes(live_shot)
                if has_arch:
                    _normalize_filename(url, "arch").write_bytes(arch_shot)

                score = None
                if has_live and has_arch:
                    score = _calc_ssim(arch_shot, live_shot)

                self.results.append({
                    "url": url, "category": category, "archive_url": archive_url,
                    "vcs_score": score, "has_live": has_live, "has_archive": has_arch,
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": None if score is not None else "missing-screenshot"
                })
        finally:
            await browser.close()
            await pw.stop()

        self._save_results()
        return self.results

    def _save_results(self):
        out_csv = OUT_DIR / "vcs_results.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            import csv
            w = csv.DictWriter(f, fieldnames=["url", "category", "archive_url", "vcs_score",
                                              "has_live", "has_archive", "error", "timestamp"])
            w.writeheader()
            w.writerows(self.results)
        print(f"Results saved to {out_csv}")

    def summarize_and_plot(self):
        # Build stats (exclude None in averages; still report counts)
        cats = sorted({r["category"] for r in self.results})
        stats = {}
        for c in cats:
            rows = [r for r in self.results if r["category"] == c]
            scores = [r["vcs_score"] for r in rows if r["vcs_score"] is not None]
            stats[c] = {
                "avg_vcs": float(np.mean(scores)) if scores else 0.0,
                "min_vcs": float(np.min(scores)) if scores else 0.0,
                "max_vcs": float(np.max(scores)) if scores else 0.0,
                "std_vcs": float(np.std(scores)) if scores else 0.0,
                "count": len(scores),
                "no_archive_count": sum(1 for r in rows if not r["has_archive"]),
                "missing_shot_count": sum(1 for r in rows if r["vcs_score"] is None and r["has_archive"]),
            }

        print("\n" + "="*60)
        print("VCS RESULTS")
        print("="*60)
        for c, s in stats.items():
            print(f"\n{c.upper()}")
            print(f"  Avg: {s['avg_vcs']:.3f}  Range: {s['min_vcs']:.3f}-{s['max_vcs']:.3f}  Std: {s['std_vcs']:.3f}")
            print(f"  Valid: {s['count']}  No-archive: {s['no_archive_count']}  Missing-shot: {s['missing_shot_count']}")

        # Plot
        labels = list(stats.keys())
        vals = [stats[c]["avg_vcs"] for c in labels]
        colors = ["green" if v > 0.7 else "orange" if v > 0.4 else "red" for v in vals]
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(labels, vals, color=colors)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Average VCS (SSIM)")
        ax.set_title("Average Visual Correspondence by Category")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")
        fig.tight_layout()
        out_png = GRAPH_DIR / "vcs_analysis.png"
        fig.savefig(out_png, dpi=160)
        print(f"\nGraph saved to {out_png}")


async def main():
    print("\n" + "="*60)
    print("VISUAL CORRESPONDENCE SCORE (VCS)")
    print("="*60)

    urls_csv = Path("../data/urls.csv")
    if not urls_csv.exists():
        print("ERROR: ../data/urls.csv not found.")
        return
    with urls_csv.open("r", encoding="utf-8") as f:
        import csv
        urls = list(csv.DictReader(f))

    analyzer = VisualCorrespondenceAnalyzer()
    await analyzer.analyze_urls(urls)
    analyzer.summarize_and_plot()

    print("\nDONE")

if __name__ == "__main__":
    try:
        if sys.platform == "win32":
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(main())
        else:
            asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted")
