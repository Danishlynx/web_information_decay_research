"""
Memento Damage Score (MDS)
Weights missing resources by importance (Brunelle et al., 2014).

Bias fixes:
- Deduplicate resource URLs (count each once).
- Optional ignore-list for trackers/analytics (prevents category skew).
- Robust Wayback 'available' resolver returning id_/ form.
- Availability is separated: no archive -> mds_score=None (excluded from averages).
"""

import asyncio
import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote

import matplotlib.pyplot as plt
import numpy as np
import requests

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("mds_metric.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("MDS")

OUT_DIR = Path("outputs/mds")
GRAPH_DIR = OUT_DIR / "graphs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

RESOURCE_WEIGHTS = {
    "document": 1.0,
    "stylesheet": 0.9,
    "script": 0.8,
    "image": 0.5,
    "font": 0.4,
    "media": 0.3,
    "xhr": 0.6,
    "fetch": 0.6,
    "other": 0.1,
}

# Optional: ignore common trackers (keeps default conservative)
IGNORED_HOST_SNIPPETS = (
    "googletagmanager.com", "google-analytics.com", "doubleclick.net",
    "facebook.net", "hotjar.com", "mixpanel.com", "segment.com"
)

VIEWPORT = {"width": 1920, "height": 1080}
TIMEOUT_MS = 45_000
EXTRA_WAIT_MS = 3_000


def _wayback_available_id(url: str, session: Optional[requests.Session] = None) -> Optional[str]:
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
        snap_url = closest.get("url", "")
        if "/web/" not in snap_url:
            return None
        ts = snap_url.split("/web/")[1].split("/")[0]
        orig = "/".join(snap_url.split("/web/")[1].split("/")[1:])
        return f"https://web.archive.org/web/{ts}id_/{orig}"
    except Exception:
        return None


async def _setup_browser():
    from playwright.async_api import async_playwright
    pw = await async_playwright().start()
    browser = await pw.chromium.launch(headless=True, args=["--disable-dev-shm-usage", "--no-sandbox"])
    context = await browser.new_context(viewport=VIEWPORT, ignore_https_errors=True,
                                        java_script_enabled=True,
                                        extra_http_headers={"Accept-Encoding": "identity"})
    return pw, browser, context


class MementoDamageAnalyzer:
    def __init__(self):
        self.results: List[Dict] = []

    async def _measure_one(self, url: str, archive_url: str, category: str, context) -> Dict:
        result = {
            "url": url, "archive_url": archive_url, "category": category,
            "total_weight": 0.0, "damage_weight": 0.0, "mds_score": None,
            "resource_breakdown": {}, "timestamp": datetime.utcnow().isoformat(), "note": ""
        }

        page = await context.new_page()
        page.set_default_timeout(TIMEOUT_MS)

        seen = set()  # dedupe by URL
        type_counts: Dict[str, int] = {}
        type_failed: Dict[str, int] = {}

        async def record(res_url: str, res_type: str, status: int, failed: bool):
            u = res_url.split("#", 1)[0]
            if any(sn in u for sn in IGNORED_HOST_SNIPPETS):
                return
            if u in seen:
                return
            seen.add(u)
            key = res_type if res_type in RESOURCE_WEIGHTS else "other"
            type_counts[key] = type_counts.get(key, 0) + 1
            if failed:
                type_failed[key] = type_failed.get(key, 0) + 1
                result["damage_weight"] += RESOURCE_WEIGHTS[key]
            result["total_weight"] += RESOURCE_WEIGHTS[key]

        async def on_response(response):
            try:
                req = response.request
                await record(req.url, req.resource_type, response.status, response.status >= 400 or response.status == 0)
            except Exception:
                pass

        async def on_failed(request):
            try:
                await record(request.url, request.resource_type, 0, True)
            except Exception:
                pass

        page.on("response", on_response)
        page.on("requestfailed", on_failed)

        try:
            await page.goto(archive_url, wait_until="networkidle", timeout=TIMEOUT_MS)
        except Exception:
            pass

        await page.wait_for_timeout(EXTRA_WAIT_MS)
        await page.close()

        if result["total_weight"] > 0:
            result["mds_score"] = result["damage_weight"] / result["total_weight"]

        result["resource_breakdown"] = {
            k: {"total": type_counts.get(k, 0), "failed": type_failed.get(k, 0), "weight": RESOURCE_WEIGHTS[k]}
            for k in sorted(set(type_counts) | set(type_failed))
        }
        return result

    async def analyze_urls(self, rows: List[Dict]) -> List[Dict]:
        pw, browser, context = await _setup_browser()
        session = requests.Session()
        try:
            for i, r in enumerate(rows, 1):
                url = r.get("url", "").strip()
                category = r.get("category", "unknown").strip().lower()
                if not url:
                    continue
                print(f"\n[{i}/{len(rows)}] {url} ({category})")

                archive_url = _wayback_available_id(url, session=session)
                if not archive_url:
                    self.results.append({
                        "url": url, "archive_url": None, "category": category,
                        "total_weight": None, "damage_weight": None, "mds_score": None,
                        "resource_breakdown": {}, "timestamp": datetime.utcnow().isoformat(),
                        "note": "no-archive"
                    })
                    continue

                result = await self._measure_one(url, archive_url, category, context)
                self.results.append(result)
                await asyncio.sleep(1.0)
        finally:
            await context.close()
            await browser.close()
            await pw.stop()

        self._save_results()
        return self.results

    def _save_results(self):
        out_csv = OUT_DIR / "mds_results.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["url", "category", "archive_url", "mds_score", "total_weight",
                            "damage_weight", "timestamp", "note"],
            )
            writer.writeheader()
            for row in self.results:
                wrow = {k: v for k, v in row.items() if k not in ("resource_breakdown",)}
                writer.writerow(wrow)
        (OUT_DIR / "mds_breakdown.json").write_text(json.dumps(self.results, indent=2), encoding="utf-8")
        print(f"Results saved to {out_csv}")

    def summarize_and_plot(self):
        cats = sorted({r["category"] for r in self.results})
        stats = {}
        for c in cats:
            rows = [r for r in self.results if r["category"] == c]
            scores = [r["mds_score"] for r in rows if r["mds_score"] is not None]
            stats[c] = {
                "avg_mds": float(np.mean(scores)) if scores else None,
                "min_mds": float(np.min(scores)) if scores else None,
                "max_mds": float(np.max(scores)) if scores else None,
                "count": len(scores),
                "no_data": sum(1 for r in rows if r["mds_score"] is None),
            }

        print("\n" + "="*60)
        print("MEMENTO DAMAGE SCORE (lower is better)")
        print("="*60)
        for c, s in stats.items():
            print(f"\n{c.upper()}: valid={s['count']} no-data={s['no_data']}")
            if s["count"]:
                print(f"  Avg {s['avg_mds']:.3f}  Range {s['min_mds']:.3f}-{s['max_mds']:.3f}")
            else:
                print("  (no measurable pages)")

        valid_cats = [c for c in stats if stats[c]["count"]]
        if not valid_cats:
            return
        vals = [stats[c]["avg_mds"] for c in valid_cats]
        colors = ["green" if v < 0.3 else "orange" if v < 0.6 else "red" for v in vals]
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(valid_cats, vals, color=colors)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Average MDS (lower is better)")
        ax.set_title("Average Memento Damage Score by Category")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")
        fig.tight_layout()
        out_png = GRAPH_DIR / "mds_analysis.png"
        fig.savefig(out_png, dpi=160)
        print(f"\nGraph saved to {out_png}")


async def main():
    print("\n" + "="*60)
    print("MEMENTO DAMAGE SCORE (MDS)")
    print("="*60)
    urls_csv = Path("../data/urls.csv")
    if not urls_csv.exists():
        print("ERROR: ../data/urls.csv not found.")
        return
    with urls_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    analyzer = MementoDamageAnalyzer()
    await analyzer.analyze_urls(rows)
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
