"""
Memento Damage Score (MDS) Metric — fixed & runnable
- Weights missing resources by their importance to UX (Brunelle et al., 2014 inspired)
- Robust Wayback lookup (tries http/https + www/non-www variants)
- Ignores Wayback-injected subresources (counts only the archived page’s own assets)
- Weighted breakdown per resource type
- Saves CSV + JSON + PNG chart in outputs/mds/
"""

import asyncio
import csv
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import matplotlib.pyplot as plt

# NOTE: these are sync libs used outside the hot path to avoid blocking the event loop
import requests

# Windows event loop policy
if sys.platform == 'win32':
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mds_metric.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------- Tunables ----------
# Resource importance weights (based on Brunelle et al., 2014)
RESOURCE_WEIGHTS = {
    'document': 1.0,
    'stylesheet': 0.9,
    'script': 0.8,
    'image': 0.5,
    'font': 0.4,
    'media': 0.3,
    'xhr': 0.6,
    'fetch': 0.6,
    'other': 0.1
}

# Some likely-blocked publishers (Wayback robots/exclusions). Used for nicer notes.
LIKELY_BLOCKED = {
    "nytimes.com", "reuters.com", "bloomberg.com", "ft.com", "wsj.com",
    "economist.com", "foxnews.com", "cbsnews.com", "nbcnews.com",
    "apnews.com", "businessinsider.com", "theguardian.com"
}

# Optional hardcoded fallbacks (useful for demos)
ARCHIVE_URLS = {
    'https://www.w3.org': 'https://web.archive.org/web/20250806070354id_/https://www.w3.org/',
    'https://example.com': 'https://web.archive.org/web/20250809095118id_/https://example.com/',
    'https://www.wikipedia.org': 'https://web.archive.org/web/20250809053232id_/https://www.wikipedia.org/',
    'https://www.bbc.com/news': 'https://web.archive.org/web/20250809010155id_/https://www.bbc.com/news',
    'https://archive.org': 'https://web.archive.org/web/20250730042019id_/https://archive.org/',
    'https://github.com': 'https://web.archive.org/web/20250809065555id_/https://github.com/',
    'https://discord.com': 'https://web.archive.org/web/20250808001155id_/https://discord.com/',
    'https://www.notion.so': 'https://web.archive.org/web/20250728204801id_/https://www.notion.so/',
    'https://twitter.com': 'https://web.archive.org/web/20250729210226id_/https://twitter.com/',
    'https://www.reddit.com': 'https://web.archive.org/web/20250809031420id_/https://www.reddit.com/',
    'https://www.facebook.com': 'https://web.archive.org/web/20250809011211id_/https://www.facebook.com/',
    'https://www.instagram.com': 'https://web.archive.org/web/20250809075255id_/https://www.instagram.com/',
}

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")

# ---------- Helpers ----------

def _domain(hostname: Optional[str]) -> str:
    return (hostname or "").lower()

def _is_wayback_injected(url: str) -> bool:
    host = _domain(urlsplit(url).hostname)
    return host.endswith("web.archive.org")

def _normalize_for_lookup(url: str) -> List[str]:
    """
    Generate http/https + www/non-www variants for best Wayback hit chance.
    """
    s = urlsplit(url)
    host = s.hostname or ""
    hosts = {host}
    if host.startswith("www."):
        hosts.add(host[4:])
    else:
        hosts.add(f"www.{host}" if host else host)

    variants = []
    for scheme in ("https", "http"):
        for h in hosts:
            if not h:
                continue
            path = s.path or "/"
            variants.append(urlunsplit((scheme, h, path, s.query, s.fragment)))
    # Make unique while preserving order
    seen = set()
    ordered = []
    for v in variants:
        if v not in seen:
            ordered.append(v)
            seen.add(v)
    return ordered

def resolve_wayback(url: str) -> Optional[str]:
    """
    Try Archive.org Wayback 'available' endpoint with multiple URL variants.
    Returns a snapshot URL with id_/ scheme to keep subresources in the same capture.
    """
    headers = {"User-Agent": USER_AGENT}
    for candidate in _normalize_for_lookup(url):
        try:
            r = requests.get(
                "https://archive.org/wayback/available",
                params={"url": candidate},
                timeout=12,
                headers=headers
            )
            c = r.json().get("archived_snapshots", {}).get("closest")
            if c and c.get("url"):
                u = c["url"]
                # Force id_ mode so subresources load from same timestamp
                u = u.replace("/http", "id_/http").replace("/https", "id_/https")
                return u
        except Exception:
            continue
    return None

def fallback_archive_map(url: str) -> Optional[str]:
    """
    Check ARCHIVE_URLS with small normalization.
    """
    norm = url.rstrip("/")
    alt = None
    if norm.startswith("http://"):
        alt = "https://" + norm[len("http://"):]
    elif norm.startswith("https://"):
        alt = "http://" + norm[len("https://"):]
    return ARCHIVE_URLS.get(norm) or (ARCHIVE_URLS.get(alt) if alt else None)

def guess_blocked_note(url: str) -> str:
    host = _domain(urlsplit(url).hostname)
    for b in LIKELY_BLOCKED:
        if host.endswith(b):
            return "Likely excluded via robots"
    return "No archive found"

# ---------- Core Analyzer ----------

class MementoDamageAnalyzer:
    def __init__(self):
        self.output_dir = Path("outputs/mds")
        self.results_file = self.output_dir / "mds_results.csv"
        self.graphs_dir = self.output_dir / "graphs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.graphs_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict] = []

    async def setup_browser(self):
        from playwright.async_api import async_playwright
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            ignore_https_errors=True,
            user_agent=USER_AGENT,
        )
        return playwright, browser, context

    async def measure_damage(self, url: str, archive_url: str, category: str, context) -> Dict:
        """
        Calculate Memento Damage Score with weighted resources.
        Excludes Wayback-injected subresources so we only judge the archived page.
        """
        result = {
            'url': url,
            'category': category,
            'total_weight': 0.0,
            'damage_weight': 0.0,
            'mds_score': 0.0,
            'resource_breakdown': {},
            'timestamp': datetime.now().isoformat(),
            'note': ''
        }

        page = None
        resources_data = []

        # Per-type tracking (counts + weight)
        type_counts = {}
        type_damages = {}
        type_weight_total = {}
        type_weight_failed = {}

        try:
            page = await context.new_page()

            async def handle_response(response):
                try:
                    request = response.request
                    if _is_wayback_injected(request.url) and request.resource_type != 'document':
                        return
                    resource_type = request.resource_type
                    weight_key = resource_type if resource_type in RESOURCE_WEIGHTS else 'other'
                    weight = float(RESOURCE_WEIGHTS[weight_key])
                    status = response.status
                    failed = (status >= 400 or status == 0)
                    resources_data.append({
                        'url': request.url,
                        'type': resource_type,
                        'status': status,
                        'weight': weight,
                        'failed': failed
                    })
                except Exception:
                    pass

            async def handle_request_failed(request):
                try:
                    if _is_wayback_injected(request.url) and request.resource_type != 'document':
                        return
                    resource_type = request.resource_type
                    weight_key = resource_type if resource_type in RESOURCE_WEIGHTS else 'other'
                    weight = float(RESOURCE_WEIGHTS[weight_key])
                    resources_data.append({
                        'url': request.url,
                        'type': resource_type,
                        'status': 0,
                        'weight': weight,
                        'failed': True
                    })
                except Exception:
                    pass

            page.on('response', handle_response)
            page.on('requestfailed', handle_request_failed)

            timeout = 45000
            page.set_default_timeout(timeout)

            print(f"  Analyzing damage for {category.upper()}: {url}")
            try:
                # 'domcontentloaded' is more reliable for Wayback; then small settle wait
                await page.goto(archive_url, wait_until='domcontentloaded', timeout=timeout)
            except Exception:
                # even if goto errors (e.g., partial), still try to collect what we got
                pass

            await page.wait_for_timeout(3000)

            # Aggregate weights + counts
            for resource in resources_data:
                res_type = resource['type']
                weight = float(resource['weight'])

                # totals
                result['total_weight'] += weight
                type_counts[res_type] = type_counts.get(res_type, 0) + 1
                type_weight_total[res_type] = type_weight_total.get(res_type, 0.0) + weight

                if resource['failed']:
                    result['damage_weight'] += weight
                    type_damages[res_type] = type_damages.get(res_type, 0) + 1
                    type_weight_failed[res_type] = type_weight_failed.get(res_type, 0.0) + weight

            if result['total_weight'] > 0:
                result['mds_score'] = result['damage_weight'] / result['total_weight']

            for res_type in set(list(type_counts.keys()) + list(type_weight_total.keys())):
                result['resource_breakdown'][res_type] = {
                    'total': type_counts.get(res_type, 0),
                    'failed': type_damages.get(res_type, 0),
                    'weight_total': float(type_weight_total.get(res_type, 0.0)),
                    'weight_failed': float(type_weight_failed.get(res_type, 0.0)),
                    'weight': float(RESOURCE_WEIGHTS.get(res_type, 0.1))
                }

        except Exception as e:
            logger.error(f"Error in measure_damage: {e}")
        finally:
            if page:
                try:
                    await page.close()
                except Exception:
                    pass

        return result

    async def analyze_urls(self, urls: List[Dict]) -> List[Dict]:
        playwright, browser, context = await self.setup_browser()
        try:
            for idx, url_data in enumerate(urls, 1):
                raw_url = url_data.get('url', '').strip()
                category = (url_data.get('category', 'unknown') or 'unknown').lower().strip()

                if not raw_url:
                    continue

                print(f"\n[{idx}/{len(urls)}] Processing {raw_url}")

                # 1) Wayback API with smart variants
                archive_url = resolve_wayback(raw_url)

                # 2) Hardcoded fallbacks
                if not archive_url:
                    archive_url = fallback_archive_map(raw_url)
                    if archive_url:
                        print(f"  Using fallback archive: {archive_url[:80]}...")

                # 3) If still nothing, record as no-data
                if not archive_url:
                    note = guess_blocked_note(raw_url)
                    print(f"  {note} - recording no-data (MDS=None)")
                    result = {
                        'url': raw_url,
                        'category': category,
                        'total_weight': None,
                        'damage_weight': None,
                        'mds_score': None,
                        'resource_breakdown': {},
                        'timestamp': datetime.now().isoformat(),
                        'note': note
                    }
                    self.results.append(result)
                    continue

                # Measure the page
                result = await self.measure_damage(raw_url, archive_url, category, context)
                self.results.append(result)
                # be polite to Wayback
                await asyncio.sleep(0.6)

        finally:
            try:
                await context.close()
            except Exception:
                pass
            try:
                await browser.close()
            except Exception:
                pass
            try:
                await playwright.stop()
            except Exception:
                pass

        self.save_results()
        return self.results

    def save_results(self):
        if not self.results:
            return
        # CSV (flat)
        with open(self.results_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['url', 'category', 'mds_score', 'total_weight',
                          'damage_weight', 'timestamp', 'note']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                row = {k: v for k, v in result.items() if k != 'resource_breakdown'}
                writer.writerow(row)
        print(f"\nResults saved to {self.results_file}")

        # JSON (detailed)
        breakdown_file = self.output_dir / "mds_breakdown.json"
        with open(breakdown_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Detailed breakdown saved to {breakdown_file}")

    def calculate_and_visualize(self):
        if not self.results:
            return
        stats = {}
        categories_all = sorted({r['category'] for r in self.results})
        for cat in categories_all:
            cat_results = [r for r in self.results if r['category'] == cat]
            cat_scores = [r['mds_score'] for r in cat_results if r['mds_score'] is not None]
            na_count = sum(1 for r in cat_results if r['mds_score'] is None)
            # weighted global mean (by total_weight) also useful:
            weighted_sum = sum((r['mds_score'] or 0) * (r['total_weight'] or 0) for r in cat_results if r['mds_score'] is not None)
            total_w = sum((r['total_weight'] or 0) for r in cat_results if r['mds_score'] is not None)
            stats[cat] = {
                'avg_mds': float(np.mean(cat_scores)) if cat_scores else None,
                'min_mds': float(np.min(cat_scores)) if cat_scores else None,
                'max_mds': float(np.max(cat_scores)) if cat_scores else None,
                'count': len(cat_scores),
                'no_data': na_count,
                'weighted_mean_mds': (weighted_sum / total_w) if total_w > 0 else None
            }

        print("\n" + "="*60)
        print("MEMENTO DAMAGE SCORE RESULTS")
        print("="*60)
        for cat in categories_all:
            s = stats.get(cat, {})
            print(f"\n{cat.upper()}:")
            print(f"  Sites with data: {s.get('count', 0)}")
            print(f"  No-archive (excluded): {s.get('no_data', 0)}")
            if s.get('count', 0) > 0:
                print(f"  Average MDS: {s['avg_mds']:.3f}")
                if s.get('weighted_mean_mds') is not None:
                    print(f"  Weighted mean MDS: {s['weighted_mean_mds']:.3f}")
                print(f"  Range: {s['min_mds']:.3f} - {s['max_mds']:.3f}")
            else:
                print("  (no measurable pages)")

        # Visualization
        if any((stats[c]['count'] or 0) > 0 for c in categories_all):
            # chart 1: average MDS per category
            cats_with_data = [c for c in categories_all if (stats[c]['count'] or 0) > 0]
            mds_scores = [stats[c]['avg_mds'] for c in cats_with_data]

            fig, ax1 = plt.subplots(1, 1, figsize=(9, 6))
            bars = ax1.bar(cats_with_data, mds_scores)
            ax1.set_ylabel('MDS Score (lower is better)')
            ax1.set_title('Average Memento Damage Score by Category')
            ax1.set_ylim(0, 1)
            for bar, score in zip(bars, mds_scores):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                         f'{score:.3f}', ha='center', fontweight='bold')

            plt.tight_layout()
            graph_path = self.graphs_dir / 'mds_analysis.png'
            plt.savefig(graph_path, dpi=150)
            print(f"\nGraph saved to {graph_path}")

        stats_file = self.output_dir / "mds_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to {stats_file}")

# ---------- CLI ----------

def load_urls_csv() -> List[Dict]:
    """
    Tries ./urls.csv, then ./data/urls.csv, then ../data/urls.csv
    """
    candidates = [Path("urls.csv"), Path("data/urls.csv"), Path("../data/urls.csv")]
    for p in candidates:
        if p.exists():
            print(f"Loading URLs from {p}")
            with open(p, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            return rows
    print("ERROR: urls.csv not found in current folder, ./data or ../data")
    return []

async def main():
    print("\n" + "="*60)
    print("MEMENTO DAMAGE SCORE (MDS) ANALYSIS")
    print("="*60)
    print("Measuring weighted resource damage in archived pages")
    print("="*60)

    test_urls = load_urls_csv()
    if not test_urls:
        return

    analyzer = MementoDamageAnalyzer()
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
