"""
Broken Link Density (BLD) Metric - Improved Version
- URL normalization for better matching
- Fallback to Wayback API if not in hardcoded mapping
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
from typing import Dict, List
import time
from urllib.parse import urlparse

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bld_metric.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Hardcoded archive URLs (normalized keys)
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

def norm_url(u: str) -> str:
    """Normalize URL for matching in mapping."""
    u = u.strip().lower().rstrip('/')
    u = u.replace('://www.', '://')
    return u

def get_wayback_url(url: str) -> str:
    """Get latest snapshot from Wayback if not in mapping."""
    try:
        r = requests.get(
            "https://archive.org/wayback/available",
            params={"url": url},
            timeout=10
        )
        data = r.json()
        snap = data.get("archived_snapshots", {}).get("closest", {})
        if snap.get("available") and snap.get("url"):
            return snap["url"].replace("/web/", "/web/").replace("/http", "id_/http")
    except Exception as e:
        logger.error(f"Wayback fallback failed for {url}: {e}")
    return None

class BrokenLinkDensityAnalyzer:
    def __init__(self):
        self.output_dir = Path("outputs/bld")
        self.results_file = self.output_dir / "bld_results.csv"
        self.graphs_dir = self.output_dir / "graphs"
        self.output_dir.mkdir(exist_ok=True)
        self.graphs_dir.mkdir(exist_ok=True)
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

            # Track responses
            async def handle_response(response):
                try:
                    r_url = response.url
                    if self.is_relevant_resource(url, r_url):
                        resources_data.append({
                            'url': r_url,
                            'status': response.status,
                            'failed': response.status >= 400 or response.status == 0
                        })
                except:
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
                except:
                    pass

            page.on('response', handle_response)
            page.on('requestfailed', handle_request_failed)

            timeout = 25000
            page.set_default_timeout(timeout)

            print(f"  Loading {category.upper()}: {url}")
            try:
                await page.goto(archive_url, wait_until='load', timeout=timeout)
                await page.wait_for_timeout(5000)
            except:
                pass

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

        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
        finally:
            if page:
                await page.close()
        return result

    def is_relevant_resource(self, original_url: str, resource_url: str) -> bool:
        """Filter out Wayback static/analytics resources."""
        parsed = urlparse(resource_url)
        if 'web.archive.org' in parsed.netloc and '/_static/' in parsed.path:
            return False
        if 'analytics' in parsed.path or 'googletagmanager' in parsed.netloc:
            return False
        return True

    async def analyze_urls(self, urls: List[Dict]) -> List[Dict]:
        playwright, browser, context = await self.setup_browser()
        try:
            for idx, url_data in enumerate(urls, 1):
                url = url_data['url']
                category = url_data['category']
                print(f"\n[{idx}/{len(urls)}] Processing {url}")
                key = norm_url(url)
                archive_url = ARCHIVE_URLS.get(key)
                if not archive_url:
                    archive_url = get_wayback_url(url)
                if not archive_url:
                    print("  [NO DATA] No archive URL available.")
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

    def calculate_and_visualize(self):
        if not self.results:
            return
        stats = {}
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
                    'error_rate': 100 * (len(cat_results) - len(valid)) / len(cat_results)
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
        if any(s['avg_bld'] is not None for s in stats.values()):
            fig, ax = plt.subplots(figsize=(10, 6))
            categories = [c for c in stats if stats[c]['avg_bld'] is not None]
            bld_scores = [stats[c]['avg_bld'] for c in categories]
            colors = ['green' if s < 0.3 else 'orange' if s < 0.6 else 'red' for s in bld_scores]
            bars = ax.bar(categories, bld_scores, color=colors)
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
        stats_file = self.output_dir / "bld_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to {stats_file}")

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
    else:
        print("ERROR: data/urls.csv not found!")
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
