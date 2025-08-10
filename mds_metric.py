"""
Memento Damage Score (MDS) Metric
Weights missing resources by their importance to user experience
Based on Brunelle et al., 2014 research
"""

import asyncio
import csv
import json
import logging
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import numpy as np

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mds_metric.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Resource importance weights (based on Brunelle et al., 2014)
RESOURCE_WEIGHTS = {
    'document': 1.0,      # HTML documents
    'stylesheet': 0.9,    # CSS - critical for layout
    'script': 0.8,        # JavaScript - functionality
    'image': 0.5,         # Images - visual content
    'font': 0.4,          # Fonts - typography
    'media': 0.3,         # Audio/Video
    'xhr': 0.6,           # API calls
    'fetch': 0.6,         # API calls
    'other': 0.1          # Ads, tracking, etc.
}

# Use existing archive URLs from previous runs
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


class MementoDamageAnalyzer:
    def __init__(self):
        self.output_dir = Path("outputs")
        self.results_file = self.output_dir / "mds_results.csv"
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
    
    async def measure_damage(self, url: str, archive_url: str, category: str, context) -> Dict:
        """Calculate Memento Damage Score with weighted resources"""
        result = {
            'url': url,
            'category': category,
            'total_weight': 0.0,
            'damage_weight': 0.0,
            'mds_score': 0.0,
            'resource_breakdown': {},
            'timestamp': datetime.now().isoformat()
        }
        
        page = None
        resources_data = []
        
        try:
            page = await context.new_page()
            
            # Track all resources with their types
            async def handle_response(response):
                try:
                    request = response.request
                    resource_type = request.resource_type
                    status = response.status
                    
                    # Map to weight categories
                    weight_key = resource_type if resource_type in RESOURCE_WEIGHTS else 'other'
                    weight = RESOURCE_WEIGHTS[weight_key]
                    
                    resources_data.append({
                        'url': request.url,
                        'type': resource_type,
                        'status': status,
                        'weight': weight,
                        'failed': status >= 400 or status == 0
                    })
                except:
                    pass
            
            async def handle_request_failed(request):
                try:
                    resource_type = request.resource_type
                    weight_key = resource_type if resource_type in RESOURCE_WEIGHTS else 'other'
                    weight = RESOURCE_WEIGHTS[weight_key]
                    
                    resources_data.append({
                        'url': request.url,
                        'type': resource_type,
                        'status': 0,
                        'weight': weight,
                        'failed': True
                    })
                except:
                    pass
            
            page.on('response', handle_response)
            page.on('requestfailed', handle_request_failed)
            
            timeout = 30000 if category == 'html' else 45000
            page.set_default_timeout(timeout)
            
            print(f"  Analyzing damage for {category.upper()}: {url}")
            
            try:
                await page.goto(archive_url, wait_until='networkidle', timeout=timeout)
            except:
                pass
            
            await page.wait_for_timeout(3000)
            
            # Calculate weighted damage
            type_counts = {}
            type_damages = {}
            
            for resource in resources_data:
                res_type = resource['type']
                weight = resource['weight']
                
                # Track totals
                result['total_weight'] += weight
                
                # Track by type
                if res_type not in type_counts:
                    type_counts[res_type] = 0
                    type_damages[res_type] = 0
                
                type_counts[res_type] += 1
                
                if resource['failed']:
                    result['damage_weight'] += weight
                    type_damages[res_type] += 1
            
            # Calculate MDS (0 = perfect, 1 = completely damaged)
            if result['total_weight'] > 0:
                result['mds_score'] = result['damage_weight'] / result['total_weight']
            
            # Store breakdown
            for res_type in type_counts:
                result['resource_breakdown'][res_type] = {
                    'total': type_counts[res_type],
                    'failed': type_damages[res_type],
                    'weight': RESOURCE_WEIGHTS.get(res_type, 0.1)
                }
            
            # Log result
            if result['mds_score'] < 0.2:
                print(f"  [GOOD] Low damage: {result['mds_score']:.3f}")
            elif result['mds_score'] < 0.5:
                print(f"  [MEDIUM] Moderate damage: {result['mds_score']:.3f}")
            else:
                print(f"  [HIGH] High damage: {result['mds_score']:.3f}")
            
        except Exception as e:
            logger.error(f"Error: {e}")
        finally:
            if page:
                await page.close()
        
        return result
    
    async def analyze_urls(self, urls: List[Dict]) -> List[Dict]:
        """Analyze multiple URLs"""
        playwright, browser, context = await self.setup_browser()
        
        try:
            for idx, url_data in enumerate(urls, 1):
                # Load from CSV or use defaults
                if isinstance(url_data, dict):
                    url = url_data.get('url', '')
                    category = url_data.get('category', 'unknown')
                else:
                    url = url_data
                    category = 'unknown'
                
                print(f"\n[{idx}/{len(urls)}] Processing {url}")
                
                archive_url = ARCHIVE_URLS.get(url)
                if not archive_url:
                    # Try to get from Wayback API
                    import requests
                    from urllib.parse import quote
                    try:
                        api_url = f"https://archive.org/wayback/available?url={quote(url)}"
                        resp = requests.get(api_url, timeout=10)
                        data = resp.json()
                        if 'archived_snapshots' in data and 'closest' in data['archived_snapshots']:
                            archive_url = data['archived_snapshots']['closest']['url']
                            if archive_url:
                                archive_url = archive_url.replace('/http', 'id_/http')
                                print(f"  Found archive: {archive_url[:80]}...")
                    except:
                        pass
                
                if not archive_url:
                    print(f"  Skipping - no archive URL")
                    continue
                
                result = await self.measure_damage(url, archive_url, category, context)
                self.results.append(result)
                
                await asyncio.sleep(1)
                
        finally:
            await context.close()
            await browser.close()
            await playwright.stop()
        
        self.save_results()
        return self.results
    
    def save_results(self):
        """Save results to CSV"""
        if not self.results:
            return
        
        with open(self.results_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['url', 'category', 'mds_score', 'total_weight', 
                         'damage_weight', 'timestamp']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                row = {k: v for k, v in result.items() if k != 'resource_breakdown'}
                writer.writerow(row)
        
        print(f"\nResults saved to {self.results_file}")
        
        # Save detailed breakdown
        breakdown_file = self.output_dir / "mds_breakdown.json"
        with open(breakdown_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Detailed breakdown saved to {breakdown_file}")
    
    def calculate_and_visualize(self):
        """Create visualizations and statistics"""
        if not self.results:
            return
        
        # Calculate stats by category
        stats = {}
        for cat in ['html', 'spa', 'social']:
            cat_results = [r for r in self.results if r['category'] == cat]
            if cat_results:
                stats[cat] = {
                    'avg_mds': np.mean([r['mds_score'] for r in cat_results]),
                    'min_mds': np.min([r['mds_score'] for r in cat_results]),
                    'max_mds': np.max([r['mds_score'] for r in cat_results]),
                    'count': len(cat_results)
                }
        
        # Print results
        print("\n" + "="*60)
        print("MEMENTO DAMAGE SCORE RESULTS")
        print("="*60)
        for cat, s in stats.items():
            print(f"\n{cat.upper()}:")
            print(f"  Average MDS: {s['avg_mds']:.3f}")
            print(f"  Range: {s['min_mds']:.3f} - {s['max_mds']:.3f}")
            print(f"  Sites analyzed: {s['count']}")
        
        # Create visualization
        if stats:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Bar chart of average MDS
            categories = list(stats.keys())
            mds_scores = [stats[c]['avg_mds'] for c in categories]
            
            colors = ['green' if s < 0.3 else 'orange' if s < 0.6 else 'red' for s in mds_scores]
            bars = ax1.bar(categories, mds_scores, color=colors)
            
            ax1.set_ylabel('MDS Score (lower is better)')
            ax1.set_title('Average Memento Damage Score by Category')
            ax1.set_ylim(0, 1)
            
            for bar, score in zip(bars, mds_scores):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', fontweight='bold')
            
            # Resource type breakdown
            resource_types = {}
            for result in self.results:
                for res_type, data in result.get('resource_breakdown', {}).items():
                    if res_type not in resource_types:
                        resource_types[res_type] = {'total': 0, 'failed': 0}
                    resource_types[res_type]['total'] += data['total']
                    resource_types[res_type]['failed'] += data['failed']
            
            if resource_types:
                types = list(resource_types.keys())[:6]  # Top 6 types
                failures = [resource_types[t]['failed'] / resource_types[t]['total'] * 100 
                           if resource_types[t]['total'] > 0 else 0 for t in types]
                
                ax2.barh(types, failures)
                ax2.set_xlabel('Failure Rate (%)')
                ax2.set_title('Resource Failure Rates by Type')
                
            plt.suptitle('Memento Damage Score Analysis', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            graph_path = self.graphs_dir / 'mds_analysis.png'
            plt.savefig(graph_path, dpi=150)
            print(f"\nGraph saved to {graph_path}")
            plt.show()
        
        # Save statistics
        stats_file = self.output_dir / "mds_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to {stats_file}")


async def main():
    print("\n" + "="*60)
    print("MEMENTO DAMAGE SCORE (MDS) ANALYSIS")
    print("="*60)
    print("Measuring weighted resource damage in archived pages")
    print("="*60)
    
    # Load URLs from CSV
    urls_file = Path("data/urls.csv")
    if urls_file.exists():
        print(f"Loading URLs from {urls_file}")
        with open(urls_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            test_urls = list(reader)
    else:
        print("ERROR: data/urls.csv not found!")
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