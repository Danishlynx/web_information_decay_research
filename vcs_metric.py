"""
Visual Correspondence Score (VCS) Metric
Compares visual similarity between live and archived pages
Based on Reyes Ayala, 2019 research using SSIM algorithm
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
from typing import Dict, List, Tuple
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import io

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

# Archive URLs for testing
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
}


class VisualCorrespondenceAnalyzer:
    def __init__(self):
        self.output_dir = Path("outputs")
        self.results_file = self.output_dir / "vcs_results.csv"
        self.graphs_dir = self.output_dir / "graphs"
        self.screenshots_dir = self.output_dir / "vcs_screenshots"
        
        self.output_dir.mkdir(exist_ok=True)
        self.graphs_dir.mkdir(exist_ok=True)
        self.screenshots_dir.mkdir(exist_ok=True)
        
        self.results = []
    
    async def setup_browser(self):
        from playwright.async_api import async_playwright
        
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={'width': 1280, 'height': 720},
            ignore_https_errors=True
        )
        return playwright, browser, context
    
    def calculate_ssim(self, img1_bytes: bytes, img2_bytes: bytes) -> float:
        """Calculate SSIM between two images"""
        try:
            # Convert bytes to PIL Images
            img1 = Image.open(io.BytesIO(img1_bytes))
            img2 = Image.open(io.BytesIO(img2_bytes))
            
            # Convert to RGB if needed
            if img1.mode != 'RGB':
                img1 = img1.convert('RGB')
            if img2.mode != 'RGB':
                img2 = img2.convert('RGB')
            
            # Resize to same dimensions (use smaller size)
            width = min(img1.width, img2.width)
            height = min(img1.height, img2.height)
            img1 = img1.resize((width, height), Image.Resampling.LANCZOS)
            img2 = img2.resize((width, height), Image.Resampling.LANCZOS)
            
            # Convert to numpy arrays
            arr1 = np.array(img1)
            arr2 = np.array(img2)
            
            # Calculate SSIM
            score = ssim(arr1, arr2, channel_axis=2, data_range=255)
            
            return score
            
        except Exception as e:
            logger.error(f"SSIM calculation error: {e}")
            return 0.0
    
    async def capture_screenshots(self, url: str, archive_url: str, category: str, context) -> Tuple[bytes, bytes]:
        """Capture screenshots of live and archived pages"""
        live_screenshot = None
        archive_screenshot = None
        
        # Capture archived page
        page = None
        try:
            page = await context.new_page()
            timeout = 30000 if category == 'html' else 45000
            page.set_default_timeout(timeout)
            
            print(f"  Capturing archived page screenshot...")
            await page.goto(archive_url, wait_until='networkidle', timeout=timeout)
            await page.wait_for_timeout(3000)
            archive_screenshot = await page.screenshot(full_page=False)
            
        except Exception as e:
            logger.error(f"Archive screenshot error: {e}")
        finally:
            if page:
                await page.close()
        
        # Capture live page
        page = None
        try:
            page = await context.new_page()
            page.set_default_timeout(30000)
            
            print(f"  Capturing live page screenshot...")
            await page.goto(url, wait_until='networkidle', timeout=30000)
            await page.wait_for_timeout(3000)
            live_screenshot = await page.screenshot(full_page=False)
            
        except Exception as e:
            logger.error(f"Live screenshot error: {e}")
        finally:
            if page:
                await page.close()
        
        return live_screenshot, archive_screenshot
    
    async def measure_vcs(self, url: str, archive_url: str, category: str, context) -> Dict:
        """Measure Visual Correspondence Score"""
        result = {
            'url': url,
            'category': category,
            'vcs_score': 0.0,
            'has_live': False,
            'has_archive': False,
            'error': None,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            print(f"  Measuring visual correspondence for {category.upper()}: {url}")
            
            # Capture screenshots
            live_screenshot, archive_screenshot = await self.capture_screenshots(
                url, archive_url, category, context
            )
            
            if live_screenshot:
                result['has_live'] = True
                # Save live screenshot
                live_path = self.screenshots_dir / f"{url.replace('://', '_').replace('/', '_')}_live.png"
                with open(live_path, 'wb') as f:
                    f.write(live_screenshot)
            
            if archive_screenshot:
                result['has_archive'] = True
                # Save archive screenshot
                archive_path = self.screenshots_dir / f"{url.replace('://', '_').replace('/', '_')}_archive.png"
                with open(archive_path, 'wb') as f:
                    f.write(archive_screenshot)
            
            # Calculate SSIM if both screenshots available
            if live_screenshot and archive_screenshot:
                result['vcs_score'] = self.calculate_ssim(live_screenshot, archive_screenshot)
                
                # Log result
                if result['vcs_score'] > 0.8:
                    print(f"  [EXCELLENT] High similarity: {result['vcs_score']:.3f}")
                elif result['vcs_score'] > 0.5:
                    print(f"  [GOOD] Moderate similarity: {result['vcs_score']:.3f}")
                else:
                    print(f"  [POOR] Low similarity: {result['vcs_score']:.3f}")
            else:
                result['error'] = "Missing screenshots"
                print(f"  [ERROR] Could not capture both screenshots")
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"VCS measurement error: {e}")
        
        return result
    
    async def analyze_urls(self, urls: List[Dict]) -> List[Dict]:
        """Analyze multiple URLs"""
        playwright, browser, context = await self.setup_browser()
        
        try:
            for idx, url_data in enumerate(urls, 1):
                if isinstance(url_data, dict):
                    url = url_data.get('url', '')
                    category = url_data.get('category', 'unknown')
                else:
                    url = url_data
                    category = 'unknown'
                
                print(f"\n[{idx}/{len(urls)}] Processing {url}")
                
                archive_url = ARCHIVE_URLS.get(url)
                if not archive_url:
                    # Get from Wayback API
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
                    except:
                        pass
                
                if not archive_url:
                    print(f"  Skipping - no archive")
                    continue
                
                result = await self.measure_vcs(url, archive_url, category, context)
                self.results.append(result)
                
                await asyncio.sleep(2)
                
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
            writer = csv.DictWriter(f, fieldnames=['url', 'category', 'vcs_score', 
                                                   'has_live', 'has_archive', 'error', 'timestamp'])
            writer.writeheader()
            writer.writerows(self.results)
        
        print(f"\nResults saved to {self.results_file}")
    
    def calculate_and_visualize(self):
        """Create visualizations"""
        if not self.results:
            return
        
        # Filter valid results
        valid_results = [r for r in self.results if r['vcs_score'] > 0]
        
        if not valid_results:
            print("No valid VCS scores to visualize")
            return
        
        # Calculate stats by category
        stats = {}
        for cat in ['html', 'spa', 'social']:
            cat_results = [r for r in valid_results if r['category'] == cat]
            if cat_results:
                scores = [r['vcs_score'] for r in cat_results]
                stats[cat] = {
                    'avg_vcs': np.mean(scores),
                    'min_vcs': np.min(scores),
                    'max_vcs': np.max(scores),
                    'std_vcs': np.std(scores),
                    'count': len(cat_results)
                }
        
        # Print results
        print("\n" + "="*60)
        print("VISUAL CORRESPONDENCE SCORE RESULTS")
        print("="*60)
        for cat, s in stats.items():
            print(f"\n{cat.upper()}:")
            print(f"  Average VCS: {s['avg_vcs']:.3f}")
            print(f"  Range: {s['min_vcs']:.3f} - {s['max_vcs']:.3f}")
            print(f"  Std Dev: {s['std_vcs']:.3f}")
            print(f"  Sites analyzed: {s['count']}")
        
        # Create visualization
        if stats:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Bar chart of average VCS
            categories = list(stats.keys())
            vcs_scores = [stats[c]['avg_vcs'] for c in categories]
            
            colors = ['green' if s > 0.7 else 'orange' if s > 0.4 else 'red' for s in vcs_scores]
            bars = ax1.bar(categories, vcs_scores, color=colors)
            
            ax1.set_ylabel('VCS Score (1.0 = perfect match)')
            ax1.set_title('Average Visual Correspondence by Category')
            ax1.set_ylim(0, 1)
            
            for bar, score in zip(bars, vcs_scores):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', fontweight='bold')
            
            # Box plot showing distribution
            data_by_cat = []
            labels = []
            for cat in categories:
                cat_results = [r['vcs_score'] for r in valid_results if r['category'] == cat]
                if cat_results:
                    data_by_cat.append(cat_results)
                    labels.append(cat.upper())
            
            if data_by_cat:
                bp = ax2.boxplot(data_by_cat, labels=labels, patch_artist=True)
                ax2.set_ylabel('VCS Score')
                ax2.set_title('VCS Score Distribution by Category')
                ax2.set_ylim(0, 1)
                
                # Color the boxes
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
            
            plt.suptitle('Visual Correspondence Score Analysis', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            graph_path = self.graphs_dir / 'vcs_analysis.png'
            plt.savefig(graph_path, dpi=150)
            print(f"\nGraph saved to {graph_path}")
            plt.show()
        
        # Save statistics
        stats_file = self.output_dir / "vcs_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to {stats_file}")


async def main():
    print("\n" + "="*60)
    print("VISUAL CORRESPONDENCE SCORE (VCS) ANALYSIS")
    print("="*60)
    print("Comparing visual similarity between live and archived pages")
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
    
    analyzer = VisualCorrespondenceAnalyzer()
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