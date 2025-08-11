"""
Web Archival Quality Assessment Pipeline
Metric 1: Replayability Ratio (RR)
Comparative analysis across HTML, SPA, Social Media, and API-based websites
"""

import asyncio
import csv
import json
import logging
import sys
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote, urlparse
import time
import numpy as np

# Windows-specific asyncio configuration
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rr_metric.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Default test URLs categorized as per dissertation requirements
DEFAULT_TEST_URLS = [
    # Traditional HTML websites (static content)
    {'url': 'https://www.w3.org', 'category': 'html', 'id': 'html_1'},
    {'url': 'https://example.com', 'category': 'html', 'id': 'html_2'},
    {'url': 'https://www.wikipedia.org', 'category': 'html', 'id': 'html_3'},
    {'url': 'https://www.archive.org', 'category': 'html', 'id': 'html_4'},
    {'url': 'https://www.bbc.com/news', 'category': 'html', 'id': 'html_5'},

    # Single-Page Applications (JavaScript-heavy)
    {'url': 'https://github.com', 'category': 'spa', 'id': 'spa_1'},
    {'url': 'https://gmail.com', 'category': 'spa', 'id': 'spa_2'},
    {'url': 'https://discord.com', 'category': 'spa', 'id': 'spa_3'},
    {'url': 'https://www.notion.so', 'category': 'spa', 'id': 'spa_4'},
    {'url': 'https://web.whatsapp.com', 'category': 'spa', 'id': 'spa_5'},

    # Social Media Platforms
    {'url': 'https://twitter.com', 'category': 'social', 'id': 'social_1'},
    {'url': 'https://www.facebook.com', 'category': 'social', 'id': 'social_2'},
    {'url': 'https://www.instagram.com', 'category': 'social', 'id': 'social_3'},
    {'url': 'https://www.reddit.com', 'category': 'social', 'id': 'social_4'},
    {'url': 'https://www.linkedin.com', 'category': 'social', 'id': 'social_5'},

    # API-based/Dynamic Content websites
    {'url': 'https://api.github.com', 'category': 'api', 'id': 'api_1'},
    {'url': 'https://jsonplaceholder.typicode.com', 'category': 'api', 'id': 'api_2'},
    {'url': 'https://newsapi.org', 'category': 'api', 'id': 'api_3'},
    {'url': 'https://openweathermap.org/api', 'category': 'api', 'id': 'api_4'},
    {'url': 'https://developer.spotify.com/web-api', 'category': 'api', 'id': 'api_5'},
]


class ReplayabilityRatioAnalyzer:
    """
    Measures the Replayability Ratio (RR) of archived web pages across different platform types.
    """

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("outputs/rr")
        self.screenshots_dir = self.output_dir / "screenshots"
        self.graphs_dir = self.output_dir / "graphs"
        self.results_file = self.output_dir / "rr_results.csv"

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        self.graphs_dir.mkdir(parents=True, exist_ok=True)

        self.results = []
        logger.info(f"RR Analyzer initialized. Output directory: {self.output_dir}")

    def get_wayback_url(self, url: str) -> Optional[str]:
        """
        Get the actual archived URL from Wayback Machine API.
        """
        try:
            # Clean URL
            clean_url = url.strip()
            if not clean_url.startswith(('http://', 'https://')):
                clean_url = 'http://' + clean_url

            # Query Wayback Machine API for latest snapshot
            api_url = f"http://archive.org/wayback/available?url={quote(clean_url)}"

            response = requests.get(api_url, timeout=10)
            data = response.json()

            # Check if archived snapshots exist
            if 'archived_snapshots' in data and 'closest' in data['archived_snapshots']:
                snapshot = data['archived_snapshots']['closest']
                if snapshot.get('available'):
                    archive_url = snapshot['url']
                    timestamp = snapshot.get('timestamp', 'unknown')

                    # Use id_ format for better replay
                    if '/web/' in archive_url and not archive_url.endswith('_/'):
                        parts = archive_url.split('/web/')
                        if len(parts) == 2:
                            timestamp_and_url = parts[1]
                            timestamp = timestamp_and_url.split('/')[0]
                            original_url = '/'.join(timestamp_and_url.split('/')[1:])
                            archive_url = f"http://web.archive.org/web/{timestamp}id_/{original_url}"

                    logger.info(f"Found archive URL from {timestamp}: {archive_url}")
                    return archive_url

            logger.warning(f"No archive found for {url}")
            return None

        except Exception as e:
            logger.error(f"Failed to get archive URL for {url}: {e}")
            return None

    async def setup_browser(self):
        """
        Set up Playwright browser instance with optimized settings for different website types.
        """
        from playwright.async_api import async_playwright

        playwright = await async_playwright().start()

        browser = await playwright.chromium.launch(
            headless=True,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-web-security',
                '--disable-features=IsolateOrigins,site-per-process'
            ]
        )

        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            ignore_https_errors=True,
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            java_script_enabled=True  # Important for SPAs
        )

        return playwright, browser, context

    async def measure_replayability(self, url: str, archive_url: str, category: str, context) -> Dict:
        """
        Measure the replayability of a single archived URL with category-specific handling.
        """
        result = {
            'url': url,
            'archive_url': archive_url,
            'category': category,
            'status_code': None,
            'screenshot_size': 0,
            'dom_elements': 0,
            'load_time': 0,
            'rr_score': 0.0,
            'error': None,
            'timestamp': datetime.now().isoformat()
        }

        page = None
        start_time = time.time()

        try:
            page = await context.new_page()

            # Different timeouts for different categories
            timeout_map = {
                'html': 30000,   # 30 seconds for HTML
                'spa': 60000,    # 60 seconds for SPAs (need more time for JS)
                'social': 60000, # 60 seconds for social media
                'api': 20000     # 20 seconds for APIs
            }
            page.set_default_timeout(timeout_map.get(category, 45000))

            logger.info(f"Loading {category.upper()} archived page: {archive_url}")

            # Navigate with category-specific wait conditions
            wait_until = 'networkidle' if category in ['spa', 'social'] else 'domcontentloaded'

            response = await page.goto(
                archive_url,
                wait_until=wait_until,
                timeout=timeout_map.get(category, 45000)
            )

            if response:
                result['status_code'] = response.status

                # Additional wait for JavaScript-heavy sites
                if category in ['spa', 'social']:
                    await page.wait_for_timeout(8000)  # Extra time for JS execution
                else:
                    await page.wait_for_timeout(3000)

                # Count DOM elements as a quality indicator
                try:
                    dom_count = await page.evaluate('document.querySelectorAll("*").length')
                    result['dom_elements'] = dom_count
                except:
                    result['dom_elements'] = 0

                # Record load time
                result['load_time'] = time.time() - start_time

                # Take screenshot
                screenshot_name = f"{category}_{urlparse(url).netloc.replace('.', '_')}_{int(time.time())}.png"
                screenshot_path = self.screenshots_dir / screenshot_name

                await page.screenshot(
                    path=str(screenshot_path),
                    full_page=False,
                    clip={'x': 0, 'y': 0, 'width': 1920, 'height': 1080}
                )

                # Check screenshot file size
                if screenshot_path.exists():
                    result['screenshot_size'] = screenshot_path.stat().st_size

                    # Calculate RR score with category-specific thresholds
                    min_screenshot_size = {
                        'html': 10000,    # 10KB for HTML
                        'spa': 50000,     # 50KB for SPAs (more complex)
                        'social': 30000,  # 30KB for social
                        'api': 5000       # 5KB for APIs (usually text)
                    }

                    min_dom_elements = {
                        'html': 50,
                        'spa': 100,
                        'social': 75,
                        'api': 10
                    }

                    threshold_size = min_screenshot_size.get(category, 10000)
                    threshold_dom = min_dom_elements.get(category, 50)

                    # Score calculation
                    if (200 <= result['status_code'] < 300 and
                        result['screenshot_size'] > threshold_size and
                        result['dom_elements'] > threshold_dom):
                        result['rr_score'] = 1.0
                        logger.info(f"✓ {category.upper()} page successfully replayed: {url}")
                    else:
                        # Partial score based on what worked
                        partial_score = 0.0
                        if 200 <= result['status_code'] < 300:
                            partial_score += 0.3
                        if result['screenshot_size'] > threshold_size:
                            partial_score += 0.4
                        if result['dom_elements'] > threshold_dom:
                            partial_score += 0.3
                        result['rr_score'] = partial_score
                        logger.warning(f"⚠ {category.upper()} page partial replay: {url} (score: {partial_score:.2f})")
            else:
                result['error'] = 'No response received'

        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error measuring {category} site {url}: {e}")

        finally:
            if page:
                try:
                    await page.close()
                except:
                    pass

        return result

    async def analyze_urls(self, urls: List[Dict[str, str]]) -> List[Dict]:
        """
        Analyze multiple URLs for replayability.
        """
        playwright, browser, context = await self.setup_browser()

        try:
            for idx, url_data in enumerate(urls, 1):
                url = url_data.get('url', '')
                category = url_data.get('category', 'unknown')
                url_id = url_data.get('id', f'{category}_{idx}')

                if not url:
                    continue

                print(f"\n[{idx}/{len(urls)}] Processing {category.upper()}: {url}")

                # Get actual wayback URL
                archive_url = self.get_wayback_url(url)
                if not archive_url:
                    result = {
                        'url': url,
                        'archive_url': None,
                        'category': category,
                        'id': url_id,
                        'status_code': None,
                        'screenshot_size': 0,
                        'dom_elements': 0,
                        'load_time': 0,
                        'rr_score': 0.0,
                        'error': 'No archive available',
                        'timestamp': datetime.now().isoformat()
                    }
                    self.results.append(result)
                    continue

                # Measure replayability
                result = await self.measure_replayability(url, archive_url, category, context)
                result['id'] = url_id

                self.results.append(result)
                self.save_results()

                # Delay between requests
                await asyncio.sleep(2)

        finally:
            await context.close()
            await browser.close()
            await playwright.stop()

        return self.results

    def save_results(self):
        """Save results to CSV file."""
        if not self.results:
            return

        fieldnames = [
            'id', 'url', 'archive_url', 'category', 'status_code',
            'screenshot_size', 'dom_elements', 'load_time', 'rr_score',
            'error', 'timestamp'
        ]

        with open(self.results_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)

        logger.info(f"Results saved to {self.results_file}")

    def calculate_statistics(self) -> Dict:
        """Calculate detailed statistics by category (with no-archive counts)."""
        if not self.results:
            return {}

        # Overall statistics
        scores = [r['rr_score'] for r in self.results if r['rr_score'] is not None]

        stats = {
            'overall': {
                'total_urls': len(self.results),
                'successful_replays': sum(1 for s in scores if s >= 0.8),
                'partial_replays': sum(1 for s in scores if 0.2 < s < 0.8),
                'failed_replays': sum(1 for s in scores if s <= 0.2),
                'average_rr': (sum(scores) / len(scores)) if scores else 0.0,
                'success_rate': (sum(1 for s in scores if s >= 0.8) / len(scores) * 100) if scores else 0.0
            },
            'by_category': {}
        }

        # Ensure all categories are present, even if empty
        categories = ['html', 'spa', 'social', 'api']
        for cat in categories:
            cat_results = [r for r in self.results if r.get('category') == cat]
            cat_scores = [r['rr_score'] for r in cat_results if r['rr_score'] is not None]
            count = len(cat_results)
            no_archive_count = sum(
                1 for r in cat_results if (r.get('archive_url') is None or r.get('error') == 'No archive available')
            )

            if count > 0:
                stats['by_category'][cat] = {
                    'count': count,
                    'successful': sum(1 for s in cat_scores if s >= 0.8),
                    'partial': sum(1 for s in cat_scores if 0.2 < s < 0.8),
                    'failed': sum(1 for s in cat_scores if s <= 0.2),
                    'average_rr': (sum(cat_scores) / len(cat_scores)) if cat_scores else 0.0,
                    'success_rate': (sum(1 for s in cat_scores if s >= 0.8) / len(cat_scores) * 100) if cat_scores else 0.0,
                    'avg_load_time': np.mean([r['load_time'] for r in cat_results if r['load_time'] > 0]) if count else 0.0,
                    'avg_dom_elements': np.mean([r['dom_elements'] for r in cat_results if r['dom_elements'] > 0]) if count else 0.0,
                    'no_archive_count': no_archive_count
                }
            else:
                # Prefill zeros so the category still appears
                stats['by_category'][cat] = {
                    'count': 0,
                    'successful': 0,
                    'partial': 0,
                    'failed': 0,
                    'average_rr': 0.0,
                    'success_rate': 0.0,
                    'avg_load_time': 0.0,
                    'avg_dom_elements': 0.0,
                    'no_archive_count': 0
                }

        # Also track overall no-archive count (optional but handy)
        stats['overall']['no_archive_count'] = sum(
            1 for r in self.results if (r.get('archive_url') is None or r.get('error') == 'No archive available')
        )

        return stats

    def create_visualizations(self, stats: Dict):
        """Create comprehensive graphs and visualizations."""

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')

        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('Web Archival Quality Assessment - Replayability Ratio (RR) Analysis',
                     fontsize=16, fontweight='bold')

        categories = ['html', 'spa', 'social', 'api']

        # 1. Success Rate by Category (Bar Chart)
        ax1 = plt.subplot(2, 3, 1)
        success_rates = [stats['by_category'][cat]['success_rate'] for cat in categories]
        colors = ['#2E7D32', '#1976D2', '#F57C00', '#7B1FA2']  # fixed, matches legend order
        bars = ax1.bar(categories, success_rates, color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Success Rate (%)', fontweight='bold')
        ax1.set_title('Success Rate by Website Category', fontweight='bold')
        ax1.set_ylim(0, 100)

        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                     f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

        # 2. Distribution of RR Scores (Stacked Bar)
        ax2 = plt.subplot(2, 3, 2)
        categories_upper = [cat.upper() for cat in categories]
        successful = [stats['by_category'][cat]['successful'] for cat in categories]
        partial = [stats['by_category'][cat]['partial'] for cat in categories]
        failed = [stats['by_category'][cat]['failed'] for cat in categories]

        width = 0.6
        p1 = ax2.bar(categories_upper, successful, width, label='Successful (≥0.8)', color='#4CAF50')
        p2 = ax2.bar(categories_upper, partial, width, bottom=successful, label='Partial (0.2–0.8)', color='#FFC107')
        p3 = ax2.bar(categories_upper, failed, width,
                     bottom=[i+j for i, j in zip(successful, partial)], label='Failed (≤0.2)', color='#F44336')

        ax2.set_ylabel('Number of Sites', fontweight='bold')
        ax2.set_title('Replay Quality Distribution by Category', fontweight='bold')
        ax2.legend(loc='upper right')

        # 3. Average RR Score Comparison
        ax3 = plt.subplot(2, 3, 3)
        avg_scores = [stats['by_category'][cat]['average_rr'] for cat in categories]
        bars = ax3.bar(categories, avg_scores, color=colors,
                       edgecolor='black', linewidth=1.5, alpha=0.7)
        ax3.set_ylabel('Average RR Score', fontweight='bold')
        ax3.set_title('Average Replayability Score by Category', fontweight='bold')
        ax3.set_ylim(0, 1.0)
        ax3.axhline(y=0.5, color='r', linestyle='--', label='Threshold')

        for bar, score in zip(bars, avg_scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

        # 4. Load Time Comparison
        ax4 = plt.subplot(2, 3, 4)
        load_times = [stats['by_category'][cat]['avg_load_time'] for cat in categories]
        bars = ax4.bar(categories, load_times, color='#607D8B', edgecolor='black', linewidth=1.5)
        ax4.set_ylabel('Average Load Time (seconds)', fontweight='bold')
        ax4.set_title('Average Page Load Time by Category', fontweight='bold')

        for bar, t in zip(bars, load_times):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                     f'{t:.1f}s', ha='center', va='bottom')

        # 5. DOM Elements Comparison
        ax5 = plt.subplot(2, 3, 5)
        dom_elements = [stats['by_category'][cat]['avg_dom_elements'] for cat in categories]
        bars = ax5.bar(categories, dom_elements, color='#00BCD4', edgecolor='black', linewidth=1.5)
        ax5.set_ylabel('Average DOM Elements', fontweight='bold')
        ax5.set_title('Average DOM Complexity by Category', fontweight='bold')

        for bar, elements in zip(bars, dom_elements):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 10,
                     f'{int(elements)}', ha='center', va='bottom')

        # 6. Overall Summary Pie Chart
        ax6 = plt.subplot(2, 3, 6)
        overall = stats['overall']
        sizes = [overall['successful_replays'], overall['partial_replays'], overall['failed_replays']]
        labels = ['Successful', 'Partial', 'Failed']
        pie_colors = ['#4CAF50', '#FFC107', '#F44336']
        explode = (0.1, 0, 0)  # Explode successful slice

        if sum(sizes) > 0:
            wedges, texts, autotexts = ax6.pie(
                sizes, labels=labels, colors=pie_colors, autopct='%1.1f%%',
                startangle=90, explode=explode, shadow=True
            )
            ax6.set_title('Overall Replay Success Distribution', fontweight='bold')
            for autotext in autotexts:
                autotext.set_fontweight('bold')
                autotext.set_color('white')
        else:
            ax6.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax6.set_title('Overall Replay Success Distribution', fontweight='bold')

        plt.tight_layout()

        # Save main figure
        graph_path = self.graphs_dir / 'rr_analysis_comprehensive.png'
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comprehensive graph saved to {graph_path}")

        # Detailed comparison table figure (now includes No Archive column)
        fig2, ax = plt.subplots(figsize=(16, 8))
        ax.axis('tight')
        ax.axis('off')

        # Prepare table data
        table_data = [['Category', 'Total', 'Success', 'Partial', 'Failed', 'No Archive', 'Avg Score', 'Success Rate']]
        for cat in categories:
            cat_stats = stats['by_category'][cat]
            table_data.append([
                cat.upper(),
                str(cat_stats['count']),
                str(cat_stats['successful']),
                str(cat_stats['partial']),
                str(cat_stats['failed']),
                str(cat_stats['no_archive_count']),
                f"{cat_stats['average_rr']:.3f}",
                f"{cat_stats['success_rate']:.1f}%"
            ])

        # Add overall row (with overall no-archive count)
        table_data.append([
            'OVERALL',
            str(stats['overall']['total_urls']),
            str(stats['overall']['successful_replays']),
            str(stats['overall']['partial_replays']),
            str(stats['overall']['failed_replays']),
            str(stats['overall'].get('no_archive_count', 0)),
            f"{stats['overall']['average_rr']:.3f}",
            f"{stats['overall']['success_rate']:.1f}%"
        ])

        table = ax.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2)

        # Style the header row
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#3F51B5')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Style the overall row
        last_row = len(table_data) - 1
        for i in range(len(table_data[0])):
            table[(last_row, i)].set_facecolor('#E0E0E0')
            table[(last_row, i)].set_text_props(weight='bold')

        # Color code other rows based on category
        category_colors = {
            'HTML': '#E8F5E9',
            'SPA': '#E3F2FD',
            'SOCIAL': '#FFF3E0',
            'API': '#F3E5F5'
        }

        for i in range(1, len(table_data) - 1):
            cat_name = table_data[i][0]
            color = category_colors.get(cat_name, '#FFFFFF')
            for j in range(len(table_data[0])):
                table[(i, j)].set_facecolor(color)

        plt.title('Detailed Category-wise Replayability Analysis', fontsize=14, fontweight='bold', pad=20)

        table_path = self.graphs_dir / 'rr_analysis_table.png'
        plt.savefig(table_path, dpi=300, bbox_inches='tight')
        logger.info(f"Analysis table saved to {table_path}")

        plt.show()


async def main():
    """Main execution function."""

    # Use default test URLs or load from CSV
    urls_file = Path("../data/urls.csv")
    if urls_file.exists():
        logger.info(f"Loading URLs from {urls_file}")
        with open(urls_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            test_urls = list(reader)
    else:
        logger.info("Using default test URLs for different website categories")
        test_urls = DEFAULT_TEST_URLS

        # Save default URLs to CSV for reference
        with open(urls_file.parent / 'default_urls.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['url', 'category', 'id'])
            writer.writeheader()
            writer.writerows(DEFAULT_TEST_URLS)

    # Initialize analyzer
    analyzer = ReplayabilityRatioAnalyzer()

    # Run analysis
    print("\n" + "="*60)
    print("WEB ARCHIVAL QUALITY ASSESSMENT - REPLAYABILITY RATIO")
    print("="*60)
    print(f"Analyzing {len(test_urls)} URLs across categories:")
    print(f"- HTML: Traditional static websites")
    print(f"- SPA: Single-Page Applications (JavaScript-heavy)")
    print(f"- Social: Social Media Platforms")
    print(f"- API: API-based/Dynamic Content")
    print("="*60)

    results = await analyzer.analyze_urls(test_urls)

    # Calculate and display statistics
    stats = analyzer.calculate_statistics()

    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)

    # Overall results
    overall = stats['overall']
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  Total URLs analyzed: {overall['total_urls']}")
    print(f"  Successful replays: {overall['successful_replays']} ({overall['successful_replays']/overall['total_urls']*100:.1f}%)" if overall['total_urls'] else "  Successful replays: 0 (0.0%)")
    print(f"  Partial replays: {overall['partial_replays']} ({overall['partial_replays']/overall['total_urls']*100:.1f}%)" if overall['total_urls'] else "  Partial replays: 0 (0.0%)")
    print(f"  Failed replays: {overall['failed_replays']} ({overall['failed_replays']/overall['total_urls']*100:.1f}%)" if overall['total_urls'] else "  Failed replays: 0 (0.0%)")
    print(f"  No-archive cases: {overall.get('no_archive_count', 0)}")
    print(f"  Average RR Score: {overall['average_rr']:.3f}")
    print(f"  Overall Success Rate: {overall['success_rate']:.1f}%")

    # Category-wise results (now includes no_archive_count)
    print(f"\n📈 CATEGORY-WISE ANALYSIS:")
    for cat in ['html', 'spa', 'social', 'api']:
        cat_stats = stats['by_category'][cat]
        print(f"\n  {cat.upper()} Websites:")
        print(f"    • Sites tested: {cat_stats['count']}")
        print(f"    • Successful: {cat_stats['successful']} ({cat_stats['success_rate']:.1f}%)")
        print(f"    • Partial: {cat_stats['partial']}")
        print(f"    • Failed: {cat_stats['failed']}")
        print(f"    • No Archive: {cat_stats['no_archive_count']}")
        print(f"    • Average RR Score: {cat_stats['average_rr']:.3f}")
        print(f"    • Avg Load Time: {cat_stats['avg_load_time']:.1f} seconds")
        print(f"    • Avg DOM Elements: {int(cat_stats['avg_dom_elements']) if cat_stats['avg_dom_elements'] else 0}")

    # Key findings based on dissertation objectives
    print(f"\n🔍 KEY FINDINGS (aligned with dissertation objectives):")

    # Find best and worst performing categories (by success rate)
    # Guard against all-zero datasets
    cat_items = list(stats['by_category'].items())
    if any(v['count'] > 0 for _, v in cat_items):
        best_cat = max(cat_items, key=lambda x: x[1]['success_rate'])
        worst_cat = min(cat_items, key=lambda x: x[1]['success_rate'])
        print(f"  1. Best performing category: {best_cat[0].upper()} ({best_cat[1]['success_rate']:.1f}% success)")
        print(f"  2. Worst performing category: {worst_cat[0].upper()} ({worst_cat[1]['success_rate']:.1f}% success)")
    else:
        print("  1–2. Not enough data to determine best/worst categories.")

    # SPA degradation vs HTML (as per dissertation hypothesis)
    if stats['by_category']['html']['count'] > 0 and stats['by_category']['spa']['count'] > 0:
        html_score = stats['by_category']['html']['average_rr']
        spa_score = stats['by_category']['spa']['average_rr']
        degradation = ((html_score - spa_score) / html_score) * 100 if html_score > 0 else 0
        print(f"  3. SPA degradation vs HTML: {degradation:.1f}% lower archival quality")

    # Social media archival challenges
    if stats['by_category']['social']['count'] > 0:
        social_fail_rate = (stats['by_category']['social']['failed'] /
                            stats['by_category']['social']['count'] * 100) if stats['by_category']['social']['count'] else 0
        print(f"  4. Social media failure rate: {social_fail_rate:.1f}% complete failures")

    # Save statistics to JSON
    stats_file = analyzer.output_dir / "rr_statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    # Create visualizations
    print(f"\n📊 Generating comprehensive visualizations...")
    analyzer.create_visualizations(stats)

    print(f"\n💾 OUTPUT FILES SAVED:")
    print(f"  • Detailed results: {analyzer.results_file}")
    print(f"  • Statistics JSON: {stats_file}")
    print(f"  • Screenshots: {analyzer.screenshots_dir}")
    print(f"  • Graphs: {analyzer.graphs_dir}")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    # Install matplotlib if not present
    try:
        import matplotlib
    except ImportError:
        print("Installing matplotlib for visualizations...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])

    try:
        if sys.platform == 'win32':
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(main())
            finally:
                loop.close()
        else:
            asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
