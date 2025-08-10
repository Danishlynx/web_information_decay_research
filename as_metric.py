"""
Archival Span (AS) Metric
Measures temporal coverage of web archives
Based on Vlassenroot et al., 2021 research
"""

import asyncio
import csv
import json
import logging
import sys
import requests
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
from collections import defaultdict

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('as_metric.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ArchivalSpanAnalyzer:
    def __init__(self):
        self.output_dir = Path("outputs")
        self.results_file = self.output_dir / "as_results.csv"
        self.graphs_dir = self.output_dir / "graphs"
        
        self.output_dir.mkdir(exist_ok=True)
        self.graphs_dir.mkdir(exist_ok=True)
        
        self.results = []
    
    def get_archive_history(self, url: str) -> Dict:
        """Get archival history from Wayback Machine CDX API"""
        result = {
            'url': url,
            'first_capture': None,
            'last_capture': None,
            'total_captures': 0,
            'span_days': 0,
            'capture_frequency': 0,
            'yearly_captures': defaultdict(int),
            'error': None
        }
        
        try:
            # Clean URL
            clean_url = url.strip()
            if not clean_url.startswith(('http://', 'https://')):
                clean_url = 'http://' + clean_url
            
            # Use CDX API to get all timestamps
            cdx_url = f"https://web.archive.org/cdx/search/cdx"
            params = {
                'url': clean_url,
                'output': 'json',
                'fl': 'timestamp,statuscode',
                'filter': 'statuscode:200',
                'collapse': 'timestamp:8'  # Collapse to daily
            }
            
            print(f"  Fetching archive history for {url}")
            response = requests.get(cdx_url, params=params, timeout=30)
            
            if response.status_code == 200 and response.text:
                lines = response.text.strip().split('\n')
                if len(lines) > 1:  # First line is header
                    captures = []
                    for line in lines[1:]:
                        try:
                            data = json.loads(line)
                            timestamp = data[0]
                            # Parse timestamp (YYYYMMDDhhmmss)
                            dt = datetime.strptime(timestamp[:14], '%Y%m%d%H%M%S')
                            captures.append(dt)
                            # Track yearly distribution
                            result['yearly_captures'][dt.year] += 1
                        except:
                            continue
                    
                    if captures:
                        captures.sort()
                        result['first_capture'] = captures[0]
                        result['last_capture'] = captures[-1]
                        result['total_captures'] = len(captures)
                        
                        # Calculate span in days
                        span = captures[-1] - captures[0]
                        result['span_days'] = span.days
                        
                        # Calculate capture frequency (captures per year)
                        if result['span_days'] > 0:
                            result['capture_frequency'] = result['total_captures'] / (result['span_days'] / 365.25)
                        
                        print(f"    Found {result['total_captures']} captures spanning {result['span_days']} days")
                    else:
                        result['error'] = "No valid captures found"
                else:
                    result['error'] = "No archive data available"
            else:
                result['error'] = "CDX API request failed"
                
        except requests.exceptions.Timeout:
            result['error'] = "Request timeout"
            logger.error(f"Timeout for {url}")
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error fetching history for {url}: {e}")
        
        return result
    
    def calculate_as_score(self, result: Dict) -> float:
        """Calculate normalized Archival Span score (0-1)"""
        if result['error'] or result['total_captures'] == 0:
            return 0.0
        
        # Factors for AS score:
        # 1. Span coverage (max 10 years = 1.0)
        span_score = min(result['span_days'] / 3650, 1.0)
        
        # 2. Capture frequency (ideal: 12+ per year = 1.0)
        freq_score = min(result['capture_frequency'] / 12, 1.0)
        
        # 3. Total captures (100+ = 1.0)
        total_score = min(result['total_captures'] / 100, 1.0)
        
        # 4. Recency (captured in last year = 1.0)
        if result['last_capture']:
            days_since_last = (datetime.now() - result['last_capture']).days
            recency_score = max(0, 1 - (days_since_last / 365))
        else:
            recency_score = 0
        
        # Weighted average
        as_score = (span_score * 0.3 + freq_score * 0.2 + 
                   total_score * 0.2 + recency_score * 0.3)
        
        return as_score
    
    def analyze_urls(self, urls: List[Dict]) -> List[Dict]:
        """Analyze multiple URLs"""
        for idx, url_data in enumerate(urls, 1):
            if isinstance(url_data, dict):
                url = url_data.get('url', '')
                category = url_data.get('category', 'unknown')
            else:
                url = url_data
                category = 'unknown'
            
            print(f"\n[{idx}/{len(urls)}] Processing {url}")
            
            result = self.get_archive_history(url)
            result['category'] = category
            result['as_score'] = self.calculate_as_score(result)
            result['timestamp'] = datetime.now().isoformat()
            
            # Log score
            if result['as_score'] > 0.7:
                print(f"  [EXCELLENT] High archival span: {result['as_score']:.3f}")
            elif result['as_score'] > 0.4:
                print(f"  [GOOD] Moderate archival span: {result['as_score']:.3f}")
            else:
                print(f"  [POOR] Low archival span: {result['as_score']:.3f}")
            
            self.results.append(result)
            
            # Small delay to avoid rate limiting
            import time
            time.sleep(2)
        
        self.save_results()
        return self.results
    
    def save_results(self):
        """Save results to CSV"""
        if not self.results:
            return
        
        # Simplified CSV without yearly_captures dict
        with open(self.results_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['url', 'category', 'as_score', 'total_captures', 
                         'span_days', 'capture_frequency', 'first_capture', 
                         'last_capture', 'error', 'timestamp']
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            
            for result in self.results:
                row = {k: v for k, v in result.items() if k != 'yearly_captures'}
                # Convert datetime to string
                if row.get('first_capture'):
                    row['first_capture'] = row['first_capture'].isoformat()
                if row.get('last_capture'):
                    row['last_capture'] = row['last_capture'].isoformat()
                writer.writerow(row)
        
        print(f"\nResults saved to {self.results_file}")
        
        # Save detailed data with yearly captures
        detailed_file = self.output_dir / "as_detailed.json"
        with open(detailed_file, 'w') as f:
            # Convert datetime objects to strings for JSON
            json_results = []
            for r in self.results:
                r_copy = r.copy()
                if r_copy.get('first_capture'):
                    r_copy['first_capture'] = r_copy['first_capture'].isoformat()
                if r_copy.get('last_capture'):
                    r_copy['last_capture'] = r_copy['last_capture'].isoformat()
                r_copy['yearly_captures'] = dict(r_copy['yearly_captures'])
                json_results.append(r_copy)
            json.dump(json_results, f, indent=2, default=str)
        print(f"Detailed data saved to {detailed_file}")
    
    def calculate_and_visualize(self):
        """Create visualizations"""
        if not self.results:
            return
        
        # Filter valid results
        valid_results = [r for r in self.results if not r.get('error')]
        
        if not valid_results:
            print("No valid results to visualize")
            return
        
        # Calculate stats by category
        stats = {}
        for cat in ['html', 'spa', 'social']:
            cat_results = [r for r in valid_results if r['category'] == cat]
            if cat_results:
                stats[cat] = {
                    'avg_as_score': np.mean([r['as_score'] for r in cat_results]),
                    'avg_span_days': np.mean([r['span_days'] for r in cat_results]),
                    'avg_captures': np.mean([r['total_captures'] for r in cat_results]),
                    'avg_frequency': np.mean([r['capture_frequency'] for r in cat_results]),
                    'count': len(cat_results)
                }
        
        # Print results
        print("\n" + "="*60)
        print("ARCHIVAL SPAN RESULTS")
        print("="*60)
        for cat, s in stats.items():
            print(f"\n{cat.upper()}:")
            print(f"  Average AS Score: {s['avg_as_score']:.3f}")
            print(f"  Average Span: {s['avg_span_days']:.0f} days ({s['avg_span_days']/365:.1f} years)")
            print(f"  Average Captures: {s['avg_captures']:.0f}")
            print(f"  Average Frequency: {s['avg_frequency']:.1f} captures/year")
            print(f"  Sites analyzed: {s['count']}")
        
        # Create visualizations
        if stats:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 1. AS Score by category
            ax1 = axes[0, 0]
            categories = list(stats.keys())
            as_scores = [stats[c]['avg_as_score'] for c in categories]
            
            colors = ['green' if s > 0.6 else 'orange' if s > 0.3 else 'red' for s in as_scores]
            bars = ax1.bar(categories, as_scores, color=colors)
            ax1.set_ylabel('AS Score')
            ax1.set_title('Average Archival Span Score by Category')
            ax1.set_ylim(0, 1)
            
            for bar, score in zip(bars, as_scores):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', fontweight='bold')
            
            # 2. Temporal span comparison
            ax2 = axes[0, 1]
            span_years = [stats[c]['avg_span_days']/365 for c in categories]
            bars = ax2.bar(categories, span_years, color='#2196F3')
            ax2.set_ylabel('Years')
            ax2.set_title('Average Archive Span by Category')
            
            for bar, years in zip(bars, span_years):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{years:.1f}y', ha='center', fontweight='bold')
            
            # 3. Capture frequency
            ax3 = axes[1, 0]
            frequencies = [stats[c]['avg_frequency'] for c in categories]
            bars = ax3.bar(categories, frequencies, color='#FF9800')
            ax3.set_ylabel('Captures per Year')
            ax3.set_title('Average Capture Frequency by Category')
            
            for bar, freq in zip(bars, frequencies):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                        f'{freq:.1f}', ha='center', fontweight='bold')
            
            # 4. Timeline visualization for all sites
            ax4 = axes[1, 1]
            for i, result in enumerate(valid_results[:10]):  # Show first 10
                if result['first_capture'] and result['last_capture']:
                    # Convert strings back to datetime if needed
                    if isinstance(result['first_capture'], str):
                        first = datetime.fromisoformat(result['first_capture'])
                        last = datetime.fromisoformat(result['last_capture'])
                    else:
                        first = result['first_capture']
                        last = result['last_capture']
                    
                    ax4.barh(i, (last - first).days / 365, 
                            left=(first - datetime(2000, 1, 1)).days / 365,
                            height=0.6, alpha=0.7)
            
            ax4.set_xlabel('Years since 2000')
            ax4.set_ylabel('Website')
            ax4.set_title('Archive Timeline Coverage')
            ax4.set_yticks(range(min(10, len(valid_results))))
            ax4.set_yticklabels([r['url'].split('//')[1][:20] 
                                for r in valid_results[:10]])
            
            plt.suptitle('Archival Span Analysis', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            graph_path = self.graphs_dir / 'as_analysis.png'
            plt.savefig(graph_path, dpi=150)
            print(f"\nGraph saved to {graph_path}")
            plt.show()
        
        # Save statistics
        stats_file = self.output_dir / "as_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to {stats_file}")


def main():
    print("\n" + "="*60)
    print("ARCHIVAL SPAN (AS) ANALYSIS")
    print("="*60)
    print("Measuring temporal coverage of web archives")
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
    
    analyzer = ArchivalSpanAnalyzer()
    analyzer.analyze_urls(test_urls)
    analyzer.calculate_and_visualize()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()