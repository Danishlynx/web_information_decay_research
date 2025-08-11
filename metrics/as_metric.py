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
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import requests

# Use the Proactor loop on Windows for better subprocess/IO behavior
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("as_metric.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class ArchivalSpanAnalyzer:
    def __init__(self):
        self.output_dir = Path("outputs/as")
        self.results_file = self.output_dir / "as_results.csv"
        self.graphs_dir = self.output_dir / "graphs"

        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.graphs_dir.mkdir(exist_ok=True, parents=True)

        self.results: List[Dict] = []

    # ---------------------------
    # Helpers
    # ---------------------------
    def _http_get_with_retries(self, url: str, params: Optional[dict] = None, timeout: int = 30):
        """Simple retry wrapper for requests.get with backoff."""
        attempts = 3
        backoff = 2.0
        last_exc = None
        for i in range(attempts):
            try:
                return requests.get(url, params=params, timeout=timeout)
            except Exception as e:
                last_exc = e
                if i < attempts - 1:
                    time.sleep(backoff)
                    backoff *= 1.5
        # Re-raise last exception to be handled by caller
        raise last_exc

    # ---------------------------
    # Core CDX fetch
    # ---------------------------
    def get_archive_history(self, url: str) -> Dict:
        """
        Get archival history from the Internet Archive CDX API.

        CDX JSON returns a single JSON array where the first row is the header.
        Example: [["timestamp","statuscode"],["20200101123456","200"], ...]
        """
        result = {
            "url": url,
            "first_capture": None,
            "last_capture": None,
            "total_captures": 0,
            "span_days": 0,
            "capture_frequency": 0.0,
            "yearly_captures": defaultdict(int),
            "error": None,
        }

        try:
            # Normalize URL
            clean_url = url.strip()
            if not clean_url.startswith(("http://", "https://")):
                clean_url = "http://" + clean_url

            cdx_url = "https://web.archive.org/cdx/search/cdx"
            params = {
                "url": clean_url,
                "output": "json",
                "fl": "timestamp,statuscode",
                "filter": "statuscode:200",
                "collapse": "timestamp:8",  # daily collapse
            }

            logger.info(f"Fetching archive history for {url}")
            resp = self._http_get_with_retries(cdx_url, params=params, timeout=30)

            if resp.status_code != 200:
                result["error"] = f"CDX API request failed (HTTP {resp.status_code})"
                return result

            # CDX returns JSON array (first row = header)
            try:
                rows = resp.json()
            except json.JSONDecodeError:
                result["error"] = "Invalid JSON returned by CDX API"
                return result

            if not rows or len(rows) <= 1:
                result["error"] = "No archive data available"
                return result

            # rows[0] is header; parse remaining rows
            captures = []
            for row in rows[1:]:
                # Expect ["YYYYMMDDhhmmss","200"]
                if not row or len(row) < 1:
                    continue
                ts = str(row[0])
                ts14 = ts[:14]
                try:
                    dt = datetime.strptime(ts14, "%Y%m%d%H%M%S")
                    captures.append(dt)
                    result["yearly_captures"][dt.year] += 1
                except Exception:
                    continue

            if not captures:
                result["error"] = "No valid captures found"
                return result

            captures.sort()
            first = captures[0]
            last = captures[-1]
            result["first_capture"] = first
            result["last_capture"] = last
            result["total_captures"] = len(captures)

            # Span & frequency
            span_days = (last - first).days
            result["span_days"] = max(span_days, 0)

            if result["span_days"] > 0:
                # captures per year (using 365.25)
                result["capture_frequency"] = result["total_captures"] / (
                    result["span_days"] / 365.25
                )

            logger.info(
                f"Found {result['total_captures']} captures spanning {result['span_days']} days"
            )

        except requests.exceptions.Timeout:
            result["error"] = "Request timeout"
            logger.error(f"Timeout for {url}")
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Error fetching history for {url}: {e}")

        return result

    # ---------------------------
    # Scoring
    # ---------------------------
    def calculate_as_score(self, result: Dict) -> Optional[float]:
        """Calculate normalized Archival Span score (0–1).
        Returns None when fetch failed or no data, so we can mark it as N/A.
        """
        if result.get("error"):
            return None  # N/A due to fetch failure
        if result.get("total_captures", 0) == 0:
            # Treat confirmed zero captures as a true 0.0 score
            return 0.0

        # 1) Span coverage (maxed at 10 years)
        span_score = min(result["span_days"] / 3650.0, 1.0)

        # 2) Capture frequency (12+ per year = 1.0)
        freq_score = min(result["capture_frequency"] / 12.0, 1.0)

        # 3) Total captures (100+ = 1.0)
        total_score = min(result["total_captures"] / 100.0, 1.0)

        # 4) Recency (captured in last year = 1.0); use UTC for consistency
        if result.get("last_capture"):
            days_since_last = (datetime.utcnow() - result["last_capture"]).days
            recency_score = max(0.0, 1.0 - (days_since_last / 365.0))
        else:
            recency_score = 0.0

        # Weighted average
        as_score = (
            span_score * 0.3 + freq_score * 0.2 + total_score * 0.2 + recency_score * 0.3
        )
        return float(as_score)

    # ---------------------------
    # Pipeline
    # ---------------------------
    def analyze_urls(self, urls: List[Dict]) -> List[Dict]:
        """Analyze multiple URLs from a list of dicts (expects keys 'url' and 'category')."""
        for idx, url_data in enumerate(urls, 1):
            # Accept both dict rows (from DictReader) or plain strings
            if isinstance(url_data, dict):
                url = url_data.get("url", "").strip()
                category = url_data.get("category", "unknown").strip().lower()
            else:
                url = str(url_data)
                category = "unknown"

            if not url:
                continue

            print(f"\n[{idx}/{len(urls)}] Processing {url}")

            result = self.get_archive_history(url)
            result["category"] = category
            result["as_score"] = self.calculate_as_score(result)
            result["timestamp"] = datetime.utcnow().isoformat()

            # Log scoring bucket
            score = result["as_score"]
            if score is None:
                print("  [NO DATA] Could not retrieve archival span.")
            elif score > 0.7:
                print(f"  [EXCELLENT] High archival span: {score:.3f}")
            elif score > 0.4:
                print(f"  [GOOD] Moderate archival span: {score:.3f}")
            else:
                print(f"  [POOR] Low archival span: {score:.3f}")

            self.results.append(result)

            # Gentle pacing to avoid rate limiting
            time.sleep(1.5)

        self.save_results()
        return self.results

    # ---------------------------
    # Output
    # ---------------------------
    def save_results(self):
        """Save results to CSV and JSON."""
        if not self.results:
            return

        # CSV (exclude yearly_captures)
        with open(self.results_file, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "url",
                "category",
                "as_score",
                "total_captures",
                "span_days",
                "capture_frequency",
                "first_capture",
                "last_capture",
                "error",
                "timestamp",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()

            for result in self.results:
                row = {k: v for k, v in result.items() if k != "yearly_captures"}
                # Serialize datetimes
                if row.get("first_capture"):
                    row["first_capture"] = row["first_capture"].isoformat()
                if row.get("last_capture"):
                    row["last_capture"] = row["last_capture"].isoformat()
                # Mark N/A scores
                if row.get("as_score") is None:
                    row["as_score"] = "N/A"
                writer.writerow(row)

        print(f"\nResults saved to {self.results_file}")

        # Detailed JSON (keeps yearly breakdown)
        detailed_file = self.output_dir / "as_detailed.json"
        json_results = []
        for r in self.results:
            rc = dict(r)
            if rc.get("first_capture"):
                rc["first_capture"] = rc["first_capture"].isoformat()
            if rc.get("last_capture"):
                rc["last_capture"] = rc["last_capture"].isoformat()
            rc["yearly_captures"] = dict(rc.get("yearly_captures", {}))
            json_results.append(rc)

        with open(detailed_file, "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2, default=str)

        print(f"Detailed data saved to {detailed_file}")

    # ---------------------------
    # Viz & stats
    # ---------------------------
    def calculate_and_visualize(self):
        """Create summary statistics and visualizations."""
        if not self.results:
            print("No results to visualize")
            return

        # Separate valid/errored for clearer reporting
        valid_results = [r for r in self.results if r.get("as_score") is not None]
        errored_results = [r for r in self.results if r.get("as_score") is None]

        if not valid_results:
            print("No valid results to visualize")
            return

        # Include ALL categories present (html, spa, social, api, unknown, etc.)
        categories_present = sorted({r.get("category", "unknown") for r in self.results})

        # Calculate stats by category
        stats: Dict[str, Dict] = {}
        error_rates: Dict[str, Dict] = {}
        for cat in categories_present:
            cat_all = [r for r in self.results if r.get("category") == cat]
            cat_valid = [r for r in cat_all if r.get("as_score") is not None]
            if cat_valid:
                stats[cat] = {
                    "avg_as_score": float(np.mean([r["as_score"] for r in cat_valid])),
                    "avg_span_days": float(np.mean([r["span_days"] for r in cat_valid])),
                    "avg_captures": float(np.mean([r["total_captures"] for r in cat_valid])),
                    "avg_frequency": float(
                        np.mean([r["capture_frequency"] for r in cat_valid])
                    ),
                    "count": len(cat_valid),
                }
            # Error rate per category (includes timeouts/failures)
            error_rates[cat] = {
                "total": len(cat_all),
                "errors": sum(1 for r in cat_all if r.get("as_score") is None),
            }

        # Print results
        print("\n" + "=" * 60)
        print("ARCHIVAL SPAN RESULTS")
        print("=" * 60)
        for cat in categories_present:
            if cat in stats:
                s = stats[cat]
                yrs = s["avg_span_days"] / 365.0 if s["avg_span_days"] else 0.0
                print(f"\n{cat.upper()}:")
                print(f"  Average AS Score: {s['avg_as_score']:.3f}")
                print(f"  Average Span: {s['avg_span_days']:.0f} days ({yrs:.1f} years)")
                print(f"  Average Captures: {s['avg_captures']:.0f}")
                print(f"  Average Frequency: {s['avg_frequency']:.1f} captures/year")
                print(f"  Sites analyzed (valid): {s['count']}")
            else:
                print(f"\n{cat.upper()}: No valid results")

            er = error_rates.get(cat, {"total": 0, "errors": 0})
            rate = (er["errors"] / er["total"] * 100) if er["total"] else 0.0
            print(f"  Error rate: {rate:.1f}% ({er['errors']}/{er['total']})")

        # Visualizations
        if stats:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # 1) AS Score by category
            ax1 = axes[0, 0]
            categories = list(stats.keys())
            as_scores = [stats[c]["avg_as_score"] for c in categories]
            colors = [
                "green" if s > 0.6 else ("orange" if s > 0.3 else "red") for s in as_scores
            ]
            bars = ax1.bar(categories, as_scores, color=colors)
            ax1.set_ylabel("AS Score")
            ax1.set_title("Average Archival Span Score by Category")
            ax1.set_ylim(0, 1)
            for bar, score in zip(bars, as_scores):
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{score:.3f}",
                    ha="center",
                    fontweight="bold",
                )

            # 2) Temporal span comparison (avg years)
            ax2 = axes[0, 1]
            span_years = [stats[c]["avg_span_days"] / 365.0 for c in categories]
            bars = ax2.bar(categories, span_years)
            ax2.set_ylabel("Years")
            ax2.set_title("Average Archive Span by Category")
            for bar, years in zip(bars, span_years):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    f"{years:.1f}y",
                    ha="center",
                    fontweight="bold",
                )

            # 3) Capture frequency
            ax3 = axes[1, 0]
            frequencies = [stats[c]["avg_frequency"] for c in categories]
            bars = ax3.bar(categories, frequencies)
            ax3.set_ylabel("Captures per Year")
            ax3.set_title("Average Capture Frequency by Category")
            for bar, freq in zip(bars, frequencies):
                ax3.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    f"{freq:.1f}",
                    ha="center",
                    fontweight="bold",
                )

            # 4) Timeline visualization for up to 10 valid sites
            ax4 = axes[1, 1]
            valid_sorted = sorted(
                [
                    r
                    for r in valid_results
                    if r.get("first_capture") and r.get("last_capture")
                ],
                key=lambda r: r["first_capture"],
            )
            subset = valid_sorted[:10]
            base = datetime(2000, 1, 1)
            for i, r in enumerate(subset):
                first = r["first_capture"]
                last = r["last_capture"]
                ax4.barh(
                    i,
                    (last - first).days / 365.0,
                    left=max(0.0, (first - base).days / 365.0),
                    height=0.6,
                    alpha=0.7,
                )
            ax4.set_xlabel("Years since 2000")
            ax4.set_ylabel("Website")
            ax4.set_title("Archive Timeline Coverage")
            ax4.set_yticks(range(len(subset)))
            ax4.set_yticklabels([r["url"].split("//", 1)[-1][:24] for r in subset])

            plt.suptitle("Archival Span Analysis", fontsize=14, fontweight="bold")
            plt.tight_layout()

            graph_path = self.graphs_dir / "as_analysis.png"
            plt.savefig(graph_path, dpi=150)
            print(f"\nGraph saved to {graph_path}")
            plt.show()

        # Save statistics JSON
        stats_file = self.output_dir / "as_statistics.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump({"stats": stats, "error_rates": error_rates}, f, indent=2)
        print(f"Statistics saved to {stats_file}")


def main():
    print("\n" + "=" * 60)
    print("ARCHIVAL SPAN (AS) ANALYSIS")
    print("=" * 60)
    print("Measuring temporal coverage of web archives")
    print("=" * 60)

    # Load URLs from CSV
    # Looks for ../data/urls.csv; if not found, try ./urls.csv as a fallback.
    urls_file = Path("../data/urls.csv")
    if not urls_file.exists():
        alt = Path("urls.csv")
        if alt.exists():
            urls_file = alt

    if urls_file.exists():
        print(f"Loading URLs from {urls_file}")
        with open(urls_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            test_urls = list(reader)
    else:
        print("ERROR: ../data/urls.csv (or ./urls.csv) not found!")
        return

    analyzer = ArchivalSpanAnalyzer()
    analyzer.analyze_urls(test_urls)
    analyzer.calculate_and_visualize()

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
