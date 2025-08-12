"""
Archival Span (AS)
Measures temporal coverage from CDX.

Bias fixes / options:
- Post-filter statuses instead of hard CDX filter (so we can include 3xx if desired).
- Daily collapse remains default, but toggleable.
- Clear separation between 'N/A' (fetch failure) and true 0.0 (confirmed no captures).
"""

import csv
import json
import logging
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import requests

# Windows asyncio not required here (no async)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("as_metric.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("AS")

OUT_DIR = Path("outputs/as")
GRAPH_DIR = OUT_DIR / "graphs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# Config
INCLUDE_3XX = True       # include 3xx captures in span/frequency
DAILY_COLLAPSE = True    # collapse to daily resolution


def _http_get(url: str, params: Optional[dict] = None, timeout: int = 30):
    attempts = 3
    backoff = 2.0
    for i in range(attempts):
        try:
            return requests.get(url, params=params, timeout=timeout)
        except Exception as e:
            if i == attempts - 1:
                raise
            time.sleep(backoff)
            backoff *= 1.6


def _fetch_cdx(url: str) -> List[List[str]]:
    """Return CDX JSON rows (header + data)."""
    clean = url if url.startswith(("http://", "https://")) else "http://" + url
    params = {
        "url": clean,
        "output": "json",
        "fl": "timestamp,statuscode",
    }
    if DAILY_COLLAPSE:
        params["collapse"] = "timestamp:8"
    r = _http_get("https://web.archive.org/cdx/search/cdx", params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"CDX HTTP {r.status_code}")
    return r.json()


def _parse_rows(rows: List[List[str]]) -> Dict:
    """Parse rows into captures (filtering statuses) and compute basics."""
    result = {
        "first_capture": None,
        "last_capture": None,
        "total_captures": 0,
        "span_days": 0,
        "capture_frequency": 0.0,
        "yearly_captures": defaultdict(int),
        "error": None,
    }
    if not rows or len(rows) <= 1:
        # confirmed none
        return result

    status_ok = re.compile(r"^2\d\d$")
    status_redir = re.compile(r"^3\d\d$")

    captures = []
    for row in rows[1:]:
        if not row or len(row) < 2:
            continue
        ts, sc = str(row[0]), str(row[1])
        if status_ok.match(sc) or (INCLUDE_3XX and status_redir.match(sc)):
            ts14 = ts[:14]
            try:
                dt = datetime.strptime(ts14, "%Y%m%d%H%M%S")
                captures.append(dt)
                result["yearly_captures"][dt.year] += 1
            except Exception:
                continue

    if not captures:
        return result

    captures.sort()
    first, last = captures[0], captures[-1]
    result["first_capture"] = first
    result["last_capture"] = last
    result["total_captures"] = len(captures)
    result["span_days"] = max(0, (last - first).days)
    if result["span_days"] > 0:
        result["capture_frequency"] = result["total_captures"] / (result["span_days"] / 365.25)
    return result


def _as_score(rec: Dict) -> Optional[float]:
    if rec.get("error"):
        return None
    if rec.get("total_captures", 0) == 0:
        return 0.0  # confirmed zero
    span_score = min(rec["span_days"] / 3650.0, 1.0)  # up to 10 years
    freq_score = min(rec["capture_frequency"] / 12.0, 1.0)  # 12+/year
    total_score = min(rec["total_captures"] / 100.0, 1.0)  # 100+ captures
    if rec.get("last_capture"):
        days_since_last = (datetime.utcnow() - rec["last_capture"]).days
        recency_score = max(0.0, 1.0 - (days_since_last / 365.0))
    else:
        recency_score = 0.0
    return float(0.3 * span_score + 0.2 * freq_score + 0.2 * total_score + 0.3 * recency_score)


class ArchivalSpanAnalyzer:
    def __init__(self):
        self.results: List[Dict] = []

    def analyze_urls(self, rows: List[Dict]) -> List[Dict]:
        for i, r in enumerate(rows, 1):
            url = r.get("url", "").strip()
            category = r.get("category", "unknown").strip().lower()
            if not url:
                continue
            print(f"\n[{i}/{len(rows)}] {url} ({category})")
            record = {
                "url": url, "category": category, "timestamp": datetime.utcnow().isoformat(),
                "first_capture": None, "last_capture": None, "yearly_captures": {},
                "total_captures": 0, "span_days": 0, "capture_frequency": 0.0, "error": None, "as_score": None
            }
            try:
                rows_json = _fetch_cdx(url)
                parsed = _parse_rows(rows_json)
                record.update(parsed)
                record["yearly_captures"] = dict(parsed.get("yearly_captures", {}))
                record["as_score"] = _as_score(record)
                if record["as_score"] is None:
                    print("  [N/A] fetch failed")
                elif record["as_score"] > 0.7:
                    print(f"  [EXCELLENT] {record['as_score']:.3f}")
                elif record["as_score"] > 0.4:
                    print(f"  [GOOD] {record['as_score']:.3f}")
                else:
                    print(f"  [POOR] {record['as_score']:.3f}")
            except Exception as e:
                record["error"] = str(e)
                record["as_score"] = None
                print(f"  [ERROR] {e}")
            self.results.append(record)
            time.sleep(1.2)  # polite pacing
        self._save()
        return self.results

    def _save(self):
        out_csv = OUT_DIR / "as_results.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["url", "category", "as_score", "total_captures", "span_days",
                            "capture_frequency", "first_capture", "last_capture", "error", "timestamp"],
            )
            writer.writeheader()
            for r in self.results:
                row = dict(r)
                if isinstance(row.get("first_capture"), datetime):
                    row["first_capture"] = row["first_capture"].isoformat()
                if isinstance(row.get("last_capture"), datetime):
                    row["last_capture"] = row["last_capture"].isoformat()
                writer.writerow(row)
        (OUT_DIR / "as_detailed.json").write_text(json.dumps(self.results, default=str, indent=2), encoding="utf-8")
        print(f"\nResults saved to {out_csv}")

    def summarize_and_plot(self):
        if not self.results:
            print("No results.")
            return
        cats = sorted({r["category"] for r in self.results})
        stats = {}
        errs = {}
        for c in cats:
            all_rows = [r for r in self.results if r["category"] == c]
            valid = [r for r in all_rows if r["as_score"] is not None]
            stats[c] = {
                "avg_as": float(np.mean([r["as_score"] for r in valid])) if valid else 0.0,
                "avg_span_days": float(np.mean([r["span_days"] for r in valid])) if valid else 0.0,
                "avg_captures": float(np.mean([r["total_captures"] for r in valid])) if valid else 0.0,
                "avg_freq": float(np.mean([r["capture_frequency"] for r in valid])) if valid else 0.0,
                "count": len(valid),
            }
            errs[c] = {"total": len(all_rows), "errors": len(all_rows) - len(valid)}

        print("\n" + "="*60)
        print("ARCHIVAL SPAN (AS)")
        print("="*60)
        for c in cats:
            s = stats[c]
            e = errs[c]
            yrs = s["avg_span_days"] / 365.0 if s["avg_span_days"] else 0.0
            print(f"\n{c.upper()}: valid={s['count']} error-rate={(e['errors']/e['total']*100 if e['total'] else 0):.1f}%")
            print(f"  Avg AS {s['avg_as']:.3f} | Span {s['avg_span_days']:.0f}d ({yrs:.1f}y) | Caps {s['avg_captures']:.0f} | Freq {s['avg_freq']:.1f}/y")

        valid_cats = [c for c in cats if stats[c]["count"]]
        if not valid_cats:
            return
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))

        # AS score
        vals = [stats[c]["avg_as"] for c in valid_cats]
        colors = ["green" if v > 0.6 else "orange" if v > 0.3 else "red" for v in vals]
        bars = axes[0, 0].bar(valid_cats, vals, color=colors)
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].set_title("Average AS Score")
        for b, v in zip(bars, vals):
            axes[0, 0].text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}", ha="center", fontweight="bold")

        # Span years
        years = [stats[c]["avg_span_days"] / 365.0 for c in valid_cats]
        bars = axes[0, 1].bar(valid_cats, years)
        axes[0, 1].set_title("Average Span (years)")
        for b, v in zip(bars, years):
            axes[0, 1].text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.1f}", ha="center", fontweight="bold")

        # Frequency
        freqs = [stats[c]["avg_freq"] for c in valid_cats]
        bars = axes[1, 0].bar(valid_cats, freqs)
        axes[1, 0].set_title("Average Capture Frequency (/year)")
        for b, v in zip(bars, freqs):
            axes[1, 0].text(b.get_x()+b.get_width()/2, v+0.05, f"{v:.1f}", ha="center", fontweight="bold")

        # Error rates
        er = [errs[c]["errors"] / errs[c]["total"] * 100 if errs[c]["total"] else 0 for c in valid_cats]
        bars = axes[1, 1].bar(valid_cats, er)
        axes[1, 1].set_title("Error Rate (%)")
        for b, v in zip(bars, er):
            axes[1, 1].text(b.get_x()+b.get_width()/2, v+0.5, f"{v:.1f}%", ha="center", fontweight="bold")

        fig.suptitle("Archival Span Analysis", fontsize=14, fontweight="bold")
        fig.tight_layout()
        out_png = GRAPH_DIR / "as_analysis.png"
        fig.savefig(out_png, dpi=160)
        print(f"\nGraph saved to {out_png}")


def main():
    print("\n" + "="*60)
    print("ARCHIVAL SPAN (AS)")
    print("="*60)
    urls_csv = Path("../data/urls.csv")
    if not urls_csv.exists():
        print("ERROR: ../data/urls.csv not found.")
        return
    with urls_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    analyzer = ArchivalSpanAnalyzer()
    analyzer.analyze_urls(rows)
    analyzer.summarize_and_plot()
    print("\nDONE")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted")
