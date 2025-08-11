# run_all_metrics.py
import sys
import os
import subprocess
from pathlib import Path
from datetime import datetime

SEPARATOR = "=" * 60

def print_header(title: str) -> None:
    print("\n" + SEPARATOR)
    print(title)
    print(SEPARATOR)

def run_and_stream(cmd, cwd: Path, extra_env=None) -> int:
    """
    Run a command and stream its output live. Returns the process return code.
    """
    env = os.environ.copy()

    # make unicode logging to console safe on Windows (✓, ⚠, etc.)
    env.setdefault("PYTHONIOENCODING", "utf-8")

    # headless plots for all subprocesses
    env.setdefault("MPLBACKEND", "Agg")

    if extra_env:
        env.update(extra_env)

    # Start the process
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",   # never crash on weird bytes
    )

    # Stream output
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")

    proc.wait()
    return proc.returncode

def main():
    root = Path(__file__).resolve().parent
    metrics_dir = root / "metrics"
    data_csv = root / "data" / "urls.csv"

    print_header("RUNNING ALL METRICS")

    # Quick sanity check for data file (scripts will still try defaults if missing)
    if not data_csv.exists():
        print(f"WARNING: {data_csv} not found. Some metrics may fall back to defaults or fail.")

    # Ordered list of metric scripts to run
    metrics = [
        ("RR_METRIC.PY", "rr_metric.py"),
        ("AS_METRIC.PY", "as_metric.py"),
        ("BLD_METRIC.PY", "bld_metric.py"),
        ("MDS_METRIC.PY", "mds_metric.py"),
        ("VCS_METRIC.PY", "vcs_metric.py"),
    ]

    # Use the current Python interpreter
    py = sys.executable

    results = []

    for title, script in metrics:
        print_header(f"RUNNING {title}")

        script_path = metrics_dir / script
        if not script_path.exists():
            print(f"ERROR: {script_path} not found. Skipping.")
            results.append((script, "missing"))
            continue

        # Run the script with CWD=metrics so ../data/urls.csv resolves correctly
        rc = run_and_stream([py, script_path.name], cwd=metrics_dir)
        status = "ok" if rc == 0 else f"failed (rc={rc})"
        results.append((script, status))

    print_header("SUMMARY")
    for script, status in results:
        print(f"{script:15s} : {status}")

    print_header("ALL METRICS COMPLETED")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
