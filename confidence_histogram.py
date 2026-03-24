"""
Plot histogram of confidence scores: Blue=Actual Wake, Red=Actual Non-wake.
Helps visualize overlap and justify threshold choice (e.g. -t 0.85).
"""
import sys
from pathlib import Path

root = Path(__file__).parent
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

import matplotlib.pyplot as plt
import numpy as np

from wakeword.config import load_config, get_project_root
from wakeword.file_test import run_file_test_with_scorecard


def main(threshold=0.70, smoothing=2, bins=20, output="confidence_histogram.png"):
    import warnings

    load_config()
    root = get_project_root()

    print(f"Running file-test (threshold={threshold}, smoothing={smoothing})...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        results = run_file_test_with_scorecard(
            threshold_override=threshold,
            smoothing_windows=smoothing,
        )
    if not results:
        print("No test results.")
        return

    wake_scores = [r["confidence_pct"] / 100.0 for r in results if r["ground_truth"] == "wake"]
    nonwake_scores = [r["confidence_pct"] / 100.0 for r in results if r["ground_truth"] == "nonwake"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(wake_scores, bins=bins, alpha=0.6, color="blue", label=f"Wake (n={len(wake_scores)})", edgecolor="navy")
    ax.hist(nonwake_scores, bins=bins, alpha=0.6, color="red", label=f"Non-wake (n={len(nonwake_scores)})", edgecolor="darkred")
    ax.axvline(x=threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold={threshold}")
    ax.set_xlabel("Confidence (P(wake))")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution: Wake vs Non-wake (Overlap shows where false alarms occur)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = root / output
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Plot confidence histogram for Wake vs Non-wake")
    p.add_argument("-t", "--threshold", type=float, default=0.70, help="Current threshold (shown as vertical line)")
    p.add_argument("-s", "--smoothing", type=int, default=2, help="Smoothing windows for file-test")
    p.add_argument("-b", "--bins", type=int, default=20, help="Histogram bins")
    p.add_argument("-o", "--output", default="confidence_histogram.png", help="Output filename")
    args = p.parse_args()
    main(threshold=args.threshold, smoothing=args.smoothing, bins=args.bins, output=args.output)
