"""
Copy FALSE ALARM files from test_samples/ to dataset/nonwake/ for retraining.
These are non-wake files that the model incorrectly predicted as wake.
Run file-test first to identify them, then run this script to add them to training data.
"""
import sys
import shutil
from pathlib import Path

root = Path(__file__).parent
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from wakeword.config import load_config, get_project_root
from wakeword.file_test import run_file_test_with_scorecard


def main(threshold=0.70, smoothing=2, dry_run=False):
    load_config()
    root = get_project_root()
    cfg = load_config()
    paths = cfg["paths"]
    test_dir = root / paths.get("test_samples", "test_samples")
    nonwake_dir = root / paths["dataset"] / "nonwake"

    if not test_dir.is_dir():
        print(f"test_samples/ not found: {test_dir}")
        return

    nonwake_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running file-test (threshold={threshold}, smoothing={smoothing})...")
    results = run_file_test_with_scorecard(
        threshold_override=threshold,
        smoothing_windows=smoothing,
    )
    if not results:
        print("No test results.")
        return

    false_alarms = [
        r for r in results
        if r["ground_truth"] == "nonwake" and r["prediction"] == "wake"
    ]

    if not false_alarms:
        print("No FALSE ALARMS found. Model is not triggering on non-wake files.")
        return

    print(f"\nFound {len(false_alarms)} FALSE ALARM(s):")
    for r in false_alarms:
        print(f"  - {r['fname']} (confidence: {r['confidence_pct']:.1f}%)")

    print(f"\n{'[DRY RUN] Would copy' if dry_run else 'Copying'} to {nonwake_dir}/")
    for r in false_alarms:
        src_path = test_dir / r["fname"]
        dst_path = nonwake_dir / r["fname"]
        if src_path.exists():
            if dry_run:
                print(f"  Would copy: {r['fname']}")
            else:
                shutil.copy2(src_path, dst_path)
                print(f"  Copied: {r['fname']}")
        else:
            print(f"  SKIP (not found): {r['fname']}")

    if not dry_run and false_alarms:
        print(f"\nDone. {len(false_alarms)} file(s) added to dataset/nonwake/.")
        print("Next: run 'python run_wakeword.py dataset' then 'python run_wakeword.py train' to retrain.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Copy FALSE ALARM files to dataset/nonwake for retraining")
    p.add_argument("-t", "--threshold", type=float, default=0.70, help="Detection threshold")
    p.add_argument("-s", "--smoothing", type=int, default=2, help="Consecutive windows required")
    p.add_argument("--dry-run", action="store_true", help="Only print what would be copied")
    args = p.parse_args()
    main(threshold=args.threshold, smoothing=args.smoothing, dry_run=args.dry_run)
