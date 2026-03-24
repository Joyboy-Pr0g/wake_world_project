"""
Identify Non-wake samples that scored above a confidence threshold (default 0.80),
move them to hard_negatives/, and generate a report by speaker.
"""
import sys
import re
import shutil
import warnings
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

root = Path(__file__).parent
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

import pandas as pd
import numpy as np
import joblib

from wakeword.config import load_config, get_project_root


def extract_speaker(file_path: str) -> str:
    """Extract speaker name from file path. E.g. 'Rabia_nonwake_1.wav' -> 'Rabia'."""
    fname = Path(file_path).stem
    # Remove suffixes like _nonwake_1, _nowake1, _no_wake1, etc.
    match = re.match(r"^(.+?)_(?:nonwake|nowake|no_wake)(?:_?\d+)?$", fname, re.I)
    if match:
        return match.group(1).strip()
    # Fallback: use filename without extension
    return fname


def main(threshold: float = 0.80, dry_run: bool = False):
    load_config()
    proj_root = get_project_root()

    model_path = proj_root / "model.pkl"
    scaler_path = proj_root / "scaler.pkl"
    config_path = proj_root / "inference_config.pkl"
    dataset_csv = proj_root / "dataset.csv"
    hn_dir = proj_root / "hard_negatives"

    if not dataset_csv.exists():
        print(f"ERROR: dataset.csv not found at {dataset_csv}")
        sys.exit(1)
    if not model_path.exists():
        print(f"ERROR: model.pkl not found. Run train_model.py first.")
        sys.exit(1)

    print(f"Loading model, scaler, config from {proj_root}")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    inf_config = joblib.load(config_path)

    print(f"Loading dataset from {dataset_csv}")
    df = pd.read_csv(dataset_csv)

    # Same preprocessing as train.py
    drop_cols = inf_config.get("drop_cols", [])
    all_cols = inf_config.get("all_feature_cols", inf_config.get("feature_cols", []))
    selected_mask = inf_config.get("selected_mask")
    feature_cols = inf_config.get("feature_cols", [])

    X = df.drop(columns=["label", "file_path"], errors="ignore").select_dtypes(include=[np.number])
    X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors="ignore")

    # Ensure column order matches config
    X = X[[c for c in all_cols if c in X.columns]]

    X_scaled = scaler.transform(X)
    if selected_mask is not None:
        X_sel = X_scaled[:, selected_mask]
    else:
        X_sel = X_scaled

    wake_idx = list(model.classes_).index("wake")
    probs = model.predict_proba(X_sel)[:, wake_idx]

    df = df.copy()
    df["prob_wake"] = probs

    # Per-file: take max probability (some files have multiple windows)
    file_probs = df.groupby("file_path").agg(
        max_prob=("prob_wake", "max"),
        label=("label", "first"),
    ).reset_index()

    nonwake_high = file_probs[
        (file_probs["label"] == "nonwake") & (file_probs["max_prob"] >= threshold)
    ].sort_values("max_prob", ascending=False)

    n_found = len(nonwake_high)
    print(f"\nFound {n_found} Non-wake file(s) with confidence >= {threshold:.2f}")

    if n_found == 0:
        print("No hard negatives to move.")
        return

    # Speaker breakdown
    speaker_counts = defaultdict(list)
    for _, row in nonwake_high.iterrows():
        speaker = extract_speaker(row["file_path"])
        speaker_counts[speaker].append((row["file_path"], row["max_prob"]))

    # Create hard_negatives dir and move files
    hn_dir.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "=" * 60,
        "HARD NEGATIVES REPORT - False Positives (Non-wake scoring high)",
        "=" * 60,
        f"Threshold: {threshold:.2f}",
        f"Total files identified: {n_found}",
        "",
        "--- Speakers triggering most False Positives (sorted by count) ---",
        "",
    ]

    for speaker, items in sorted(speaker_counts.items(), key=lambda x: -len(x[1])):
        report_lines.append(f"  {speaker}: {len(items)} file(s)")

    report_lines.extend([
        "",
        "--- Detailed file list ---",
        "",
    ])

    for speaker, items in sorted(speaker_counts.items(), key=lambda x: -len(x[1])):
        report_lines.append(f"\n[{speaker}] ({len(items)} files):")
        for fp, prob in sorted(items, key=lambda x: -x[1]):
            fname = Path(fp).name
            report_lines.append(f"  {fname}  |  confidence: {prob:.2%}")
            if not dry_run:
                src_path = Path(fp)
                if src_path.exists():
                    dst_path = hn_dir / fname
                    try:
                        shutil.move(str(src_path), str(dst_path))
                        report_lines.append(f"    -> moved to hard_negatives/")
                    except Exception as e:
                        report_lines.append(f"    -> ERROR moving: {e}")
                else:
                    report_lines.append(f"    -> WARNING: source file not found")

    if dry_run:
        report_lines.append("\n[DRY RUN - no files were moved]")

    report_text = "\n".join(report_lines)
    try:
        print(report_text)
    except UnicodeEncodeError:
        print(report_text.encode("ascii", "replace").decode("ascii"))

    report_path = proj_root / "hard_negatives_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nReport saved to: {report_path}")

    if not dry_run and n_found > 0:
        print(f"\nMoved {n_found} file(s) to {hn_dir}/")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Identify and move Non-wake files scoring above threshold")
    p.add_argument("-t", "--threshold", type=float, default=0.80, help="Confidence threshold (default: 0.80)")
    p.add_argument("-n", "--dry-run", action="store_true", help="Show report without moving files")
    args = p.parse_args()
    main(threshold=args.threshold, dry_run=args.dry_run)
