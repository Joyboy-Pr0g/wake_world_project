import os
import shutil
from pathlib import Path

from .config import load_config, get_project_root
from .features import extract_features, FEATURE_COLUMNS
from .inference import load_artifacts
from .file_test import features_to_df, process_file, safe_print


def is_nonwake_file(fname):
    lower = fname.lower()
    return "nowake" in lower or "nonwake" in lower


def run_mine_hard_negatives(
    input_dir=None,
    confidence_threshold=0.5,
    dry_run=False,
):
    cfg = load_config()
    root = get_project_root()
    paths = cfg["paths"]
    input_dir = Path(input_dir) if input_dir else root / paths.get("test_samples", "test_samples")
    hn_dir = root / paths["hard_negatives"]
    ft_cfg = cfg.get("file_test", {})
    audio_cfg = cfg.get("audio", {})
    sr = ft_cfg.get("sample_rate", 16000)
    window_size = ft_cfg.get("window_samples", 16000)
    hop_size = ft_cfg.get("hop_samples", 4000)
    normalize = audio_cfg.get("normalize_rms", True)
    target_rms = audio_cfg.get("target_rms", 0.05)

    if not input_dir.is_dir():
        print(f"Folder '{input_dir}/' not found.")
        return

    print("Loading model...")
    model, scaler, config = load_artifacts()
    hn_dir.mkdir(parents=True, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".wav")]
    nonwake_files = [f for f in files if is_nonwake_file(f)]

    if not nonwake_files:
        print(f"No nonwake files found in {input_dir}/ (filenames with 'nowake' or 'nonwake').")
        return

    print(f"Scanning {len(nonwake_files)} nonwake files (confidence > {confidence_threshold:.0%} -> hard_negatives)...")
    copied = 0
    for fname in sorted(nonwake_files):
        path = input_dir / fname
        try:
            probs = process_file(path, model, scaler, config, sr, window_size, hop_size, normalize, target_rms)
        except Exception as e:
            safe_print(f"[{fname}] ERROR: {e}")
            continue
        max_prob = max(probs)
        if max_prob > confidence_threshold:
            dst = hn_dir / fname
            if dry_run:
                safe_print(f"  Would copy: {fname} (confidence {max_prob:.1%})")
            else:
                try:
                    shutil.copy2(path, dst)
                    safe_print(f"  Copied: {fname} (confidence {max_prob:.1%})")
                except Exception as e:
                    safe_print(f"  FAILED {fname}: {e}")
                    continue
            copied += 1

    print(f"\nDone. {'Would copy' if dry_run else 'Copied'} {copied} file(s) to {hn_dir}/")
