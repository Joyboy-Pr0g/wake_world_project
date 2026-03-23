"""File-based test (no microphone)."""
import os
import sys
import numpy as np
import pandas as pd
import librosa

from .config import load_config
from .features import extract_features, FEATURE_COLUMNS
from .inference import load_artifacts


def features_to_df(features_list, config):
    """Convert feature list to DataFrame for scaler (preserves feature names)."""
    d = dict(zip(FEATURE_COLUMNS, features_list))
    all_cols = config.get("all_feature_cols", config["feature_cols"])
    return pd.DataFrame([[d[c] for c in all_cols]], columns=all_cols)


def safe_print(text):
    """Print text, handling Unicode on Windows console."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))


def process_file(path, model, scaler, config, sr, window_size, hop_size, normalize=True, target_rms=0.05):
    """Process a single audio file and return wake probabilities per window."""
    audio, _ = librosa.load(path, sr=sr)
    if len(audio) < window_size:
        audio = np.pad(audio, (0, window_size - len(audio)))
    all_probs = []
    for start in range(0, len(audio) - window_size + 1, hop_size):
        window = audio[start : start + window_size]
        features_list = extract_features(window, sr, max_len=window_size, normalize=normalize, target_rms=target_rms)
        X_full = features_to_df(features_list, config)
        X_scaled = scaler.transform(X_full)
        selected_mask = config.get("selected_mask")
        if selected_mask is not None:
            X_sel = X_scaled[:, selected_mask]
        else:
            X_sel = X_scaled
        proba = model.predict_proba(X_sel)[0, list(model.classes_).index("wake")]
        all_probs.append(proba)
    if not all_probs:
        window = audio[:window_size]
        features_list = extract_features(window, sr, max_len=window_size, normalize=normalize, target_rms=target_rms)
        X_full = features_to_df(features_list, config)
        X_scaled = scaler.transform(X_full)
        selected_mask = config.get("selected_mask")
        if selected_mask is not None:
            X_sel = X_scaled[:, selected_mask]
        else:
            X_sel = X_scaled
        proba = model.predict_proba(X_sel)[0, list(model.classes_).index("wake")]
        all_probs = [proba]
    return all_probs


def _has_consecutive_high(probs, threshold, n_consecutive):
    """True if probs has at least n_consecutive consecutive values above threshold."""
    run = 0
    for p in probs:
        if p > threshold:
            run += 1
            if run >= n_consecutive:
                return True
        else:
            run = 0
    return False


def run_file_test(threshold_override=None, smoothing_windows=2):
    """Run file-based test on test_samples/ directory.
    threshold_override: if set, use this (recommended 0.70).
    smoothing_windows: require N consecutive high-confidence windows to trigger (default 2)."""
    from .config import get_project_root
    cfg = load_config()
    root = get_project_root()
    ft_cfg = cfg.get("file_test", {})
    paths = cfg["paths"]
    sr = ft_cfg.get("sample_rate", 16000)
    window_size = ft_cfg.get("window_samples", 16000)
    hop_size = ft_cfg.get("hop_samples", 4000)
    test_dir = root / paths.get("test_samples", "test_samples")
    audio_cfg = cfg.get("audio", {})
    normalize = audio_cfg.get("normalize_rms", True)
    target_rms = audio_cfg.get("target_rms", 0.05)

    if not test_dir.is_dir():
        print(f"Folder '{test_dir}/' not found. Create it and add .wav files.")
        return

    print("Loading model, scaler, inference_config...")
    model, scaler, config = load_artifacts()
    threshold = threshold_override if threshold_override is not None else config.get("threshold", 0.70)
    feature_cols = config.get("feature_cols", [])
    print(f"Using threshold: {threshold:.3f}" + (" (override)" if threshold_override is not None else ""))
    print(f"Temporal smoothing: {smoothing_windows} consecutive windows required")
    print(f"Selected features: {len(feature_cols)}")
    print(f"\nScanning {test_dir}/ for .wav files...\n")

    files = [f for f in os.listdir(test_dir) if f.lower().endswith(".wav")]
    if not files:
        print(f"No .wav files found in {test_dir}/")
        return

    for fname in sorted(files):
        path = test_dir / fname
        try:
            probs = process_file(path, model, scaler, config, sr, window_size, hop_size, normalize, target_rms)
        except Exception as e:
            safe_print(f"[{fname}] ERROR: {e}")
            continue
        max_prob = max(probs)
        confidence_pct = max_prob * 100
        triggered = _has_consecutive_high(probs, threshold, smoothing_windows)
        prediction = "wake" if triggered else "nonwake"
        print("-" * 50)
        safe_print(f"File: {fname}")
        print(f"  Wake Confidence (max): {confidence_pct:.1f}%")
        print(f"  Prediction: {prediction}")
        if triggered:
            print()
            print("\033[92m" + "=" * 50)
            print("  TRIGGERED: Hey Pakize detected!")
            print("=" * 50 + "\033[0m")
            print()
        print()
    print("-" * 50)
    print("Done.")
