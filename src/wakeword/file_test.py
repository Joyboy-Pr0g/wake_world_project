import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import librosa

from .config import load_config
from .features import extract_features, FEATURE_COLUMNS
from .inference import load_artifacts


def features_to_df(features_list, config):
    d = dict(zip(FEATURE_COLUMNS, features_list))
    all_cols = config.get("all_feature_cols", config["feature_cols"])
    return pd.DataFrame([[d[c] for c in all_cols]], columns=all_cols)


def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))


def process_audio(audio, model, scaler, config, sr, window_size, hop_size, normalize=True, target_rms=0.05):
    """Process in-memory mono float32 audio (same pipeline as process_file)."""
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
            X_sel = np.asarray(X_scaled[:, selected_mask])
        else:
            X_sel = np.asarray(X_scaled)
        proba = model.predict_proba(X_sel)[0, list(model.classes_).index("wake")]
        all_probs.append(proba)
    if not all_probs:
        window = audio[:window_size]
        features_list = extract_features(window, sr, max_len=window_size, normalize=normalize, target_rms=target_rms)
        X_full = features_to_df(features_list, config)
        X_scaled = scaler.transform(X_full)
        selected_mask = config.get("selected_mask")
        if selected_mask is not None:
            X_sel = np.asarray(X_scaled[:, selected_mask])
        else:
            X_sel = np.asarray(X_scaled)
        proba = model.predict_proba(X_sel)[0, list(model.classes_).index("wake")]
        all_probs = [proba]
    return all_probs


def process_file(path, model, scaler, config, sr, window_size, hop_size, normalize=True, target_rms=0.05):
    audio, _ = librosa.load(path, sr=sr)
    if len(audio) < window_size:
        audio = np.pad(audio, (0, window_size - len(audio)))
    return process_audio(audio, model, scaler, config, sr, window_size, hop_size, normalize, target_rms)


def _is_nonwake_filename(fname):
    """Infer ground truth from filename: nowake/nonwake = nonwake, else wake."""
    lower = fname.lower()
    return "nowake" in lower or "nonwake" in lower


def _has_consecutive_high(probs, threshold, n_consecutive, high_confidence_trigger=0.95):
    if not probs:
        return False
    if max(probs) >= high_confidence_trigger:
        return True
    run = 0
    for p in probs:
        if p > threshold:
            run += 1
            if run >= n_consecutive:
                return True
        else:
            run = 0
    return False


def test_single_file(path, threshold_override=None, smoothing_windows=2):
    """
    Test a single WAV file. Returns dict with triggered, max_prob, confidence_pct, error.
    For use by UI or APIs.
    """
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    from .config import get_project_root
    cfg = load_config()
    root = get_project_root()
    audio_cfg = cfg.get("audio", {})
    ft_cfg = cfg.get("file_test", {})
    sr = audio_cfg.get("sample_rate", ft_cfg.get("sample_rate", 16000))
    window_size = audio_cfg.get("max_len_samples", ft_cfg.get("window_samples", 16000))
    hop_size = ft_cfg.get("hop_samples", 4000)
    normalize = audio_cfg.get("normalize_rms", True)
    target_rms = audio_cfg.get("target_rms", 0.05)

    path = Path(path) if not isinstance(path, Path) else path
    if not path.exists():
        return {"triggered": False, "error": f"File not found: {path}"}

    try:
        model, scaler, config = load_artifacts()
        threshold = threshold_override if threshold_override is not None else config.get("threshold", 0.70)
        high_conf = config.get("high_confidence_trigger", 0.95)
        probs = process_file(path, model, scaler, config, sr, window_size, hop_size, normalize, target_rms)
        max_prob = max(probs)
        triggered = _has_consecutive_high(probs, threshold, smoothing_windows, high_confidence_trigger=high_conf)
        return {
            "triggered": triggered,
            "max_prob": max_prob,
            "confidence_pct": max_prob * 100,
            "prediction": "wake" if triggered else "nonwake",
            "error": None,
        }
    except Exception as e:
        return {"triggered": False, "error": str(e)}


def run_file_test(threshold_override=None, smoothing_windows=2):
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    from .config import get_project_root
    cfg = load_config()
    root = get_project_root()
    paths = cfg["paths"]
    audio_cfg = cfg.get("audio", {})
    ft_cfg = cfg.get("file_test", {})
    sr = audio_cfg.get("sample_rate", ft_cfg.get("sample_rate", 16000))
    window_size = audio_cfg.get("max_len_samples", ft_cfg.get("window_samples", 16000))
    hop_size = ft_cfg.get("hop_samples", 4000)
    test_dir = root / paths.get("test_samples", "test_samples")
    normalize = audio_cfg.get("normalize_rms", True)
    target_rms = audio_cfg.get("target_rms", 0.05)

    if not test_dir.is_dir():
        print(f"Folder '{test_dir}/' not found. Create it and add .wav files.")
        return

    print("Loading model, scaler, inference_config...")
    model, scaler, config = load_artifacts()
    threshold = threshold_override if threshold_override is not None else config.get("threshold", 0.70)
    high_conf = config.get("high_confidence_trigger", 0.95)
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
            safe_print(f"[TEST] File: {fname} | Confidence: N/A | Status: ERROR - {e}")
            continue
        max_prob = max(probs)
        triggered = _has_consecutive_high(probs, threshold, smoothing_windows, high_confidence_trigger=high_conf)
        status = "TRIGGERED" if triggered else "nonwake"
        safe_print(f"[TEST] File: {fname} | Confidence: {max_prob:.2%} | Status: {status}")
    print("-" * 50)
    print("Done.")


def run_file_test_with_scorecard(threshold_override=None, smoothing_windows=2):
    """
    Run file-test on test_samples/, print per-file results (like run_file_test),
    then return structured data for a scorecard: totals, successes, failures, wrong files.
    """
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    from .config import get_project_root
    cfg = load_config()
    root = get_project_root()
    paths = cfg["paths"]
    audio_cfg = cfg.get("audio", {})
    ft_cfg = cfg.get("file_test", {})
    sr = audio_cfg.get("sample_rate", ft_cfg.get("sample_rate", 16000))
    window_size = audio_cfg.get("max_len_samples", ft_cfg.get("window_samples", 16000))
    hop_size = ft_cfg.get("hop_samples", 4000)
    test_dir = root / paths.get("test_samples", "test_samples")
    normalize = audio_cfg.get("normalize_rms", True)
    target_rms = audio_cfg.get("target_rms", 0.05)

    if not test_dir.is_dir():
        return None

    model, scaler, config = load_artifacts()
    threshold = threshold_override if threshold_override is not None else config.get("threshold", 0.70)
    high_conf = config.get("high_confidence_trigger", 0.95)

    files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".wav")])
    if not files:
        return None

    results = []
    for fname in files:
        path = test_dir / fname
        ground_truth = "nonwake" if _is_nonwake_filename(fname) else "wake"
        try:
            probs = process_file(path, model, scaler, config, sr, window_size, hop_size, normalize, target_rms)
            max_prob = max(probs)
            triggered = _has_consecutive_high(probs, threshold, smoothing_windows, high_confidence_trigger=high_conf)
            prediction = "wake" if triggered else "nonwake"
            correct = prediction == ground_truth
        except Exception as e:
            max_prob = 0.0
            prediction = "error"
            correct = False
        results.append({
            "fname": fname,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "confidence_pct": max_prob * 100,
            "correct": correct,
        })

    return results
