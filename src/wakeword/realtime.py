import sys
import numpy as np

try:
    import pyaudio
except ImportError:
    print("Install pyaudio: pip install pyaudio")
    print("On Windows, if that fails: pip install pipwin && pipwin install pyaudio")
    sys.exit(1)

from .config import load_config
from .features import extract_features, FEATURE_COLUMNS
from .inference import load_artifacts, StreamingWakeDetector


def features_to_dict(features_list, config):
    d = dict(zip(FEATURE_COLUMNS, features_list))
    all_cols = config.get("all_feature_cols")
    if all_cols:
        return {k: d[k] for k in all_cols if k in d}
    drop = set(config.get("drop_cols", []))
    return {k: d[k] for k in FEATURE_COLUMNS if k not in drop and k in d}


def run_realtime(threshold_override=None, smoothing_windows=None, vad_disabled=False):
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    cfg = load_config()
    audio_cfg = cfg.get("audio", {})
    rt_cfg = cfg.get("realtime", {})
    sr = audio_cfg.get("sample_rate", rt_cfg.get("sample_rate", 16000))
    hop = rt_cfg.get("hop_samples", 4000)
    buffer_size = audio_cfg.get("max_len_samples", rt_cfg.get("buffer_samples", 16000))
    cooldown_sec = rt_cfg.get("cooldown_seconds", 2.0)
    hop_sec = hop / sr
    cooldown_windows = max(0, int(cooldown_sec / hop_sec))
    vad_enabled = False if vad_disabled else rt_cfg.get("vad_enabled", False)
    vad_rms = rt_cfg.get("vad_rms_threshold", 0.01)

    print("Loading model...")
    model, scaler, config = load_artifacts()
    print("Model loaded.")
    if threshold_override is not None:
        config = dict(config)
        config["sequential_threshold"] = threshold_override
    n_consecutive = smoothing_windows if smoothing_windows is not None else rt_cfg.get("sequential_windows", 2)
    detector = StreamingWakeDetector(
        model, scaler, config,
        vad_enabled=vad_enabled,
        vad_rms_threshold=vad_rms,
        cooldown_windows=cooldown_windows,
        sequential_windows_override=n_consecutive,
    )

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sr,
        input=True,
        frames_per_buffer=hop,
    )

    buffer = np.zeros(buffer_size, dtype=np.float32)
    print("Ready. Start talking!")
    hct = config.get("high_confidence_trigger", 0.95)
    print(f"(Trigger: {n_consecutive} consecutive windows above threshold, or immediate if >{hct:.0%}. Ctrl+C to stop)\n")

    try:
        while True:
            raw = stream.read(hop, exception_on_overflow=False)
            chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

            buffer[:-hop] = buffer[hop:]
            buffer[-hop:] = chunk

            audio_cfg = cfg.get("audio", {})
            max_len = audio_cfg.get("max_len_samples", 16000)
            norm_rms = audio_cfg.get("normalize_rms", True)
            target_rms = audio_cfg.get("target_rms", 0.05)
            features_list = extract_features(buffer.copy(), sr, max_len=max_len, normalize=norm_rms, target_rms=target_rms)
            features_dict = features_to_dict(features_list, config)

            triggered, prob = detector.process_window(features_dict)
            if triggered:
                print("\033[92m" + "=" * 40)
                print("Detecting.... Wake word detected!")
                print("=" * 40 + "\033[0m")
                print(f"  (P(wake)={prob:.2f})\n")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


def record_from_mic_until_stop(stop_event):
    """
    Record from default input until stop_event is set (check between chunk reads).
    Returns mono float32 audio in [-1, 1], same sample rate as config.
    """
    cfg = load_config()
    audio_cfg = cfg.get("audio", {})
    rt_cfg = cfg.get("realtime", {})
    sr = audio_cfg.get("sample_rate", rt_cfg.get("sample_rate", 16000))
    hop = rt_cfg.get("hop_samples", 4000)
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sr,
        input=True,
        frames_per_buffer=hop,
    )
    frames = []
    try:
        while not stop_event.is_set():
            raw = stream.read(hop, exception_on_overflow=False)
            frames.append(np.frombuffer(raw, dtype=np.int16))
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
    if not frames:
        return np.array([], dtype=np.float32)
    return np.concatenate(frames).astype(np.float32) / 32768.0


def analyze_recorded_audio(audio, threshold_override=None, smoothing_windows=None):
    """
    Run wake-word detection on recorded buffer (same pipeline as file-test).
    Returns dict: triggered, max_prob, confidence_pct, prediction, lines (list of str for printing).
    """
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    from .file_test import process_audio, _has_consecutive_high

    cfg = load_config()
    audio_cfg = cfg.get("audio", {})
    ft_cfg = cfg.get("file_test", {})
    rt_cfg = cfg.get("realtime", {})
    sr = audio_cfg.get("sample_rate", rt_cfg.get("sample_rate", 16000))
    hop = rt_cfg.get("hop_samples", 4000)
    window_size = audio_cfg.get("max_len_samples", ft_cfg.get("window_samples", 16000))
    norm_rms = audio_cfg.get("normalize_rms", True)
    target_rms = audio_cfg.get("target_rms", 0.05)

    lines = []
    if len(audio) < window_size:
        lines.append(f"Warning: recording very short ({len(audio)} samples). Need at least {window_size} for one window.")

    model, scaler, inf_config = load_artifacts()
    threshold = threshold_override if threshold_override is not None else inf_config.get("threshold", 0.70)
    high_conf = inf_config.get("high_confidence_trigger", 0.95)
    n_consecutive = smoothing_windows if smoothing_windows is not None else rt_cfg.get("sequential_windows", 2)

    probs = process_audio(
        audio, model, scaler, inf_config, sr, window_size, hop,
        normalize=norm_rms, target_rms=target_rms,
    )
    triggered = _has_consecutive_high(probs, threshold, n_consecutive, high_confidence_trigger=high_conf)
    max_prob = max(probs) if probs else 0.0

    lines.append("")
    lines.append("=" * 50)
    if triggered:
        lines.append("*** WAKE WORD DETECTED ***")
    else:
        lines.append("No wake word detected.")
    lines.append(f"  Max confidence: {max_prob:.1%}")
    lines.append(f"  Threshold: {threshold:.2f} (smoothing: {n_consecutive} windows)")
    lines.append("=" * 50)

    return {
        "triggered": triggered,
        "max_prob": max_prob,
        "confidence_pct": max_prob * 100,
        "prediction": "wake" if triggered else "nonwake",
        "lines": lines,
    }
