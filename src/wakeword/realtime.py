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


def run_record_test(threshold_override=None, smoothing_windows=2, duration_sec=5, interactive=True):
    """
    Record audio from microphone, then run wake-word detection on it (like a test sample).
    Flow: Press Enter to start (or auto-start if interactive=False) → record → process → show result.
    """
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    from .file_test import process_audio, _has_consecutive_high

    cfg = load_config()
    audio_cfg = cfg.get("audio", {})
    rt_cfg = cfg.get("realtime", {})
    ft_cfg = cfg.get("file_test", {})
    sr = audio_cfg.get("sample_rate", rt_cfg.get("sample_rate", 16000))
    hop = rt_cfg.get("hop_samples", 4000)
    window_size = audio_cfg.get("max_len_samples", ft_cfg.get("window_samples", 16000))
    norm_rms = audio_cfg.get("normalize_rms", True)
    target_rms = audio_cfg.get("target_rms", 0.05)

    print("Loading model...")
    model, scaler, config = load_artifacts()
    print("Model loaded.")
    threshold = threshold_override if threshold_override is not None else config.get("threshold", 0.70)
    high_conf = config.get("high_confidence_trigger", 0.95)
    n_consecutive = smoothing_windows if smoothing_windows is not None else rt_cfg.get("sequential_windows", 2)

    n_samples = int(duration_sec * sr)
    print(f"\nRecord-then-test mode: will record {duration_sec} seconds of audio.")
    if interactive:
        print("Press Enter to start recording...")
        try:
            input()
        except EOFError:
            pass
    else:
        import time
        for i in range(3, 0, -1):
            print(f"Recording starts in {i}...")
            time.sleep(1)

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sr,
        input=True,
        frames_per_buffer=hop,
    )

    print(f"Recording for {duration_sec} seconds... Say your wake word now!")
    frames = []
    n_chunks = (n_samples + hop - 1) // hop
    for _ in range(n_chunks):
        raw = stream.read(hop, exception_on_overflow=False)
        frames.append(np.frombuffer(raw, dtype=np.int16))

    stream.stop_stream()
    stream.close()
    p.terminate()

    audio = np.concatenate(frames).astype(np.float32) / 32768.0
    if len(audio) > n_samples:
        audio = audio[:n_samples]

    print("Processing...")
    probs = process_audio(
        audio, model, scaler, config, sr, window_size, hop,
        normalize=norm_rms, target_rms=target_rms,
    )
    triggered = _has_consecutive_high(probs, threshold, n_consecutive, high_confidence_trigger=high_conf)
    max_prob = max(probs)

    print("\n" + "=" * 50)
    if triggered:
        print("\033[92m*** WAKE WORD DETECTED ***\033[0m")
    else:
        print("No wake word detected.")
    print(f"  Max confidence: {max_prob:.1%}")
    print(f"  Threshold: {threshold:.2f} (smoothing: {n_consecutive} windows)")
    print("=" * 50)
