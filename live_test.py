"""
Live wake-word detection using the microphone.
Uses StreamingWakeDetector with 2 consecutive windows at P(wake) > 0.5.
"""
import sys
import numpy as np

try:
    import pyaudio
except ImportError:
    print("Install pyaudio: pip install pyaudio")
    print("On Windows, if that fails: pip install pipwin && pipwin install pyaudio")
    sys.exit(1)

from create_dataset import extract_features, FEATURE_COLUMNS
from inference import load_artifacts, StreamingWakeDetector

# Audio config (must match training: 16 kHz, mono)
SAMPLE_RATE = 16000
CHUNK = 16000  # 1 second per window
FORMAT = pyaudio.paInt16
CHANNELS = 1

# Sliding window: update every 0.25s with last 1s of audio (reduces latency)
HOP_SAMPLES = 4000  # 0.25 seconds
BUFFER_SIZE = 16000  # 1 second


def features_to_dict(features_list, config):
    """Convert extract_features list to dict. Keys = all_feature_cols (after drop)."""
    d = dict(zip(FEATURE_COLUMNS, features_list))
    all_cols = config.get("all_feature_cols")
    if all_cols:
        return {k: d[k] for k in all_cols if k in d}
    drop = set(config.get("drop_cols", []))
    return {k: d[k] for k in FEATURE_COLUMNS if k not in drop and k in d}


def main():
    print("Loading model...")
    model, scaler, config = load_artifacts()
    detector = StreamingWakeDetector(model, scaler, config)

    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=HOP_SAMPLES,  # Small reads to avoid buffer overflow
    )

    # Ring buffer for sliding 1s window
    buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
    print("Listening... (Ctrl+C to stop)")
    print("Say your wake word twice in a row to trigger.\n")

    try:
        while True:
            # Read 0.25s chunk (non-blocking would need threading; blocking is fine if processing is fast)
            raw = stream.read(HOP_SAMPLES, exception_on_overflow=False)
            chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

            # Slide buffer: shift left, append new
            buffer[:-HOP_SAMPLES] = buffer[HOP_SAMPLES:]
            buffer[-HOP_SAMPLES:] = chunk

            # Extract features from full 1s window
            features_list = extract_features(buffer.copy(), SAMPLE_RATE)
            features_dict = features_to_dict(features_list, config)

            triggered, prob = detector.process_window(features_dict)
            if triggered:
                print("\033[92m" + "=" * 40)
                print("*** WAKE WORD DETECTED ***")
                print("=" * 40 + "\033[0m")
                print(f"  (P(wake)={prob:.2f})\n")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    main()
