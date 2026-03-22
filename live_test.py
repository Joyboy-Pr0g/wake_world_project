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

SAMPLE_RATE = 16000
CHUNK = 16000
FORMAT = pyaudio.paInt16
CHANNELS = 1

HOP_SAMPLES = 4000
BUFFER_SIZE = 16000


def features_to_dict(features_list, config):
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
        frames_per_buffer=HOP_SAMPLES,
    )

    buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
    print("Listening... (Ctrl+C to stop)")
    print("Say your wake word twice in a row to trigger.\n")

    try:
        while True:
            raw = stream.read(HOP_SAMPLES, exception_on_overflow=False)
            chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

            buffer[:-HOP_SAMPLES] = buffer[HOP_SAMPLES:]
            buffer[-HOP_SAMPLES:] = chunk

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
