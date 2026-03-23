import sys
import argparse
from pathlib import Path

root = Path(__file__).parent
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from wakeword.config import load_config
from wakeword.file_test import run_file_test


def main():
    p = argparse.ArgumentParser(description="Test .wav files in test_samples/")
    p.add_argument("-t", "--threshold", type=float, default=None, help="Threshold (recommended: 0.70)")
    p.add_argument("-s", "--smoothing", type=int, default=2, help="Consecutive windows required (default: 2)")
    args = p.parse_args()
    load_config()
    run_file_test(threshold_override=args.threshold, smoothing_windows=args.smoothing)


if __name__ == "__main__":
    main()
