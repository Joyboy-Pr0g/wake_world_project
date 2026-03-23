"""Backward compatibility: delegates to wakeword package. Run: python train_model.py"""
import sys
from pathlib import Path

root = Path(__file__).parent
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from wakeword.config import load_config
from wakeword.train import train_model, XGBWrapper

# Re-export for file_test_simulation: from train_model import XGBWrapper
__all__ = ["train_model", "XGBWrapper"]


def main():
    load_config()
    train_model()


if __name__ == "__main__":
    main()
