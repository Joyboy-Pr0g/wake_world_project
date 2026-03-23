"""Backward compatibility: delegates to wakeword package. Run: python create_dataset.py"""
import sys
from pathlib import Path

root = Path(__file__).parent
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from wakeword.config import load_config
from wakeword.dataset import build_dataset
from wakeword.features import extract_features, FEATURE_COLUMNS

# Re-export for scripts that do: from create_dataset import extract_features, FEATURE_COLUMNS
__all__ = ["extract_features", "FEATURE_COLUMNS", "build_dataset"]


def main():
    load_config()
    build_dataset()


if __name__ == "__main__":
    main()
