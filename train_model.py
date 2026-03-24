import sys
import warnings
from pathlib import Path

root = Path(__file__).parent
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from wakeword.config import load_config
from wakeword.train import train_model, XGBWrapper

__all__ = ["train_model", "XGBWrapper"]


def main():
    load_config()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        train_model()


if __name__ == "__main__":
    main()
