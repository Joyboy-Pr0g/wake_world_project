import sys
from pathlib import Path

root = Path(__file__).parent
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from wakeword.config import load_config
from wakeword.collect import collect_hard_negatives


def main():
    load_config()
    collect_hard_negatives()


if __name__ == "__main__":
    main()
