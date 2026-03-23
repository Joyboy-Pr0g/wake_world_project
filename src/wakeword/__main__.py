"""CLI entry point: python -m wakeword <command> [args]"""
import sys
from pathlib import Path

# Ensure src is on path when run as python -m wakeword from project root
_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from wakeword.cli import main

if __name__ == "__main__":
    main()
