#!/usr/bin/env python
"""Run wakeword CLI without installing. Usage: python run_wakeword.py <command> [args]"""
import sys
from pathlib import Path

root = Path(__file__).parent
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from wakeword.cli import main

if __name__ == "__main__":
    main()
