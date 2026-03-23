"""Pytest fixtures and path setup."""
import sys
from pathlib import Path

# Add src to path
root = Path(__file__).resolve().parent.parent
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))
