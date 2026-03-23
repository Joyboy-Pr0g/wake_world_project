"""Configuration loading from config.yaml."""
from pathlib import Path
import yaml

_CONFIG = None
_PROJECT_ROOT = None


def get_project_root():
    """Return project root (directory containing config.yaml)."""
    global _PROJECT_ROOT
    if _PROJECT_ROOT is not None:
        return _PROJECT_ROOT
    # Assume we're in src/wakeword/ or project root
    p = Path(__file__).resolve().parent
    for _ in range(5):
        if (p / "config.yaml").exists():
            _PROJECT_ROOT = p
            return p
        p = p.parent
    _PROJECT_ROOT = Path.cwd()
    return _PROJECT_ROOT


def load_config(config_path=None):
    """Load config from config.yaml. Returns dict."""
    global _CONFIG
    if _CONFIG is not None and config_path is None:
        return _CONFIG
    root = get_project_root()
    path = Path(config_path) if config_path else root / "config.yaml"
    if not path.exists():
        path = root / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"config.yaml not found (looked in {path})")
    with open(path, encoding="utf-8") as f:
        _CONFIG = yaml.safe_load(f)
    return _CONFIG
