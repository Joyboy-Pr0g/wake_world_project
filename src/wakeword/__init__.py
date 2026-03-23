"""
Wake word detection package.
"""
from .config import load_config, get_project_root
from .features import extract_features, FEATURE_COLUMNS
from .inference import load_artifacts, StreamingWakeDetector, predict_with_threshold
from .dataset import build_dataset

__all__ = [
    "load_config",
    "get_project_root",
    "extract_features",
    "FEATURE_COLUMNS",
    "load_artifacts",
    "StreamingWakeDetector",
    "predict_with_threshold",
    "build_dataset",
]
