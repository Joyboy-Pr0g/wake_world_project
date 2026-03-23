import sys
from pathlib import Path

root = Path(__file__).parent
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from wakeword.inference import (
    load_artifacts,
    StreamingWakeDetector,
    predict_with_threshold,
    predict_from_features,
    get_adaptive_threshold,
)

__all__ = [
    "load_artifacts",
    "StreamingWakeDetector",
    "predict_with_threshold",
    "predict_from_features",
    "get_adaptive_threshold",
]
