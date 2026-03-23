"""Simple energy-based Voice Activity Detection."""
import numpy as np


def is_speech(audio, rms_threshold=0.01, zcr_threshold=0.02):
    """Return True if audio segment likely contains speech (not silence).
    Uses RMS and zero-crossing rate as simple energy indicators."""
    if len(audio) == 0:
        return False
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
    zcr = np.mean(np.abs(np.diff(np.sign(audio.astype(np.float64) + 1e-10)))) * 0.5
    return rms >= rms_threshold and zcr >= zcr_threshold


def vad_from_features(features_dict, rms_threshold=0.01):
    """VAD using pre-extracted features. Uses rms_mean if available."""
    if hasattr(features_dict, "get"):
        rms = features_dict.get("rms_mean", 0)
    elif isinstance(features_dict, (list, np.ndarray)):
        # rms_mean is typically at a fixed index in FEATURE_COLUMNS
        from .features import FEATURE_COLUMNS
        try:
            idx = FEATURE_COLUMNS.index("rms_mean")
            rms = features_dict[idx] if len(features_dict) > idx else 0
        except (ValueError, IndexError):
            rms = 0
    else:
        rms = 0
    return rms >= rms_threshold
