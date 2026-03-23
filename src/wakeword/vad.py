import numpy as np


def is_speech(audio, rms_threshold=0.01, zcr_threshold=0.02):
    if len(audio) == 0:
        return False
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
    zcr = np.mean(np.abs(np.diff(np.sign(audio.astype(np.float64) + 1e-10)))) * 0.5
    return rms >= rms_threshold and zcr >= zcr_threshold


def vad_from_features(features_dict, rms_threshold=0.01):
    if hasattr(features_dict, "get"):
        rms = features_dict.get("rms_mean", 0)
    elif isinstance(features_dict, (list, np.ndarray)):
        from .features import FEATURE_COLUMNS
        try:
            idx = FEATURE_COLUMNS.index("rms_mean")
            rms = features_dict[idx] if len(features_dict) > idx else 0
        except (ValueError, IndexError):
            rms = 0
    else:
        rms = 0
    return rms >= rms_threshold
