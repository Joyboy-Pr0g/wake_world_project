import numpy as np
import joblib

from .config import load_config, get_project_root


def predict_with_threshold(model, X_scaled, threshold=0.7, wake_class="wake"):
    classes = model.classes_
    wake_idx = list(classes).index(wake_class)
    proba = model.predict_proba(X_scaled)[:, wake_idx]
    return np.where(proba > threshold, wake_class, "nonwake")


def get_adaptive_threshold(features_row, config, feature_cols):
    p25 = config.get("contrast_p25", 12.0)
    p75 = config.get("contrast_p75", 20.0)
    contrast_cols = [c for c in feature_cols if c.startswith("contrast_") and "std" not in c]
    if not contrast_cols:
        return config.get("threshold", 0.7)
    contrast_mean = np.mean([features_row[c] for c in contrast_cols if c in features_row.index])
    t = np.clip((contrast_mean - p25) / (p75 - p25 + 1e-6), 0, 1)
    return 0.8 - 0.3 * t


def load_artifacts(model_path=None, scaler_path=None, config_path=None):
    try:
        import __main__
        from .train import XGBWrapper
        __main__.XGBWrapper = XGBWrapper
    except Exception:
        pass
    cfg = load_config()
    root = get_project_root()
    model_path = model_path or root / cfg["paths"]["model"]
    scaler_path = scaler_path or root / cfg["paths"]["scaler"]
    config_path = config_path or root / cfg["paths"]["inference_config"]
    return (
        joblib.load(model_path),
        joblib.load(scaler_path),
        joblib.load(config_path),
    )


def predict_from_features(features_df, model, scaler, config, use_adaptive=True):
    all_cols = config.get("all_feature_cols", config["feature_cols"])
    selected_mask = config.get("selected_mask")
    X_full = features_df[all_cols]
    X_scaled_full = scaler.transform(X_full)
    if selected_mask is not None:
        X = X_scaled_full[:, selected_mask]
    else:
        X = X_scaled_full
    feature_cols = config["feature_cols"]

    classes = model.classes_
    wake_idx = list(classes).index("wake")
    proba = model.predict_proba(X)[:, wake_idx]

    contrast_cols = [c for c in feature_cols if c.startswith("contrast_") and "std" not in c]
    if use_adaptive and "contrast_p25" in config and contrast_cols:
        thresholds = np.array([
            get_adaptive_threshold(features_df.iloc[i], config, feature_cols)
            for i in range(len(features_df))
        ])
    else:
        thresholds = np.full(len(features_df), config.get("threshold", 0.7))

    return np.where(proba > thresholds, "wake", "nonwake")


class StreamingWakeDetector:

    def __init__(self, model, scaler, config, vad_enabled=True, vad_rms_threshold=0.01,
                 cooldown_windows=0, sequential_windows_override=None):
        self.model = model
        self.scaler = scaler
        self.config = config
        self.wake_idx = list(model.classes_).index("wake")
        cfg = load_config()
        rt_cfg = cfg.get("realtime", {})
        self.n_required = (
            sequential_windows_override
            if sequential_windows_override is not None
            else config.get("sequential_windows", rt_cfg.get("sequential_windows", 2))
        )
        self.seq_threshold = config.get("sequential_threshold", 0.5)
        self._consecutive_high = 0
        self.vad_enabled = vad_enabled if vad_enabled is not None else rt_cfg.get("vad_enabled", False)
        self.vad_rms_threshold = vad_rms_threshold or rt_cfg.get("vad_rms_threshold", 0.01)
        self.cooldown_windows = cooldown_windows if cooldown_windows is not None else 0
        self._windows_in_cooldown = 0

    def process_window(self, features_1d, skip_vad=False):
        self._windows_in_cooldown = max(0, self._windows_in_cooldown - 1)

        if self.vad_enabled and not skip_vad:
            from .vad import vad_from_features
            if not vad_from_features(features_1d, self.vad_rms_threshold):
                return False, 0.0

        if self._windows_in_cooldown > 0:
            return False, 0.0

        all_cols = self.config.get("all_feature_cols") or self.config.get("feature_cols", [])
        selected_mask = self.config.get("selected_mask")
        if hasattr(features_1d, "keys"):
            X_full = np.array([[features_1d[fc] for fc in all_cols]]).reshape(1, -1)
        else:
            X_full = np.asarray(features_1d).reshape(1, -1)
        X_scaled_full = self.scaler.transform(X_full)
        if selected_mask is not None:
            X = X_scaled_full[:, selected_mask]
        else:
            X = X_scaled_full
        proba = float(self.model.predict_proba(X)[0, self.wake_idx])

        if proba > self.seq_threshold:
            self._consecutive_high += 1
            if self._consecutive_high >= self.n_required:
                self._consecutive_high = 0
                self._windows_in_cooldown = self.cooldown_windows
                return True, proba
        else:
            self._consecutive_high = 0

        return False, proba

    def reset(self):
        self._consecutive_high = 0
        self._windows_in_cooldown = 0
