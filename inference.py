"""
Production-ready inference: sequential probability, adaptive thresholding.
"""
import numpy as np
import joblib


def predict_with_threshold(model, X_scaled, threshold=0.7, wake_class="wake"):
    """Predict wake only if P(wake) > threshold."""
    classes = model.classes_
    wake_idx = list(classes).index(wake_class)
    proba = model.predict_proba(X_scaled)[:, wake_idx]
    return np.where(proba > threshold, wake_class, "nonwake")


def get_adaptive_threshold(features_row, config, feature_cols):
    """
    Dynamic threshold: noisy (low contrast) -> 0.8, quiet (high contrast) -> 0.5.
    Uses contrast_0..6 mean; interpolates between contrast_p25 (0.8) and contrast_p75 (0.5).
    """
    p25 = config.get("contrast_p25", 12.0)
    p75 = config.get("contrast_p75", 20.0)
    contrast_cols = [c for c in feature_cols if c.startswith("contrast_") and "std" not in c]
    if not contrast_cols:
        return config.get("threshold", 0.7)
    contrast_mean = np.mean([features_row[c] for c in contrast_cols if c in features_row.index])
    # Low contrast (noisy) -> 0.8; high contrast (quiet) -> 0.5
    t = np.clip((contrast_mean - p25) / (p75 - p25 + 1e-6), 0, 1)
    return 0.8 - 0.3 * t  # 0.8 when t=0, 0.5 when t=1


def load_artifacts():
    """Load model, scaler, and inference config."""
    return (
        joblib.load("model.pkl"),
        joblib.load("scaler.pkl"),
        joblib.load("inference_config.pkl"),
    )


def predict_from_features(features_df, model, scaler, config, use_adaptive=True):
    """
    Predict for a DataFrame. Uses adaptive threshold if use_adaptive and config has contrast percentiles.
    """
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
    """
    Sequential probability: require N consecutive windows with P(wake) > seq_threshold.
    Reduces random-spike false positives.
    """

    def __init__(self, model, scaler, config):
        self.model = model
        self.scaler = scaler
        self.config = config
        self.wake_idx = list(model.classes_).index("wake")
        self.n_required = config.get("sequential_windows", 2)
        self.seq_threshold = config.get("sequential_threshold", 0.5)
        self._consecutive_high = 0

    def process_window(self, features_1d):
        """Process one 1-sec window; returns (is_wake_triggered, P(wake)).
        features_1d: dict-like (all features) or array in all_feature_cols order."""
        all_cols = self.config.get("all_feature_cols", self.config["feature_cols"])
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
        proba = self.model.predict_proba(X)[0, self.wake_idx]

        if proba > self.seq_threshold:
            self._consecutive_high += 1
            if self._consecutive_high >= self.n_required:
                self._consecutive_high = 0  # reset after trigger
                return True, proba
        else:
            self._consecutive_high = 0

        return False, proba

    def reset(self):
        self._consecutive_high = 0
