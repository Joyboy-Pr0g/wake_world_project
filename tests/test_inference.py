"""Tests for inference path and threshold/cooldown logic."""
import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from wakeword.inference import StreamingWakeDetector, predict_with_threshold


class MockModel(BaseEstimator, ClassifierMixin):
    """Always returns configurable proba for wake."""

    def __init__(self, wake_proba=0.6):
        self.wake_proba = wake_proba
        self.classes_ = np.array(["nonwake", "wake"])

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        return np.array([[1 - self.wake_proba, self.wake_proba]] * len(X))


@pytest.fixture
def dummy_model():
    return MockModel(wake_proba=0.6)


@pytest.fixture
def mock_scaler():
    """Fitted scaler for 10 features."""
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit(np.random.randn(20, 10))
    return sc


@pytest.fixture
def mock_config():
    cols = [f"f{i}" for i in range(10)]
    return {
        "threshold": 0.5,
        "sequential_windows": 2,
        "sequential_threshold": 0.5,
        "all_feature_cols": cols,
        "feature_cols": cols,
        "selected_mask": [True] * 10,
    }


def test_predict_with_threshold(dummy_model):
    """Threshold filters predictions correctly."""
    high_model = MockModel(wake_proba=0.9)
    low_model = MockModel(wake_proba=0.2)
    X = np.ones((5, 10))
    pred_high = predict_with_threshold(high_model, X, threshold=0.5)
    pred_low = predict_with_threshold(low_model, X, threshold=0.5)
    assert all(p == "wake" for p in pred_high)
    assert all(p == "nonwake" for p in pred_low)


def test_streaming_detector_sequential(dummy_model, mock_scaler, mock_config):
    """Requires N consecutive high windows."""
    detector = StreamingWakeDetector(
        dummy_model, mock_scaler, mock_config,
        vad_enabled=False,
        cooldown_windows=0,
    )
    features_high = {f"f{i}": 1.0 for i in range(10)}
    features_high["rms_mean"] = 0.1
    features_low = {f"f{i}": 0.0 for i in range(10)}
    features_low["rms_mean"] = 0.1

    triggered, _ = detector.process_window(features_high, skip_vad=True)
    assert not triggered
    triggered, _ = detector.process_window(features_high, skip_vad=True)
    assert triggered


def test_streaming_detector_reset(dummy_model, mock_scaler, mock_config):
    """Reset clears consecutive count."""
    detector = StreamingWakeDetector(
        dummy_model, mock_scaler, mock_config,
        vad_enabled=False,
        cooldown_windows=0,
    )
    features = {f"f{i}": 1.0 for i in range(10)}
    features["rms_mean"] = 0.1
    detector.process_window(features, skip_vad=True)
    detector.reset()
    assert detector._consecutive_high == 0


def test_streaming_detector_cooldown(dummy_model, mock_scaler, mock_config):
    """Cooldown blocks immediate re-trigger."""
    detector = StreamingWakeDetector(
        dummy_model, mock_scaler, mock_config,
        vad_enabled=False,
        cooldown_windows=3,
    )
    features = {f"f{i}": 1.0 for i in range(10)}
    features["rms_mean"] = 0.1
    triggered1, _ = detector.process_window(features, skip_vad=True)
    triggered2, _ = detector.process_window(features, skip_vad=True)
    assert triggered2
    for _ in range(2):
        t, _ = detector.process_window(features, skip_vad=True)
        assert not t
