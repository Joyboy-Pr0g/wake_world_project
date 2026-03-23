import numpy as np
import pytest

from wakeword.features import extract_features, FEATURE_COLUMNS, apply_reverb


def test_extract_features_shape():
    np.random.seed(42)
    sr = 16000
    audio = np.random.randn(sr).astype(np.float32) * 0.1
    features = extract_features(audio, sr)
    assert len(features) == len(FEATURE_COLUMNS)


def test_extract_features_padding():
    np.random.seed(42)
    sr = 16000
    audio_short = np.random.randn(8000).astype(np.float32) * 0.1
    features = extract_features(audio_short, sr, max_len=16000)
    assert len(features) == len(FEATURE_COLUMNS)


def test_extract_features_truncation():
    np.random.seed(42)
    sr = 16000
    audio_long = np.random.randn(32000).astype(np.float32) * 0.1
    features = extract_features(audio_long, sr, max_len=16000)
    assert len(features) == len(FEATURE_COLUMNS)


def test_extract_features_deterministic():
    np.random.seed(42)
    sr = 16000
    audio = np.random.randn(sr).astype(np.float32) * 0.1
    f1 = extract_features(audio, sr)
    f2 = extract_features(audio, sr)
    assert f1 == f2


def test_apply_reverb_deterministic():
    np.random.seed(42)
    audio = np.random.randn(16000).astype(np.float32) * 0.1
    r1 = apply_reverb(audio, 16000)
    r2 = apply_reverb(audio, 16000)
    np.testing.assert_array_almost_equal(r1, r2)
