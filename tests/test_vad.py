import numpy as np
import pytest

from wakeword.vad import is_speech, vad_from_features


def test_is_speech_silence():
    audio = np.zeros(16000, dtype=np.float32)
    assert not is_speech(audio, rms_threshold=0.01)


def test_is_speech_loud():
    np.random.seed(42)
    audio = (np.random.randn(16000).astype(np.float32) * 0.2 + 0.1)
    assert is_speech(audio, rms_threshold=0.01)


def test_vad_from_features_dict():
    assert vad_from_features({"rms_mean": 0.02}, rms_threshold=0.01)
    assert not vad_from_features({"rms_mean": 0.005}, rms_threshold=0.01)
