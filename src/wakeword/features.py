"""Audio feature extraction."""
import warnings

warnings.filterwarnings("ignore", message="PySoundFile failed")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings("ignore", message="Trying to estimate tuning")

import librosa
import numpy as np

# Feature column names (computed from extract_features output)
delta2_cols = [f"delta2_{i}" for i in range(13)] + [f"delta2_std_{i}" for i in range(13)]
contrast_cols = [f"contrast_{i}" for i in range(7)] + [f"contrast_std_{i}" for i in range(7)]
mel_cols = [f"mel_mean_{i}" for i in range(40)] + [f"mel_std_{i}" for i in range(40)]
mel_band_energy_cols = [f"mel_band_energy_{i}" for i in range(40)]
chroma_cols = [f"chroma_mean_{i}" for i in range(12)]
FEATURE_COLUMNS = (
    [f"mfcc_{i}" for i in range(13)] + [f"mfcc_std_{i}" for i in range(13)]
    + [f"delta_{i}" for i in range(13)] + delta2_cols + contrast_cols
    + ["zcr_mean", "zcr_std", "rms_mean"]
    + ["spectral_centroid_mean", "spectral_centroid_std", "spectral_rolloff_mean", "spectral_rolloff_std"]
    + mel_cols + mel_band_energy_cols + chroma_cols
)


def apply_reverb(audio, sr_audio, room_scale=0.25):
    """Apply simple reverb effect."""
    impulse_len = int(sr_audio * room_scale)
    t = np.linspace(0, room_scale, impulse_len)
    rng = np.random.default_rng(seed=42)
    impulse = np.exp(-6 * t) * rng.standard_normal(impulse_len).astype(np.float32)
    impulse /= (np.max(np.abs(impulse)) + 1e-8)
    reverbed = np.convolve(audio, impulse, mode="full")[:len(audio)]
    return reverbed.astype(np.float32)


def normalize_rms(audio, target_rms=0.05):
    """Scale audio to target RMS for low-volume robustness. Returns copy."""
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
    if rms < 1e-8:
        return audio.astype(np.float32)
    scale = target_rms / rms
    return (audio * scale).astype(np.float32)


def extract_features(audio, sample_rate, max_len=16000, normalize=True, target_rms=0.05):
    """Extract feature vector from audio (MFCC, spectral contrast, mel, chroma, zcr, rms).
    If normalize=True, apply RMS scaling for low-volume robustness."""
    if len(audio) < max_len:
        audio = np.pad(audio, (0, max_len - len(audio)))
    else:
        audio = audio[:max_len]
    if normalize:
        audio = normalize_rms(audio, target_rms)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate, n_bands=6)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    delta_mean = np.mean(delta, axis=1)
    delta2_mean = np.mean(delta2, axis=1)
    delta2_std = np.std(delta2, axis=1)
    contrast_mean = np.mean(contrast, axis=1)
    contrast_std = np.std(contrast, axis=1)

    zcr = librosa.feature.zero_crossing_rate(audio)
    zcr_mean = float(np.mean(zcr))
    zcr_std = float(np.std(zcr))

    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=40)
    mel_db = librosa.power_to_db(mel)
    mel_mean = np.mean(mel_db, axis=1)
    mel_std = np.std(mel_db, axis=1)
    mel_band_energy = np.mean(mel, axis=1)

    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    spectral_centroid_mean = float(np.mean(spectral_centroid))
    spectral_centroid_std = float(np.std(spectral_centroid))
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
    spectral_rolloff_mean = float(np.mean(spectral_rolloff))
    spectral_rolloff_std = float(np.std(spectral_rolloff))

    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma_mean = np.mean(chroma, axis=1)

    rms = librosa.feature.rms(y=audio)
    rms_mean = float(np.mean(rms))

    return (
        list(mfcc_mean) + list(mfcc_std) + list(delta_mean) + list(delta2_mean) + list(delta2_std)
        + list(contrast_mean) + list(contrast_std)
        + [zcr_mean, zcr_std, rms_mean]
        + [spectral_centroid_mean, spectral_centroid_std, spectral_rolloff_mean, spectral_rolloff_std]
        + list(mel_mean) + list(mel_std) + list(mel_band_energy)
        + list(chroma_mean)
    )
