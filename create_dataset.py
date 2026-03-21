import os
import librosa
import numpy as np
import pandas as pd

DATASET_PATH = "dataset"
HARD_NEG_DIR = "hard_negatives"
labels = ["wake", "nonwake"]
max_len = 16000
sr = 16000


def apply_reverb(audio, sr_audio, room_scale=0.25):
    """Synthetic reverb via exponentially decaying noise impulse."""
    impulse_len = int(sr_audio * room_scale)
    t = np.linspace(0, room_scale, impulse_len)
    rng = np.random.default_rng(seed=42)
    impulse = np.exp(-6 * t) * rng.standard_normal(impulse_len).astype(np.float32)
    impulse /= (np.max(np.abs(impulse)) + 1e-8)
    reverbed = np.convolve(audio, impulse, mode="full")[:len(audio)]
    return reverbed.astype(np.float32)


def extract_features(audio, sample_rate):
    """Pad/truncate to 1 sec, extract MFCC, delta, delta-delta, spectral contrast, ZCR, Mel, chroma, RMS."""
    if len(audio) < max_len:
        audio = np.pad(audio, (0, max_len - len(audio)))
    else:
        audio = audio[:max_len]
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

    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma_mean = np.mean(chroma, axis=1)

    rms = librosa.feature.rms(y=audio)
    rms_mean = float(np.mean(rms))

    return (
        list(mfcc_mean) + list(mfcc_std) + list(delta_mean) + list(delta2_mean) + list(delta2_std)
        + list(contrast_mean) + list(contrast_std)
        + [zcr_mean, zcr_std, rms_mean]
        + list(mel_mean) + list(mel_std) + list(chroma_mean)
    )


data = []
manifest = []  # (path, label, aug_type) for hard negative mining

for label in labels:
    folder = os.path.join(DATASET_PATH, label)
    for file in os.listdir(folder):
        if not file.endswith(".wav"):
            continue
        path = os.path.join(folder, file)
        try:
            audio, _ = librosa.load(path, sr=sr)
        except Exception as e:
            print(f"Skipping file: {path} for {e}")
            continue

        if label == "wake":
            # 10 augmentations per wake file for better recall
            noise = np.random.randn(len(audio)).astype(np.float32) * 0.005 * np.std(audio)
            for aug_type, aug_audio in [
                ("original", audio.copy()),
                ("pitch_shift_up2", librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=2)),
                ("pitch_shift_down2", librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=-2)),
                ("pitch_shift_up1", librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=1)),
                ("pitch_shift_down1", librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=-1)),
                ("time_stretch_fast", librosa.effects.time_stretch(y=audio, rate=1.1)),
                ("time_stretch_slow", librosa.effects.time_stretch(y=audio, rate=0.9)),
                ("time_stretch_105", librosa.effects.time_stretch(y=audio, rate=1.05)),
                ("noise", audio + noise),
                ("reverb", apply_reverb(audio, sr)),
            ]:
                row = extract_features(aug_audio, sr) + [label, path]
                data.append(row)
                manifest.append((path, label, aug_type))
        else:
            row = extract_features(audio, sr) + [label, path]
            data.append(row)
            manifest.append((path, label, "original"))

# Targeted hard negative augmentation: 15 versions per file in hard_negatives/
def augment_hard_negatives():
    """Generate 15 augmented nonwake samples per file in hard_negatives/ folder."""
    if not os.path.isdir(HARD_NEG_DIR):
        return [], []
    hndata, hnmanifest = [], []
    for fname in os.listdir(HARD_NEG_DIR):
        if not fname.endswith(".wav"):
            continue
        path = os.path.join(HARD_NEG_DIR, fname)
        try:
            audio, _ = librosa.load(path, sr=sr)
        except Exception as e:
            print(f"Skipping hard negative {path}: {e}")
            continue
        # Use source path for tracking (file in hard_negatives/ is a copy)
        source_path = path
        for i in range(15):
            rng = np.random.default_rng(seed=hash(fname + str(i)) % 2**32)
            aug = audio.copy()
            # Pitch shift: -2 to +2 semitones
            n_steps = rng.integers(-2, 3)
            if n_steps != 0:
                aug = librosa.effects.pitch_shift(y=aug, sr=sr, n_steps=n_steps)
            # Time stretch: 0.9x to 1.1x
            rate = float(rng.uniform(0.9, 1.1))
            aug = librosa.effects.time_stretch(y=aug, rate=rate)
            # Add tiny white noise (SNR ~40dB)
            noise = rng.standard_normal(len(aug)) * 0.01 * np.std(aug)
            aug = aug + noise.astype(np.float32)
            row = extract_features(aug, sr) + ["nonwake", source_path]
            hndata.append(row)
            hnmanifest.append((source_path, "nonwake", f"hard_neg_aug_{i}"))
    return hndata, hnmanifest


hndata, hnmanifest = augment_hard_negatives()
if hndata:
    data.extend(hndata)
    manifest.extend(hnmanifest)
    print(f"Added {len(hndata)} hard negative augmented samples from {HARD_NEG_DIR}/")

# Delta-delta (acceleration) + spectral contrast + ZCR + Mel + chroma + RMS
delta2_cols = [f"delta2_{i}" for i in range(13)] + [f"delta2_std_{i}" for i in range(13)]
contrast_cols = [f"contrast_{i}" for i in range(7)] + [f"contrast_std_{i}" for i in range(7)]
mel_cols = [f"mel_mean_{i}" for i in range(40)] + [f"mel_std_{i}" for i in range(40)]
chroma_cols = [f"chroma_mean_{i}" for i in range(12)]
FEATURE_COLUMNS = (
    [f"mfcc_{i}" for i in range(13)] + [f"mfcc_std_{i}" for i in range(13)]
    + [f"delta_{i}" for i in range(13)] + delta2_cols + contrast_cols
    + ["zcr_mean", "zcr_std", "rms_mean"] + mel_cols + chroma_cols
)
columns = FEATURE_COLUMNS + ["label", "file_path"]
df = pd.DataFrame(data, columns=columns)

df.to_csv("dataset.csv", index=False)
pd.DataFrame(manifest, columns=["path", "label", "aug_type"]).to_csv("dataset_manifest.csv", index=False)

print("dataset.csv created")
print(f"Class counts: {df['label'].value_counts().to_dict()}")