"""
Augment hard negative samples to reduce false positives.
Run after train_model.py produces hard_negatives.txt.
Adds pitch_shift and time_stretch variants of FP nonwake files to the dataset.
"""
import os
import sys

# Ensure we can append to dataset; run create_dataset after this adds new rows
# This script outputs a list of augmented rows to append to dataset.csv
# For a full pipeline: run create_dataset with an option to include hard negative augmentation

DATASET_PATH = "dataset"
sr = 16000
max_len = 16000

if not os.path.exists("hard_negatives.txt"):
    print("Run train_model.py first to generate hard_negatives.txt")
    sys.exit(1)

with open("hard_negatives.txt") as f:
    paths = [line.strip() for line in f if line.strip() and not line.startswith("#")]

if not paths:
    print("No hard negatives to augment.")
    sys.exit(0)

import librosa
import numpy as np
import pandas as pd

# Reuse extract_features from create_dataset
def extract_features(audio, sample_rate):
    if len(audio) < max_len:
        audio = np.pad(audio, (0, max_len - len(audio)))
    else:
        audio = audio[:max_len]
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate, n_bands=6)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    delta_mean = np.mean(delta, axis=1)
    contrast_mean = np.mean(contrast, axis=1)
    contrast_std = np.std(contrast, axis=1)
    return list(mfcc_mean) + list(mfcc_std) + list(delta_mean) + list(contrast_mean) + list(contrast_std)

data = []
manifest = []
for path in paths:
    if not os.path.exists(path):
        continue
    try:
        audio, _ = librosa.load(path, sr=sr)
    except Exception as e:
        print(f"Skipping {path}: {e}")
        continue
    for aug_type, aug_audio in [
        ("pitch_shift", librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=-1)),
        ("time_stretch", librosa.effects.time_stretch(y=audio, rate=0.9)),
    ]:
        row = extract_features(aug_audio, sr) + ["nonwake"]
        data.append(row)
        manifest.append((path, "nonwake", aug_type))

if not data:
    print("No valid files augmented.")
    sys.exit(0)

contrast_cols = [f"contrast_{i}" for i in range(7)] + [f"contrast_std_{i}" for i in range(7)]
columns = [f"mfcc_{i}" for i in range(13)] + [f"mfcc_std_{i}" for i in range(13)] + [f"delta_{i}" for i in range(13)] + contrast_cols + ["label"]
new_df = pd.DataFrame(data, columns=columns)

# Append to dataset.csv
df = pd.read_csv("dataset.csv")
df = pd.concat([df, new_df], ignore_index=True)
df.to_csv("dataset.csv", index=False)

# Append to manifest
manifest_df = pd.read_csv("dataset_manifest.csv")
manifest_df = pd.concat([manifest_df, pd.DataFrame(manifest, columns=["path", "label", "aug_type"])], ignore_index=True)
manifest_df.to_csv("dataset_manifest.csv", index=False)

print(f"Added {len(data)} augmented nonwake samples from {len(paths)} hard negative files.")
print("Re-run train_model.py to retrain with augmented data.")
