import os
import librosa
import numpy as np
import pandas as pd

DATASET_PATH = "dataset"
labels = ["wake", "nonwake"]
max_len = 16000
sr = 16000

def extract_features(audio, sample_rate):
    if len(audio) < max_len:
        audio = np.pad(audio, (0, max_len - len(audio)))
    else:
        audio = audio[:max_len]
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    delta_mean = np.mean(delta, axis=1)
    return list(mfcc_mean) + list(mfcc_std) + list(delta_mean)

data = []

for label in labels:
    folder = os.path.join(DATASET_PATH, label)
    for file in os.listdir(folder):
        if not file.endswith(".wav"):
            continue
        path = os.path.join(folder, file)
        try:
            audio, _ = librosa.load(path, sr=sr)
        except Exception as e:
            continue

        if label == "wake":
            # Original
            row = extract_features(audio.copy(), sr) + [label, path]
            data.append(row)
            # Higher pitch
            audio_pitch = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=2)
            row = extract_features(audio_pitch, sr) + [label, path] # Use same path
            data.append(row)
            # Faster
            audio_fast = librosa.effects.time_stretch(y=audio, rate=1.1)
            row = extract_features(audio_fast, sr) + [label, path] # Use same path
            data.append(row)
        else:
            row = extract_features(audio, sr) + [label, path]
            data.append(row)

# Add "file_path" to columns
columns = [f"mfcc_{i}" for i in range(13)] + [f"mfcc_std_{i}" for i in range(13)] + [f"delta_{i}" for i in range(13)] + ["label", "file_path"]
df = pd.DataFrame(data, columns=columns)
df.to_csv("dataset.csv", index=False)
print("Updated dataset.csv with file_path column.")