"""Dataset building from audio files."""
import os
import librosa
import numpy as np
import pandas as pd

from .config import load_config, get_project_root
from .features import extract_features, apply_reverb, FEATURE_COLUMNS


def _get_audio_config():
    cfg = load_config()
    return (
        cfg["audio"]["sample_rate"],
        cfg["audio"]["max_len_samples"],
        cfg["audio"].get("normalize_rms", True),
        cfg["audio"].get("target_rms", 0.05),
    )


def augment_hard_negatives():
    """Load and augment hard negative samples. Returns (data_rows, manifest_rows)."""
    cfg = load_config()
    root = get_project_root()
    hn_dir = root / cfg["paths"]["hard_negatives"]
    sr, max_len, norm_rms, target_rms = _get_audio_config()
    hn_aug_count = cfg["dataset"]["hard_neg_aug_count"]

    if not hn_dir.is_dir():
        return [], []

    hndata, hnmanifest = [], []
    for fname in os.listdir(hn_dir):
        if not fname.endswith(".wav"):
            continue
        path = os.path.join(hn_dir, fname)
        try:
            audio, _ = librosa.load(path, sr=sr)
        except Exception as e:
            print(f"Skipping hard negative {path}: {e}")
            continue
        source_path = path
        for i in range(hn_aug_count):
            rng = np.random.default_rng(seed=hash(fname + str(i)) % 2**32)
            aug = audio.copy()
            n_steps = rng.integers(-2, 3)
            if n_steps != 0:
                aug = librosa.effects.pitch_shift(y=aug, sr=sr, n_steps=n_steps)
            rate = float(rng.uniform(0.9, 1.1))
            aug = librosa.effects.time_stretch(y=aug, rate=rate)
            noise = rng.standard_normal(len(aug)) * 0.01 * np.std(aug)
            aug = aug + noise.astype(np.float32)
            row = extract_features(aug, sr, max_len) + ["nonwake", source_path]
            hndata.append(row)
            hnmanifest.append((source_path, "nonwake", f"hard_neg_aug_{i}"))
    return hndata, hnmanifest


def build_dataset():
    """Build dataset.csv and dataset_manifest.csv from audio folders."""
    cfg = load_config()
    root = get_project_root()
    dataset_path = root / cfg["paths"]["dataset"]
    hn_dir = root / cfg["paths"]["hard_negatives"]
    sr, max_len, norm_rms, target_rms = _get_audio_config()
    labels = cfg["labels"]
    aug_list = cfg["dataset"]["wake_augmentations"]
    noise_scale = cfg["dataset"]["noise_scale"]
    reverb_scale = cfg["dataset"]["reverb_room_scale"]

    data = []
    manifest = []

    for label in labels:
        folder = dataset_path / label
        if not folder.is_dir():
            continue
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
                noise = np.random.randn(len(audio)).astype(np.float32) * noise_scale * np.std(audio)
                for aug in aug_list:
                    aug_type = aug.get("type", "unknown")
                    if aug_type == "original":
                        aug_audio = audio.copy()
                    elif aug_type == "pitch_shift":
                        n_steps = aug.get("n_steps", 0)
                        aug_audio = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)
                    elif aug_type == "time_stretch":
                        rate = aug.get("rate", 1.0)
                        aug_audio = librosa.effects.time_stretch(y=audio, rate=rate)
                    elif aug_type == "noise":
                        aug_audio = audio + noise
                    elif aug_type == "reverb":
                        aug_audio = apply_reverb(audio, sr, room_scale=reverb_scale)
                    else:
                        continue
                    row = extract_features(aug_audio, sr, max_len, normalize=norm_rms, target_rms=target_rms) + [label, path]
                    data.append(row)
                    manifest.append((path, label, aug_type))
            else:
                row = extract_features(audio, sr, max_len, normalize=norm_rms, target_rms=target_rms) + [label, path]
                data.append(row)
                manifest.append((path, label, "original"))

    hndata, hnmanifest = augment_hard_negatives()
    if hndata:
        data.extend(hndata)
        manifest.extend(hnmanifest)
        print(f"Added {len(hndata)} hard negative augmented samples from {hn_dir}/")

    columns = FEATURE_COLUMNS + ["label", "file_path"]
    df = pd.DataFrame(data, columns=columns)

    def safe_to_csv(path, dataframe):
        try:
            dataframe.to_csv(path, index=False)
        except PermissionError:
            alt = str(path).replace(".csv", "_new.csv")
            try:
                dataframe.to_csv(alt, index=False)
                os.replace(alt, path)
            except Exception:
                print(f"Permission denied: {path}. Close it in Excel/other apps, or run again.")
                raise

    csv_path = root / cfg["paths"]["dataset_csv"]
    manifest_path = root / cfg["paths"]["dataset_manifest"]

    safe_to_csv(csv_path, df)
    safe_to_csv(manifest_path, pd.DataFrame(manifest, columns=["path", "label", "aug_type"]))
    print(f"{cfg['paths']['dataset_csv']} created")
    print(f"Class counts: {df['label'].value_counts().to_dict()}")
    return df
