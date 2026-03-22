# Wake Word Detection

Real-time wake word detection using a stacking ensemble (SVC, Random Forest, XGBoost) trained on audio features (MFCC, spectral contrast, mel spectrogram, etc.).

---

## Requirements

- Python 3.8+

### Required Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| librosa | ≥ 0.10.0 | Audio loading and feature extraction |
| soundfile | ≥ 0.12.0 | WAV/FLAC loading (fixes PySoundFile failed warning) |
| numpy | ≥ 1.23.0 | Numerical operations |
| pandas | ≥ 1.5.0 | Dataset handling |
| scikit-learn | ≥ 1.2.0 | ML models, scaling, cross-validation |
| imbalanced-learn | ≥ 0.11.0 | SMOTE, BalancedRandomForest |
| xgboost | ≥ 2.0.0 | Gradient boosting in ensemble |
| joblib | ≥ 1.2.0 | Model and config serialization |
| matplotlib | ≥ 3.6.0 | Plotting (confusion matrix) |
| seaborn | ≥ 0.12.0 | Heatmap visualization |
| pyaudio | ≥ 0.2.13 | Microphone input for live detection |

---

## Installation

```powershell
pip install -r requirements.txt
```

### FFmpeg (optional, system install)

If you use **MP3 or other non-WAV formats**, install FFmpeg so librosa can load them via audioread:

- **Windows:** [ffmpeg.org](https://ffmpeg.org/download.html) or `winget install FFmpeg`
- **macOS:** `brew install ffmpeg`
- **Linux:** `apt install ffmpeg` or `yum install ffmpeg`

For **WAV files only**, `soundfile` in requirements.txt is sufficient.

### Windows – If `pyaudio` fails

```powershell
pip install pipwin
pipwin install pyaudio
```

### Alternative: install libraries manually

```powershell
pip install librosa soundfile numpy pandas scikit-learn imbalanced-learn xgboost joblib matplotlib seaborn pyaudio
```

---

## Project Structure

```
wake_world_project/
├── dataset/
│   ├── wake/          # Wake word .wav files
│   └── nonwake/       # Non-wake background .wav files
├── hard_negatives/    # FP files for hard negative augmentation
├── create_dataset.py  # Build dataset.csv from audio
├── train_model.py     # Train ensemble, save model artifacts
├── collect_hard_negatives.py   # Copy FP files to hard_negatives/
├── inference.py       # Model loading, StreamingWakeDetector
├── live_test.py       # Real-time mic detection
├── realtime_detection.py      # Alias for live_test.py
├── model.pkl          # Trained model (after training)
├── scaler.pkl         # Feature scaler
├── inference_config.pkl       # Threshold, feature config
├── dataset.csv        # Feature dataset
├── dataset_manifest.csv       # Path manifest
└── requirements.txt
```

---

## Usage

### 1. Prepare data

Place `.wav` files in:
- `dataset/wake/` – recordings of the wake word
- `dataset/nonwake/` – background speech / other phrases

### 2. Build dataset

```powershell
python create_dataset.py
```

Creates `dataset.csv` and `dataset_manifest.csv`. Wake samples are augmented; hard negatives from `hard_negatives/` are added with 15 variants each.

### 3. Train model

```powershell
python train_model.py
```

Outputs:
- `model.pkl` – trained ensemble
- `scaler.pkl` – StandardScaler
- `inference_config.pkl` – threshold and feature config
- `confusion_matrix.png` – evaluation
- `hard_negatives.txt` – FP file paths for the next cycle

### 4. Reduce false positives (optional)

```powershell
python collect_hard_negatives.py
python create_dataset.py
python train_model.py
```

### 5. Live detection

```powershell
python live_test.py
```

or:

```powershell
python realtime_detection.py
```

- Uses the default microphone
- 16 kHz, 1 s window, 0.25 s hop
- Requires 2 consecutive windows above threshold to trigger
- Press **Ctrl+C** to stop

---

## Quick Start

```powershell
cd wake_world_project
pip install -r requirements.txt
python create_dataset.py
python train_model.py
python live_test.py
```

---

## License

MIT
