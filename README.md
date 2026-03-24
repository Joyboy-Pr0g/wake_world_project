# Wake Word Detection

Real-time wake word detection ("Hey Pakize") using a stacking ensemble trained on acoustic features. Designed for low-latency, offline-capable deployment.

---

## Overview

This project implements a keyword spotting (KWS) system that:

- Detects a custom wake phrase in real time from microphone input
- Uses traditional ML (no deep learning) for fast inference and small footprint
- Supports an iterative workflow to reduce false positives via hard negative mining
- Produces evaluation reports (ROC/PR curves, FP/hour) for tuning

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Microphone │────▶│ Sliding 1s   │────▶│ Feature     │
│  (16 kHz)   │     │ window 0.25s │     │ Extraction  │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                 │
                                                 ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Trigger   │◀────│ N consecutive│◀────│ Stacking    │
│   (output)  │     │ windows > θ  │     │ Ensemble    │
└─────────────┘     └──────────────┘     └─────────────┘
```

### Components

| Component | Description |
|-----------|-------------|
| **Features** | MFCC (13), deltas, spectral contrast (7), mel (40), chroma (12), ZCR, RMS |
| **Model** | Stacking: SVC + Random Forest + XGBoost → Logistic Regression |
| **Inference** | Threshold tuning, optional VAD gate, cooldown after trigger |
| **Config** | Single `config.yaml` for paths, audio, training, realtime params |

### Pipeline

1. **Dataset** – Build feature CSV from `dataset/wake/` and `dataset/nonwake/` with augmentation
2. **Train** – SMOTE, RFE (65 features), GridSearchCV, threshold optimization
3. **Collect** – Copy false positives to `hard_negatives/` for next iteration
4. **Evaluate** – ROC/PR curves, metrics, FP/hour estimate
5. **Realtime** – Live mic with VAD, cooldown, sequential window requirement

---

## Requirements

- **Python** 3.8+
- See `requirements.txt` for pinned versions

---

## Installation

```powershell
pip install -r requirements.txt
```

**Optional:** Install in editable mode for CLI:

```powershell
pip install -e .
```

**Windows – pyaudio:** If `pip install pyaudio` fails:

```powershell
pip install pipwin
pipwin install pyaudio
```

**FFmpeg** (optional): For MP3/non-WAV support. WAV-only needs `soundfile` only.

---

## Quick Start

```powershell
# 1. Prepare data
#    Add .wav files to dataset/wake/ and dataset/nonwake/

# 2. Build dataset and train
python run_wakeword.py dataset
python run_wakeword.py train

# 3. Run live detection
python run_wakeword.py realtime
```

---

## Usage

### CLI Commands

| Command | Description |
|---------|-------------|
| `dataset` | Build dataset.csv from audio folders |
| `train` | Train model, save artifacts, run evaluation |
| `evaluate` | Generate ROC/PR report (standalone) |
| `realtime` | Live microphone detection |
| `file-test` | Test .wav files in test_samples/ |
| `ui` | Simple Tkinter GUI (realtime + file-test) |
| `collect` | Copy hard_negatives.txt entries to hard_negatives/ |

```powershell
python run_wakeword.py <command>
# or after pip install -e .:
wakeword <command>
```

### Main Commands

| Task | Command |
|------|---------|
| Live mic detection | `python run_wakeword.py realtime -t 0.70 -s 2` |
| Test .wav files | `python run_wakeword.py file-test -t 0.70 -s 2` |
| Simple GUI | `python run_wakeword.py ui` or `python simple_ui.py` |
| Rebuild dataset | `python run_wakeword.py dataset` |
| Train model | `python run_wakeword.py train` |
| Evaluation report | `python run_wakeword.py evaluate` |
| Mine hard negatives | `python run_wakeword.py mine -t 0.5` |

### Configuration

Edit `config.yaml` to adjust:

- **paths** – dataset, model, evaluation output
- **audio** – sample_rate, max_len, n_mfcc, n_mels
- **dataset** – augmentation list, hard_neg count
- **train** – RFE features, threshold search, min recall
- **realtime** – VAD, cooldown, sequential windows

### Evaluation Report

After training (or via `wakeword evaluate`), output appears in `evaluation_report/`:

- `report.md` – Metrics summary
- `report.json` – Machine-readable metrics
- `roc_pr_curves.png` – ROC and PR curves
- `confusion_matrix_eval.png` – Confusion matrix

---

## Model Comparison

| Baseline | Typical performance | Notes |
|----------|---------------------|------|
| **Current (Stacking)** | ~79% wake recall, ~86% accuracy | SVC+RF+XGB, RFE, threshold tuning |
| **Single SVC** | Lower recall | Simpler, faster |
| **Single RF** | Similar | No probability calibration |
| **XGBoost only** | Comparable | Single model, faster train |

For production, consider:

- **Smaller feature set** – Fewer RFE features for lower latency
- **Quantization** – Export to ONNX/OpenVino for edge
- **Deep model** – Small CNN/RNN for higher accuracy (larger footprint)

---

## Testing

```powershell
pytest tests/ -v
```

Tests cover: feature extraction, inference path, threshold/cooldown logic, VAD.

---

## Project Structure

```
wake_world_project/
├── config.yaml
├── src/wakeword/
│   ├── config.py, features.py, dataset.py
│   ├── train.py, inference.py, evaluate.py
│   ├── vad.py, realtime.py, file_test.py, collect.py
│   └── cli.py
├── tests/
├── run_wakeword.py
├── create_dataset.py, train_model.py, ...  # legacy wrappers
├── evaluation_report/
├── dataset/, hard_negatives/, test_samples/
└── model.pkl, scaler.pkl, inference_config.pkl
```

---

## Future Work

- [ ] End-to-end deep model (e.g., small CNN on mel spectrogram)
- [ ] Multi-wake-word support
- [ ] Export to ONNX / TensorFlow Lite for mobile
- [ ] Speaker adaptation / personalization
- [ ] Quantized inference for MCU deployment

---

## License

MIT
