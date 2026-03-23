# Wake Word Project – Run Order

## One-time setup

```powershell
pip install -r requirements.txt
```

**Windows:** If `pyaudio` fails to install, try:
```powershell
pip install pipwin
pipwin install pyaudio
```

---

## Pipeline (run in this order)

### Option A: New unified CLI

```powershell
python run_wakeword.py dataset    # Build dataset
python run_wakeword.py train     # Train model (+ evaluation report)
python run_wakeword.py evaluate  # Standalone evaluation (ROC/PR, FP/hour)
python run_wakeword.py mine      # Copy nonwake FPs (>50% conf) to hard_negatives/
python run_wakeword.py collect   # Copy hard_negatives.txt to hard_negatives/
python run_wakeword.py realtime  # Live mic (VAD + cooldown)
python run_wakeword.py file-test -t 0.75  # Test with threshold override (reduce FPs)
```

Or after `pip install -e .`:
```powershell
wakeword dataset
wakeword train
wakeword collect
wakeword realtime
```

### Option B: Original scripts (backward compatible)

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `python create_dataset.py` | Build dataset from `dataset/wake/` and `dataset/nonwake/`, add hard negative augmentation from `hard_negatives/` |
| 2 | `python train_model.py` | Train ensemble (SVC, RF, GBC), save `model.pkl`, `scaler.pkl`, `inference_config.pkl` |
| 3 | `python collect_hard_negatives.py` | Copy FP `.wav` files to `hard_negatives/` for next augmentation cycle |
| 4 | `python live_test.py` | Live mic test with `StreamingWakeDetector` |

---

## Typical workflow

**First run:**
1. `create_dataset.py` – needs `dataset/wake/` and `dataset/nonwake/` with `.wav` files
2. `train_model.py` – produces model artifacts
3. `live_test.py` – test live detection

**After training (to reduce false positives):**
1. `collect_hard_negatives.py` – copies FP files to `hard_negatives/`
2. `create_dataset.py` – adds 10 augmented versions of each hard negative
3. `train_model.py` – retrain with augmented data
4. `live_test.py` – test again

---

## Live test

```powershell
python live_test.py
```

- Uses default microphone
- Sliding window: 0.25 s hop, 1 s window
- Requires **2 consecutive windows** with P(wake) > 0.5 to trigger
- Press **Ctrl+C** to stop
