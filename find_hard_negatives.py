import pandas as pd
import joblib
import os
import shutil
import numpy as np

OUTPUT_FOLDER = "hard_negatives"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
df = pd.read_csv("dataset2.csv")

X = df.drop(["label", "file_path"], axis=1)
drop_cols = [c for c in X.columns if c in ("mfcc_0", "mfcc_std_0", "delta_0")]
X = X.drop(columns=drop_cols, errors="ignore")

X_scaled = scaler.transform(X)
wake_idx = list(model.classes_).index("wake")
probs = model.predict_proba(X_scaled)[:, wake_idx]

THRESHOLD = 0.7
df['probability'] = probs
hard_negatives = df[(df['label'] == 'nonwake') & (df['probability'] >= THRESHOLD)]

print(f"Found {len(hard_negatives)} hard negatives.")

with open("hard_negative_report.txt", "w") as f:
    f.write("Files that tricked the model (False Positives):\n")
    f.write("-" * 50 + "\n")
    
    for i, row in hard_negatives.iterrows():
        original_path = row['file_path']
        prob_score = row['probability']
        filename = os.path.basename(original_path)
        
        dest_path = os.path.join(OUTPUT_FOLDER, f"prob_{prob_score:.2f}_{filename}")
        
        if os.path.exists(original_path):
            shutil.copy(original_path, dest_path)
            f.write(f"File: {filename} | Wake Probability: {prob_score:.4f}\n")
        else:
            f.write(f"ERROR: Could not find {original_path}\n")

print(f"Done! Check the '{OUTPUT_FOLDER}' folder to hear the sounds.")