"""Evaluation: ROC/PR curves, metrics, FP/hour, report generation."""
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

from .config import load_config, get_project_root
from .inference import load_artifacts


def compute_metrics(y_true, y_pred, labels=None):
    """Compute precision, recall, F1, accuracy, confusion matrix."""
    labels = labels or ["wake", "nonwake"]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_wake": float(precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)[labels.index("wake")]),
        "precision_nonwake": float(precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)[labels.index("nonwake")]),
        "recall_wake": float(recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)[labels.index("wake")]),
        "recall_nonwake": float(recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)[labels.index("nonwake")]),
        "f1_wake": float(f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)[labels.index("wake")]),
        "f1_nonwake": float(f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)[labels.index("nonwake")]),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def fp_per_hour(n_fp, n_test_windows, window_sec=1.0, hop_sec=0.25):
    """Estimate FP per hour given FP count and total windows.
    Assumes sliding window: window_sec=1.0, hop_sec=0.25 -> 4 windows/sec."""
    windows_per_hour = 3600 / hop_sec
    if n_test_windows <= 0:
        return 0.0
    return float(n_fp * windows_per_hour / n_test_windows)


def run_evaluation(
    X_test,
    y_test,
    model,
    scaler,
    config,
    selected_mask,
    selected_cols,
    threshold,
    output_dir=None,
):
    """Full evaluation: ROC, PR, metrics, confusion matrix, FP/hour."""
    X_test_sel = X_test[:, selected_mask] if selected_mask is not None else X_test
    proba = model.predict_proba(X_test_sel)
    wake_idx = list(model.classes_).index("wake")
    y_proba = proba[:, wake_idx]
    y_pred = np.where(y_proba > threshold, "wake", "nonwake")

    labels = ["wake", "nonwake"]
    metrics = compute_metrics(y_test, y_pred, labels)
    cm = np.array(metrics["confusion_matrix"])
    n_fp = int(((np.array(y_test) == "nonwake") & (np.array(y_pred) == "wake")).sum())
    metrics["false_positives"] = n_fp
    metrics["fp_per_hour"] = fp_per_hour(n_fp, len(y_test))
    metrics["threshold"] = threshold
    metrics["n_test_samples"] = len(y_test)

    try:
        y_bin = (np.array(y_test) == "wake").astype(int)
        auc_roc = roc_auc_score(y_bin, y_proba)
        auc_pr = average_precision_score(y_bin, y_proba)
    except Exception:
        auc_roc = 0.0
        auc_pr = 0.0
    metrics["auc_roc"] = float(auc_roc)
    metrics["auc_pr"] = float(auc_pr)

    fpr, tpr, _ = roc_curve((np.array(y_test) == "wake").astype(int), y_proba)
    prec, rec, _ = precision_recall_curve((np.array(y_test) == "wake").astype(int), y_proba)

    cfg = load_config()
    root = get_project_root()
    out = Path(output_dir) if output_dir else root / cfg["paths"].get("evaluation_dir", "evaluation_report")
    out.mkdir(parents=True, exist_ok=True)

    # ROC curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(fpr, tpr, "b-", label=f"ROC (AUC={auc_roc:.3f})")
    ax1.plot([0, 1], [0, 1], "k--")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # PR curve
    ax2.plot(rec, prec, "b-", label=f"PR (AP={auc_pr:.3f})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out / "roc_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 4))
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (threshold={threshold:.3f})")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix_eval.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Report MD
    report_md = _format_report_md(metrics, threshold)
    (out / "report.md").write_text(report_md, encoding="utf-8")

    # Report JSON
    (out / "report.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics, out


def _format_report_md(metrics, threshold):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    cm = metrics["confusion_matrix"]
    return f"""# Wake Word Evaluation Report

**Generated:** {ts}  
**Threshold:** {threshold:.3f}

## Metrics

| Metric | Value |
|--------|-------|
| Accuracy | {metrics['accuracy']:.4f} |
| Wake Precision | {metrics['precision_wake']:.4f} |
| Wake Recall | {metrics['recall_wake']:.4f} |
| Wake F1 | {metrics['f1_wake']:.4f} |
| Nonwake Precision | {metrics['precision_nonwake']:.4f} |
| Nonwake Recall | {metrics['recall_nonwake']:.4f} |
| Macro F1 | {metrics['f1_macro']:.4f} |
| AUC-ROC | {metrics['auc_roc']:.4f} |
| AUC-PR (AP) | {metrics['auc_pr']:.4f} |

## False Positives

- **FP count:** {metrics['false_positives']}
- **FP/hour (estimated):** {metrics['fp_per_hour']:.2f}

## Confusion Matrix

```
              wake  nonwake
wake      {cm[0][0]:>6}  {cm[0][1]:>6}
nonwake   {cm[1][0]:>6}  {cm[1][1]:>6}
```

## Outputs

- `roc_pr_curves.png` - ROC and PR curves
- `confusion_matrix_eval.png` - Confusion matrix heatmap
"""


def run_evaluate_command():
    """CLI: load model, dataset, run evaluation, save report."""
    cfg = load_config()
    root = get_project_root()
    paths = cfg["paths"]
    train_cfg = cfg["train"]

    df = pd.read_csv(root / paths["dataset_csv"])
    X = df.drop(columns=["label", "file_path"], errors="ignore").select_dtypes(include=[np.number])
    y = df["label"]
    drop_cols = [c for c in X.columns if c in train_cfg.get("drop_cols", [])]
    X = X.drop(columns=drop_cols, errors="ignore")

    groups = df["file_path"].values
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=train_cfg["test_size"], random_state=train_cfg["random_state"])
    _, idx_test = next(gss.split(X, y, groups=groups))
    X_test = X.iloc[idx_test].reset_index(drop=True)
    y_test = y.iloc[idx_test].reset_index(drop=True)

    model, scaler, config = load_artifacts()
    selected_cols = config["feature_cols"]
    all_cols = config.get("all_feature_cols", selected_cols)
    selected_mask = np.array(config.get("selected_mask", [True] * len(selected_cols)))
    X_full = X_test[all_cols]
    X_scaled = scaler.transform(X_full)
    X_test_sel = X_scaled[:, selected_mask] if selected_mask is not None else X_scaled
    threshold = config["threshold"]

    metrics, out_dir = run_evaluation(
        np.asarray(X_scaled),
        y_test.values,
        model,
        scaler,
        config,
        selected_mask,
        selected_cols,
        threshold,
    )

    print(f"\nEvaluation saved to {out_dir}/")
    print(f"Wake Recall: {metrics['recall_wake']:.1%}  FP/hour: {metrics['fp_per_hour']:.2f}")
