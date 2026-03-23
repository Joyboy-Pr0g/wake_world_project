# Wake Word Evaluation Report

**Generated:** 2026-03-23 05:02  
**Threshold:** 0.270

## Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.8847 |
| Wake Precision | 0.7712 |
| Wake Recall | 0.8667 |
| Wake F1 | 0.8161 |
| Nonwake Precision | 0.9411 |
| Nonwake Recall | 0.8922 |
| Macro F1 | 0.8661 |
| AUC-ROC | 0.9491 |
| AUC-PR (AP) | 0.8682 |

## False Positives

- **FP count:** 54
- **FP/hour (estimated):** 1093.67

## Confusion Matrix

```
              wake  nonwake
wake         182      28
nonwake       54     447
```

## Outputs

- `roc_pr_curves.png` - ROC and PR curves
- `confusion_matrix_eval.png` - Confusion matrix heatmap
