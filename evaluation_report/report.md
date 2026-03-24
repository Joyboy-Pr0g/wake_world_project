# Wake Word Evaluation Report

**Generated:** 2026-03-24 07:23  
**Threshold:** 0.500

## Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.9077 |
| Wake Precision | 0.8519 |
| Wake Recall | 0.8050 |
| Wake F1 | 0.8278 |
| Nonwake Precision | 0.9274 |
| Nonwake Recall | 0.9468 |
| Macro F1 | 0.8824 |
| AUC-ROC | 0.9516 |
| AUC-PR (AP) | 0.8366 |

## False Positives

- **FP count:** 28
- **FP/hour (estimated):** 555.37

## Confusion Matrix

```
              wake  nonwake
wake         161      39
nonwake       28     498
```

## Outputs

- `roc_pr_curves.png` - ROC and PR curves
- `confusion_matrix_eval.png` - Confusion matrix heatmap
