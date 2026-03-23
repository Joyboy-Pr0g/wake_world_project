# Wake Word Evaluation Report

**Generated:** 2026-03-23 03:34  
**Threshold:** 0.180

## Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.8805 |
| Wake Precision | 0.7706 |
| Wake Recall | 0.8476 |
| Wake F1 | 0.8073 |
| Nonwake Precision | 0.9333 |
| Nonwake Recall | 0.8942 |
| Macro F1 | 0.8603 |
| AUC-ROC | 0.9505 |
| AUC-PR (AP) | 0.8926 |

## False Positives

- **FP count:** 53
- **FP/hour (estimated):** 1073.42

## Confusion Matrix

```
              wake  nonwake
wake         178      32
nonwake       53     448
```

## Outputs

- `roc_pr_curves.png` - ROC and PR curves
- `confusion_matrix_eval.png` - Confusion matrix heatmap
