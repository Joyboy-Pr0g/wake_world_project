"""Model training pipeline."""
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

try:
    from imblearn.ensemble import BalancedRandomForestClassifier
    from imblearn.over_sampling import SMOTE
    USE_BALANCED_RF = True
    USE_SMOTE = True
except ImportError:
    USE_BALANCED_RF = False
    USE_SMOTE = False

try:
    from xgboost import XGBClassifier
    from sklearn.base import BaseEstimator, ClassifierMixin
    USE_XGB = True
except ImportError:
    USE_XGB = False

from .config import load_config, get_project_root


class XGBWrapper(BaseEstimator, ClassifierMixin):
    """XGBoost wrapper for sklearn compatibility."""

    def __init__(self, n_estimators=200, max_depth=6, learning_rate=0.05, scale_pos_weight=1, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.le_ = LabelEncoder()
        self.xgb = None
        self._build_xgb()

    def _build_xgb(self):
        self.xgb = XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, scale_pos_weight=self.scale_pos_weight,
            random_state=self.random_state
        )

    def fit(self, X, y):
        self.le_.fit(y)
        self.xgb.fit(X, self.le_.transform(y))
        self.classes_ = self.le_.classes_
        return self

    def predict(self, X):
        return self.le_.inverse_transform(self.xgb.predict(X).astype(int))

    def predict_proba(self, X):
        return self.xgb.predict_proba(X)

    def get_params(self, deep=True):
        return {"n_estimators": self.n_estimators, "max_depth": self.max_depth,
                "learning_rate": self.learning_rate, "scale_pos_weight": self.scale_pos_weight,
                "random_state": self.random_state}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        self._build_xgb()
        return self


def train_model():
    """Run full training pipeline: load data, train ensemble, save artifacts."""
    cfg = load_config()
    root = get_project_root()
    train_cfg = cfg["train"]
    paths = cfg["paths"]

    df = pd.read_csv(root / paths["dataset_csv"])
    X = df.drop(columns=["label", "file_path"], errors="ignore").select_dtypes(include=[np.number])
    y = df["label"]
    drop_cols = [c for c in X.columns if c in train_cfg.get("drop_cols", [])]
    X = X.drop(columns=drop_cols, errors="ignore")

    groups = df["file_path"].values
    rs = train_cfg["random_state"]
    gss = GroupShuffleSplit(n_splits=1, test_size=train_cfg["test_size"], random_state=rs)
    idx_trainval, idx_test = next(gss.split(X, y, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=train_cfg["val_size"], random_state=0)
    idx_train, idx_val = next(gss2.split(X.iloc[idx_trainval], y.iloc[idx_trainval], groups=groups[idx_trainval]))
    idx_train = idx_trainval[idx_train]
    idx_val = idx_trainval[idx_val]

    X_train = X.iloc[idx_train].reset_index(drop=True)
    X_val = X.iloc[idx_val].reset_index(drop=True)
    X_test = X.iloc[idx_test].reset_index(drop=True)
    y_train = y.iloc[idx_train].reset_index(drop=True)
    y_val = y.iloc[idx_val].reset_index(drop=True)
    y_test = y.iloc[idx_test].reset_index(drop=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    if USE_SMOTE:
        train_counts = pd.Series(y_train).value_counts()
        k = min(5, int(train_counts.min()) - 1)
        if k < 1:
            X_train_bal, y_train_bal = X_train_scaled, y_train
        else:
            smote = SMOTE(sampling_strategy="minority", random_state=rs, k_neighbors=k)
            X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
        print(f"SMOTE: {len(y_train)} -> {len(y_train_bal)} samples (wake balanced)")
    else:
        X_train_bal, y_train_bal = X_train_scaled, y_train

    n_select = min(train_cfg["n_features_rfe"], X_train_bal.shape[1])
    if n_select >= X_train_bal.shape[1]:
        selected_mask = np.ones(X_train_bal.shape[1], dtype=bool)
        selected_cols = X_train.columns.tolist()
        X_train_sel, X_val_sel, X_test_sel = X_train_bal, X_val_scaled, X_test_scaled
    else:
        rfe_rf = RandomForestClassifier(n_estimators=100, random_state=rs, class_weight="balanced")
        rfe = RFE(rfe_rf, n_features_to_select=n_select, step=1, verbose=0)
        rfe.fit(X_train_bal, y_train_bal)
        selected_mask = rfe.support_
        selected_cols = X_train.columns[selected_mask].tolist()
        X_train_sel = X_train_bal[:, selected_mask]
        X_val_sel = X_val_scaled[:, selected_mask]
        X_test_sel = X_test_scaled[:, selected_mask]
    print(f"RFE: selected top {n_select} features")

    thr_min = train_cfg["threshold_search_min"]
    thr_max = train_cfg["threshold_search_max"]
    thr_step = train_cfg["threshold_search_step"]
    min_wr = train_cfg["min_wake_recall"]

    svc_grid = GridSearchCV(
        SVC(kernel="rbf", probability=True, class_weight="balanced"),
        {"C": [0.1, 1, 10, 100], "gamma": [1e-3, 1e-4, "scale"]},
        cv=5, scoring="recall_macro", n_jobs=-1, verbose=1,
    )
    svc_grid.fit(X_train_sel, y_train_bal)

    RFCls = BalancedRandomForestClassifier if USE_BALANCED_RF else RandomForestClassifier
    rf_kwargs = {"random_state": rs} if USE_BALANCED_RF else {"class_weight": "balanced", "random_state": rs}
    rf_grid = GridSearchCV(
        RFCls(**rf_kwargs),
        {"n_estimators": [150, 250], "max_depth": [12, 16, 20], "min_samples_leaf": [2, 3]},
        cv=5, scoring="recall_macro", n_jobs=-1, verbose=1,
    )
    rf_grid.fit(X_train_sel, y_train_bal)

    if USE_XGB:
        scale_pos = (y_train_bal == "nonwake").sum() / max((y_train_bal == "wake").sum(), 1)
        xgb_grid = GridSearchCV(
            XGBWrapper(scale_pos_weight=scale_pos),
            {"n_estimators": [200, 400], "max_depth": [4, 6], "learning_rate": [0.05, 0.1]},
            cv=5, scoring="recall_macro", n_jobs=-1, verbose=1,
        )
        xgb_grid.fit(X_train_sel, y_train_bal)
        estimators_for_stack = [
            ("svc", svc_grid.best_estimator_),
            ("rf", rf_grid.best_estimator_),
            ("xgb", xgb_grid.best_estimator_),
        ]
        print("Using XGBoost in ensemble")
    else:
        gbc_grid = GridSearchCV(
            GradientBoostingClassifier(random_state=rs),
            {"n_estimators": [200, 400], "max_depth": [5, 7], "learning_rate": [0.03, 0.05]},
            cv=5, scoring="recall_macro", n_jobs=-1, verbose=1,
        )
        gbc_grid.fit(X_train_sel, y_train_bal)
        estimators_for_stack = [
            ("svc", svc_grid.best_estimator_),
            ("rf", rf_grid.best_estimator_),
            ("gbc", gbc_grid.best_estimator_),
        ]

    model = StackingClassifier(
        estimators=estimators_for_stack,
        final_estimator=LogisticRegression(random_state=rs, max_iter=1000, class_weight="balanced"),
        cv=5, n_jobs=-1,
    )
    model.fit(X_train_sel, y_train_bal)

    wake_idx = list(model.classes_).index("wake") if "wake" in model.classes_ else 1
    proba_val = model.predict_proba(X_val_sel)[:, wake_idx]

    best_thr, best_fp, best_wr = 0.5, int(1e9), 0.0
    fallback_thr, fallback_fp, fallback_wr = 0.35, int(1e9), 0.0
    for thr in np.arange(thr_min, thr_max, thr_step):
        pp = np.where(proba_val > thr, "wake", "nonwake")
        tp_ = int(((y_val == "wake") & (pp == "wake")).sum())
        fn_ = int(((y_val == "wake") & (pp == "nonwake")).sum())
        fp_ = int(((y_val == "nonwake") & (pp == "wake")).sum())
        wr = tp_ / max(tp_ + fn_, 1)
        if fp_ < fallback_fp:
            fallback_thr, fallback_fp, fallback_wr = float(thr), fp_, wr
        if wr >= min_wr and fp_ < best_fp:
            best_fp, best_thr, best_wr = fp_, float(thr), wr
        elif wr >= min_wr and fp_ == best_fp and wr > best_wr:
            best_thr, best_wr = float(thr), wr

    if best_fp < int(1e9):
        optimal_threshold = best_thr
        print(f"\nOptimal threshold (val, min FP s.t. wr>={min_wr*100:.0f}%): {optimal_threshold:.3f} (val_FP={best_fp}, val_wr={best_wr:.3f})")
    else:
        optimal_threshold = fallback_thr
        print(f"\nOptimal threshold (fallback, min FP): {optimal_threshold:.3f} (val_FP={fallback_fp}, val_wr={fallback_wr:.3f})")

    proba_test = model.predict_proba(X_test_sel)[:, wake_idx]
    y_pred = np.where(proba_test > optimal_threshold, "wake", "nonwake")

    print("\n--- Test set (threshold-optimized) ---")
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=["wake", "nonwake"])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=["wake", "nonwake"], yticklabels=["wake", "nonwake"], cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (threshold={optimal_threshold:.3f})")
    plt.tight_layout()
    cm_path = root / paths["confusion_matrix"]
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    print(f"Saved {paths['confusion_matrix']}")
    plt.close()

    contrast_cols = [c for c in selected_cols if c.startswith("contrast_") and "std" not in c]
    if contrast_cols:
        contrast_agg = pd.DataFrame(X_train_sel, columns=selected_cols)[contrast_cols].mean(axis=1)
        contrast_p25, contrast_p75 = float(contrast_agg.quantile(0.25)), float(contrast_agg.quantile(0.75))
    else:
        contrast_p25, contrast_p75 = 12.0, 20.0

    rt_cfg = cfg.get("realtime", {})
    n_seq = rt_cfg.get("sequential_windows", 2)

    joblib.dump(model, root / paths["model"])
    joblib.dump(scaler, root / paths["scaler"])
    joblib.dump({
        "threshold": optimal_threshold,
        "drop_cols": drop_cols,
        "feature_cols": selected_cols,
        "all_feature_cols": X_train.columns.tolist(),
        "selected_mask": selected_mask.tolist() if hasattr(selected_mask, "tolist") else list(selected_mask),
        "contrast_p25": contrast_p25,
        "contrast_p75": contrast_p75,
        "sequential_windows": n_seq,
        "sequential_threshold": optimal_threshold,
    }, root / paths["inference_config"])

    fp_mask = (y_test.values == "nonwake") & (np.array(y_pred) == "wake")
    fp_count = int(fp_mask.sum())
    if fp_count > 0:
        fp_positions = np.where(fp_mask)[0]
        fp_dataset_indices = idx_test[fp_positions]
        manifest = pd.read_csv(root / paths["dataset_manifest"])
        fp_paths = manifest.iloc[fp_dataset_indices]["path"].unique().tolist()
        hn_txt = root / paths["hard_negatives_txt"]
        with open(hn_txt, "w", encoding="utf-8") as f:
            f.write("# Nonwake files incorrectly predicted as Wake\n")
            for p in fp_paths:
                f.write(str(p) + "\n")
        print(f"Hard negative mining: {fp_count} FP -> {len(fp_paths)} files -> {paths['hard_negatives_txt']}")

    wake_recall = int(((y_test == "wake") & (y_pred == "wake")).sum()) / max(int((y_test == "wake").sum()), 1)
    print(f"\n>>> Wake recall: {wake_recall:.1%}  Accuracy: {acc:.3f}  Threshold: {optimal_threshold:.3f} <<<")

    # Run full evaluation (ROC, PR, report)
    try:
        from .evaluate import run_evaluation
        all_cols = X_train.columns.tolist()
        X_test_full = X_test[all_cols].values
        X_test_scaled_full = scaler.transform(X_test_full)
        run_evaluation(
            X_test_scaled_full,
            y_test.values,
            model,
            scaler,
            {"threshold": optimal_threshold, "feature_cols": selected_cols},
            selected_mask,
            selected_cols,
            optimal_threshold,
        )
    except Exception as e:
        print(f"Evaluation report skipped: {e}")
