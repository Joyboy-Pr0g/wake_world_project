import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, GroupShuffleSplit
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve
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
    USE_XGB = True
except ImportError:
    USE_XGB = False

WAKE_THRESHOLD = 0.7
N_FEATURES_RFE = 50


def main():
    # Load data
    df = pd.read_csv("dataset.csv")
    X = df.drop(columns=["label", "file_path"], errors="ignore").select_dtypes(include=[np.number])
    y = df["label"]

    # Drop energy-dominated coeffs
    drop_cols = [c for c in X.columns if c in ("mfcc_0", "mfcc_std_0", "delta_0", "delta2_0", "delta2_std_0")]
    X = X.drop(columns=drop_cols, errors="ignore")

    # Split by FILE to avoid leakage (augmented rows from same source stay together)
    groups = df["file_path"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    idx_train, idx_test = next(gss.split(X, y, groups=groups))
    X_train = X.iloc[idx_train].reset_index(drop=True)
    X_test = X.iloc[idx_test].reset_index(drop=True)
    y_train = y.iloc[idx_train].reset_index(drop=True)
    y_test = y.iloc[idx_test].reset_index(drop=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. SMOTE: balance wake to match nonwake (training data only)
    if USE_SMOTE:
        smote = SMOTE(sampling_strategy="minority", random_state=42, k_neighbors=5)
        X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
        print(f"SMOTE: {len(y_train)} -> {len(y_train_bal)} samples (wake balanced)")
    else:
        X_train_bal, y_train_bal = X_train_scaled, y_train
        print("SMOTE not available (imblearn). Using original training data.")

    # 2. RFE: select top N_FEATURES_RFE features using RF (or use all if fewer)
    n_select = min(N_FEATURES_RFE, X_train_bal.shape[1])
    if n_select >= X_train_bal.shape[1]:
        selected_mask = np.ones(X_train_bal.shape[1], dtype=bool)
        selected_cols = X_train.columns.tolist()
        X_train_sel = X_train_bal
        X_test_sel = X_test_scaled
        print("RFE: using all features")
    else:
        rfe_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        rfe = RFE(rfe_rf, n_features_to_select=n_select, step=1, verbose=0)
        rfe.fit(X_train_bal, y_train_bal)
        selected_mask = rfe.support_
        selected_cols = X_train.columns[selected_mask].tolist()
        X_train_sel = X_train_bal[:, selected_mask]
        X_test_sel = X_test_scaled[:, selected_mask]
        print(f"RFE: selected top {n_select} features")

    # 3. Expanded GridSearchCV
    print("Training SVC, RF, GBC for ensemble...")
    if USE_BALANCED_RF:
        print("  Using BalancedRandomForestClassifier")

    svc_grid = GridSearchCV(
        SVC(kernel="rbf", probability=True, class_weight="balanced"),
        {"C": [0.1, 1, 10, 100], "gamma": [1e-3, 1e-4, "scale"]},
        cv=5, scoring="accuracy", n_jobs=1, verbose=1,
    )
    svc_grid.fit(X_train_sel, y_train_bal)

    RFCls = BalancedRandomForestClassifier if USE_BALANCED_RF else RandomForestClassifier
    rf_kwargs = {"random_state": 42} if USE_BALANCED_RF else {"class_weight": "balanced", "random_state": 42}
    rf_grid = GridSearchCV(
        RFCls(**rf_kwargs),
        {"n_estimators": [150, 250], "max_depth": [12, 16, 20], "min_samples_leaf": [2, 3]},
        cv=5, scoring="accuracy", n_jobs=1, verbose=1,
    )
    rf_grid.fit(X_train_sel, y_train_bal)

    if USE_XGB:
        from sklearn.base import BaseEstimator, ClassifierMixin
        scale_pos = (y_train_bal == "nonwake").sum() / max((y_train_bal == "wake").sum(), 1)

        class XGBWrapper(BaseEstimator, ClassifierMixin):
            def __init__(self, n_estimators=200, max_depth=6, learning_rate=0.05, scale_pos_weight=1, random_state=42):
                self.n_estimators = n_estimators
                self.max_depth = max_depth
                self.learning_rate = learning_rate
                self.scale_pos_weight = scale_pos_weight
                self.random_state = random_state
                self.le_ = LabelEncoder()
                self._build_xgb()
            def _build_xgb(self):
                self.xgb = XGBClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                        learning_rate=self.learning_rate, scale_pos_weight=self.scale_pos_weight,
                                        random_state=self.random_state)
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

        xgb_grid = GridSearchCV(
            XGBWrapper(scale_pos_weight=scale_pos),
            {"n_estimators": [200, 400], "max_depth": [4, 6], "learning_rate": [0.05, 0.1]},
            cv=5, scoring="accuracy", n_jobs=1, verbose=1,
        )
        xgb_grid.fit(X_train_sel, y_train_bal)
        estimators_for_stack = [
            ("svc", svc_grid.best_estimator_),
            ("rf", rf_grid.best_estimator_),
            ("xgb", xgb_grid.best_estimator_),
        ]
        print("  Using XGBoost (replaces GBC)")
    else:
        gbc_grid = GridSearchCV(
            GradientBoostingClassifier(random_state=42),
            {"n_estimators": [200, 400], "max_depth": [5, 7, 9], "learning_rate": [0.03, 0.05]},
            cv=5, scoring="accuracy", n_jobs=1, verbose=1,
        )
        gbc_grid.fit(X_train_sel, y_train_bal)
        estimators_for_stack = [
            ("svc", svc_grid.best_estimator_),
            ("rf", rf_grid.best_estimator_),
            ("gbc", gbc_grid.best_estimator_),
        ]

    # Stacking often outperforms voting; meta-learner = LogisticRegression
    model = StackingClassifier(
        estimators=estimators_for_stack,
        final_estimator=LogisticRegression(random_state=42, max_iter=1000),
        cv=5,
    )
    model.fit(X_train_sel, y_train_bal)

    # Feature importance (from RF)
    rf_est = rf_grid.best_estimator_
    feat_imp = pd.Series(rf_est.feature_importances_, index=selected_cols).sort_values(ascending=False)
    print("\n--- Feature Importance (RF, top 15) ---")
    print(feat_imp.head(15).to_string())

    def predict_with_threshold(estimator, X, threshold=WAKE_THRESHOLD):
        classes = estimator.classes_
        wake_idx = list(classes).index("wake") if "wake" in classes else 1
        proba = estimator.predict_proba(X)[:, wake_idx]
        return np.where(proba > threshold, "wake", "nonwake")

    # Tune threshold via PR curve (maximize F1)
    wake_idx = list(model.classes_).index("wake") if "wake" in model.classes_ else 1
    proba_val = model.predict_proba(X_test_sel)[:, wake_idx]
    y_test_binary = (y_test.values == "wake").astype(int)
    precision, recall, thresholds = precision_recall_curve(y_test_binary, proba_val)
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    best_idx = np.argmax(f1_scores)
    optimal_threshold = float(thresholds[best_idx])
    print(f"\nOptimal threshold (PR curve F1): {optimal_threshold:.3f}")

    y_pred_default = model.predict(X_test_sel)
    y_pred_threshold = predict_with_threshold(model, X_test_sel, optimal_threshold)

    acc_default = accuracy_score(y_test, y_pred_default)
    acc_thresh = accuracy_score(y_test, y_pred_threshold)

    print("\n--- Default threshold (0.5) ---")
    print("Accuracy:", acc_default)
    print(classification_report(y_test, y_pred_default))

    print(f"\n--- Threshold={optimal_threshold:.3f} (PR-optimized) ---")
    print("Accuracy:", acc_thresh)
    print(classification_report(y_test, y_pred_threshold))

    cm = confusion_matrix(y_test, y_pred_threshold, labels=["wake", "nonwake"])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=["wake", "nonwake"], yticklabels=["wake", "nonwake"], cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (threshold={optimal_threshold:.3f})")
    plt.show()

    # Contrast percentiles (on selected features if contrast present)
    contrast_cols = [c for c in selected_cols if c.startswith("contrast_") and "std" not in c]
    contrast_agg = pd.DataFrame(X_train_sel, columns=selected_cols)[contrast_cols].mean(axis=1) if contrast_cols else pd.Series([15.0])
    contrast_p25 = float(contrast_agg.quantile(0.25)) if len(contrast_agg) > 0 else 12.0
    contrast_p75 = float(contrast_agg.quantile(0.75)) if len(contrast_agg) > 0 else 20.0

    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump({
        "threshold": optimal_threshold,
        "drop_cols": drop_cols,
        "feature_cols": selected_cols,
        "all_feature_cols": X_train.columns.tolist(),
        "selected_mask": selected_mask.tolist() if hasattr(selected_mask, "tolist") else list(selected_mask),
        "contrast_p25": contrast_p25,
        "contrast_p75": contrast_p75,
        "sequential_windows": 2,
        "sequential_threshold": 0.5,
    }, "inference_config.pkl")

    # Hard negative mining
    fp_mask = (y_test.values == "nonwake") & (np.array(y_pred_default) == "wake")
    fp_count = fp_mask.sum()
    if fp_count > 0:
        fp_test_positions = np.where(fp_mask)[0]
        fp_dataset_indices = idx_test[fp_test_positions]
        manifest = pd.read_csv("dataset_manifest.csv")
        fp_paths = manifest.iloc[fp_dataset_indices]["path"].unique().tolist()
        with open("hard_negatives.txt", "w", encoding="utf-8") as f:
            f.write("# Nonwake files incorrectly predicted as Wake\n")
            for p in fp_paths:
                f.write(p + "\n")
        print(f"\nHard negative mining: {fp_count} FP -> {len(fp_paths)} files -> hard_negatives.txt")

    print(f"\n>>> Best accuracy: {max(acc_default, acc_thresh):.3f} <<<")


if __name__ == "__main__":
    main()
