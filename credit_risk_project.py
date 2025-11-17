"""
Credit Risk Prediction – Version with automatic scikit‑learn compatibility
This version detects your sklearn version at runtime and selects the correct
OneHotEncoder argument (`sparse` vs `sparse_output`).

Fully compatible with sklearn 0.24 → 1.5+.

Usage:
- Set DATA_PATH and TARGET_COL.
- Run: python credit_risk_project.py
"""

import os
import json
import joblib
import sklearn
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

DATA_PATH = "credit_data.csv"
TARGET_COL = "default"
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_OUTPUT = "best_credit_model.joblib"
RESULTS_JSON = "model_results.json"


# -------- Runtime OneHotEncoder compatibility --------
def make_onehot():
    ver = tuple(int(v) for v in sklearn.__version__.split(".")[:2])
    if ver >= (1, 2):
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# -----------------------------------------------------
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(path)


def summarize_target(y):
    values = y.value_counts().to_dict()
    total = len(y)
    out = {int(k): int(v) for k, v in values.items()}
    out["total"] = int(total)
    return out


# -----------------------------------------------------
def main():
    print("=== CREDIT RISK PREDICTION PIPELINE ===")
    # FIXED broken newline above
    # The line below intentionally left blank

    df = load_data(DATA_PATH)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Columns: {df.columns.tolist()}")

    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    if before != after:
        print(f"Dropped {before-after} duplicate rows")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    print("Target distribution:", summarize_target(y))

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"Numeric: {len(numeric_cols)} columns")
    print(f"Categorical: {len(categorical_cols)} columns")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print("Train target distribution:", summarize_target(y_train))
    print("Test target distribution:", summarize_target(y_test))

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_onehot()),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])

    results = {}

    # Logistic Regression
    print("--- Logistic Regression (balanced) ---")
    lr_pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE))
    ])
    lr_pipe.fit(X_train, y_train)
    lr_pred = lr_pipe.predict(X_test)
    lr_proba = lr_pipe.predict_proba(X_test)[:, 1]
    results["logistic_regression"] = evaluate(y_test, lr_pred, lr_proba)

    # Random Forest (class_weight)
    print("--- RandomForest (balanced) ---")
    rf_pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE))
    ])
    rf_pipe.fit(X_train, y_train)
    rf_pred = rf_pipe.predict(X_test)
    rf_proba = rf_pipe.predict_proba(X_test)[:, 1]
    results["random_forest_class_weight"] = evaluate(y_test, rf_pred, rf_proba)

    # Random Forest + SMOTE
    print("--- RandomForest with SMOTE ---")
    smote = SMOTE(random_state=RANDOM_STATE)
    smote_rf_pipe = ImbPipeline([
        ("prep", preprocessor),
        ("smote", smote),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE))
    ])
    smote_rf_pipe.fit(X_train, y_train)
    smote_pred = smote_rf_pipe.predict(X_test)
    smote_proba = smote_rf_pipe.predict_proba(X_test)[:, 1]
    results["random_forest_smote"] = evaluate(y_test, smote_pred, smote_proba)

    # XGBoost
    if XGBOOST_AVAILABLE:
        print("--- XGBoost ---")
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        scale_pos_weight = neg / max(1, pos)

        xgb_pipe = Pipeline([
            ("prep", preprocessor),
            ("clf", XGBClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
            ))
        ])

        xgb_pipe.fit(X_train, y_train)
        xp = xgb_pipe.predict(X_test)
        xp_proba = xgb_pipe.predict_proba(X_test)[:, 1]
        results["xgboost"] = evaluate(y_test, xp, xp_proba)
    else:
        print("XGBoost not installed → skipping.")

    # Select best
    best_name, best_info = sorted(results.items(), key=lambda x: x[1]["roc_auc"], reverse=True)[0]
    print(f"Best model → {best_name}  (AUC={best_info['roc_auc']:.4f})")

    model_map = {
        "logistic_regression": lr_pipe,
        "random_forest_class_weight": rf_pipe,
        "random_forest_smote": smote_rf_pipe,
    }
    if XGBOOST_AVAILABLE:
        model_map["xgboost"] = xgb_pipe

    best_model = model_map[best_name]
    joblib.dump(best_model, MODEL_OUTPUT)
    print(f"Model saved → {MODEL_OUTPUT}")

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved → {RESULTS_JSON}")

    print("Done.")


# -----------------------------------------------------
def evaluate(y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred)
    roc = roc_auc_score(y_true, y_proba)
    pr_p, pr_r, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(pr_r, pr_p)

    print("Confusion Matrix:", cm)
    print(classification_report(y_true, y_pred))
    print(f"ROC AUC = {roc:.4f}")
    print(f"PR  AUC = {pr_auc:.4f}")

    return {
        "roc_auc": float(roc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist(),
        "report": classification_report(y_true, y_pred, output_dict=True),
    }


if __name__ == "__main__":
    main()
