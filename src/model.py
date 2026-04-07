"""
model.py — Train, evaluate, save, load, and predict with all models.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
    r2_score, mean_squared_error, mean_absolute_error,
)
from xgboost import XGBClassifier

from src.config import (
    MODELS_DIR, RANDOM_STATE, TEST_SIZE, CV_FOLDS,
    XGBOOST_PARAMS, RF_PARAMS, TARGET_COL,
)


CLASSIFICATION_MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree":       DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest":       RandomForestClassifier(**RF_PARAMS),
    "Gradient Boosting":   GradientBoostingClassifier(random_state=RANDOM_STATE),
    "XGBoost":             XGBClassifier(**XGBOOST_PARAMS, eval_metric="logloss",
                                          use_label_encoder=False),
}


def prepare_Xy(df: pd.DataFrame, target: str = TARGET_COL):
    """Split DataFrame into X (features) and y (target)."""
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def train_all_classifiers(X_train, y_train) -> dict:
    """Train all classification models and return fitted estimators."""
    fitted = {}
    for name, model in CLASSIFICATION_MODELS.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        print(f"[model] Trained: {name}")
    return fitted


def evaluate_classifiers(fitted: dict, X_test, y_test, cv_X=None, cv_y=None) -> pd.DataFrame:
    """Evaluate all models and return a comparison DataFrame."""
    rows = []
    for name, model in fitted.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average="weighted")
        auc  = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan

        cv_acc = np.nan
        if cv_X is not None and cv_y is not None:
            cv_acc = cross_val_score(model, cv_X, cv_y, cv=CV_FOLDS, scoring="accuracy").mean()

        rows.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "F1 Score": round(f1, 4),
            "AUC-ROC": round(auc, 4) if not np.isnan(auc) else "-",
            f"CV Accuracy ({CV_FOLDS}-fold)": round(cv_acc, 4) if not np.isnan(cv_acc) else "-",
        })
        print(f"[model] {name}: Acc={acc:.4f} | F1={f1:.4f} | AUC={auc:.4f}")
    return pd.DataFrame(rows).sort_values("AUC-ROC", ascending=False)


def save_model(model, name: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name.replace(' ', '_')}.pkl"
    joblib.dump(model, path)
    print(f"[model] Saved -> {path}")
    return path


def load_model(name: str):
    path = MODELS_DIR / f"{name.replace(' ', '_')}.pkl"
    return joblib.load(path)


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Extract feature importances from tree-based models."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_[0])
    else:
        return pd.DataFrame()

    return (
        pd.DataFrame({"Feature": feature_names, "Importance": imp})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
