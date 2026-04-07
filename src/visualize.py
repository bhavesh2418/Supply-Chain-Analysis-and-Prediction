"""
visualize.py — All plot functions. Each saves to images/ directory.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc

from src.config import IMAGES_DIR, PLOT_STYLE, FIGURE_DPI, PALETTE

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
plt.style.use(PLOT_STYLE)


def _save(fig, filename: str) -> Path:
    path = IMAGES_DIR / filename
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Saved -> {path}")
    return path


# ── EDA ────────────────────────────────────────────────────────────────────────

def plot_missing_values(df: pd.DataFrame) -> Path:
    missing = df.isnull().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        print("[visualize] No missing values — skipping missing values plot.")
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    missing.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Missing Values per Column")
    ax.set_ylabel("Count")
    ax.set_xlabel("Column")
    plt.xticks(rotation=45, ha="right")
    return _save(fig, "01_missing_values.png")


def plot_numeric_distributions(df: pd.DataFrame, cols: list = None) -> Path:
    num_cols = cols or df.select_dtypes(include="number").columns.tolist()
    n = len(num_cols)
    cols_per_row = 3
    rows = (n + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(cols_per_row * 5, rows * 4))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        axes[i].hist(df[col].dropna(), bins=30, color="steelblue", edgecolor="white")
        axes[i].set_title(col)
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Frequency")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Numeric Feature Distributions", fontsize=14)
    plt.tight_layout()
    return _save(fig, "02_numeric_distributions.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> Path:
    corr = df.select_dtypes(include="number").corr()
    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap")
    return _save(fig, "03_correlation_heatmap.png")


def plot_target_distribution(df: pd.DataFrame, target: str) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df[target].value_counts()
    counts.plot(kind="bar", ax=ax, color=["steelblue", "tomato"], edgecolor="white")
    ax.set_title(f"Target Distribution: {target}")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_xticklabels(["On-Time (0)", "Delayed (1)"], rotation=0)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    return _save(fig, "04_target_distribution.png")


# ── Logistics & Warehouse Analysis ────────────────────────────────────────────

def plot_shipment_mode_performance(df: pd.DataFrame, mode_col: str, delay_col: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    delay_by_mode = df.groupby(mode_col)[delay_col].mean().sort_values()
    delay_by_mode.plot(kind="barh", ax=axes[0], color="steelblue")
    axes[0].set_title("Delay Rate by Shipment Mode")
    axes[0].set_xlabel("Avg Delay Rate")

    count_by_mode = df[mode_col].value_counts()
    count_by_mode.plot(kind="pie", ax=axes[1], autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Shipment Volume by Mode")
    axes[1].set_ylabel("")
    plt.tight_layout()
    return _save(fig, "05_shipment_mode_performance.png")


def plot_warehouse_efficiency(df: pd.DataFrame, warehouse_col: str, metric_col: str) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5))
    efficiency = df.groupby(warehouse_col)[metric_col].mean().sort_values(ascending=False)
    efficiency.plot(kind="bar", ax=ax, color="teal", edgecolor="white")
    ax.set_title(f"Warehouse Efficiency: Avg {metric_col}")
    ax.set_xlabel(warehouse_col)
    ax.set_ylabel(f"Avg {metric_col}")
    plt.xticks(rotation=45, ha="right")
    return _save(fig, "06_warehouse_efficiency.png")


def plot_inventory_optimization(eoq_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    eoq_df.sort_values("EOQ", ascending=False).head(15).plot(
        x="Product", y="EOQ", kind="barh", ax=axes[0], color="darkorange", legend=False
    )
    axes[0].set_title("Top 15 Products by EOQ")
    axes[0].set_xlabel("Economic Order Quantity")

    eoq_df.sort_values("Safety_Stock", ascending=False).head(15).plot(
        x="Product", y="Safety_Stock", kind="barh", ax=axes[1], color="mediumseagreen", legend=False
    )
    axes[1].set_title("Top 15 Products by Safety Stock")
    axes[1].set_xlabel("Safety Stock (units)")
    plt.tight_layout()
    return _save(fig, "07_inventory_optimization.png")


# ── Model Evaluation ──────────────────────────────────────────────────────────

def plot_model_comparison(results_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    results_df.set_index("Model")[["Accuracy", "F1 Score"]].plot(
        kind="bar", ax=ax, color=["steelblue", "tomato"], edgecolor="white"
    )
    ax.set_title("Model Comparison — Accuracy & F1 Score")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    plt.xticks(rotation=30, ha="right")
    ax.set_ylim(0, 1)
    return _save(fig, "08_model_comparison.png")


def plot_confusion_matrix(y_true, y_pred, model_name: str) -> Path:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["On-Time", "Delayed"],
                yticklabels=["On-Time", "Delayed"])
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fname = f"09_confusion_matrix_{model_name.replace(' ', '_')}.png"
    return _save(fig, fname)


def plot_roc_curves(fitted: dict, X_test, y_test) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in fitted.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_title("ROC Curves — All Models")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    return _save(fig, "10_roc_curves.png")


def plot_feature_importance(importance_df: pd.DataFrame, model_name: str, top_n: int = 20) -> Path:
    top = importance_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["Feature"][::-1], top["Importance"][::-1], color="steelblue")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}")
    ax.set_xlabel("Importance")
    return _save(fig, f"11_feature_importance_{model_name.replace(' ', '_')}.png")
