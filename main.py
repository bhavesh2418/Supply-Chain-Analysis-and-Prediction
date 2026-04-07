"""
main.py — Full Supply Chain Analysis and Prediction pipeline runner.

Runs: Load -> EDA plots -> Preprocess -> Feature Selection -> Model -> Save results
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RAW_CSV, PROCESSED_CSV, REPORTS_DIR, TARGET_COL,
    TEST_SIZE, RANDOM_STATE, HOLDING_COST_RATE, ORDERING_COST,
    LEAD_TIME_DAYS, SAFETY_STOCK_Z
)
from src.data_loader import load_raw_data, quick_summary
from src.preprocessing import run_full_preprocessing
from src.model import (
    prepare_Xy, train_all_classifiers, evaluate_classifiers,
    save_model, get_feature_importance
)
from src.visualize import (
    plot_missing_values, plot_numeric_distributions,
    plot_correlation_heatmap, plot_target_distribution,
    plot_shipment_mode_performance, plot_warehouse_efficiency,
    plot_inventory_optimization, plot_model_comparison,
    plot_confusion_matrix, plot_roc_curves, plot_feature_importance,
)

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def run_eda(df_raw: pd.DataFrame):
    print("\n=== Phase 3: EDA Plots ===")
    plot_missing_values(df_raw)
    plot_numeric_distributions(df_raw)
    plot_correlation_heatmap(df_raw)
    if TARGET_COL in df_raw.columns:
        plot_target_distribution(df_raw, TARGET_COL)


def run_logistics_analysis(df: pd.DataFrame):
    print("\n=== Logistics Performance Analysis ===")
    # Try common column name variants
    mode_col = next((c for c in df.columns if "Shipment_Mode" in c or "shipping_type" in c.lower()), None)
    delay_col = TARGET_COL if TARGET_COL in df.columns else None

    if mode_col and delay_col:
        plot_shipment_mode_performance(df, mode_col, delay_col)
    else:
        print(f"[main] Skipping shipment mode plot — columns not found (mode={mode_col}, delay={delay_col})")


def run_warehouse_analysis(df: pd.DataFrame):
    print("\n=== Warehouse Efficiency Analysis ===")
    warehouse_col = next((c for c in df.columns if "Warehouse" in c or "warehouse" in c.lower()), None)
    metric_col = next((c for c in df.columns if "Order_Item_Quantity" in c), None)

    if warehouse_col and metric_col:
        plot_warehouse_efficiency(df, warehouse_col, metric_col)
    else:
        print(f"[main] Skipping warehouse plot — columns not found (warehouse={warehouse_col})")


def run_inventory_optimization(df: pd.DataFrame):
    print("\n=== Inventory Optimization (EOQ + Safety Stock) ===")
    qty_col  = next((c for c in df.columns if "Order_Item_Quantity" in c), None)
    cost_col = next((c for c in df.columns if "Product_Price" in c or "Item_Product_Price" in c), None)
    prod_col = next((c for c in df.columns if "Product_Name" in c or "Product_Category" in c), None)

    if not all([qty_col, cost_col, prod_col]):
        print(f"[main] Skipping EOQ — required columns not found.")
        return

    # Compute EOQ and Safety Stock per product
    summary = df.groupby(prod_col).agg(
        Avg_Demand=(qty_col, "mean"),
        Std_Demand=(qty_col, "std"),
        Unit_Cost=(cost_col, "mean"),
    ).reset_index()
    summary.columns = ["Product", "Avg_Demand", "Std_Demand", "Unit_Cost"]
    summary["Std_Demand"] = summary["Std_Demand"].fillna(0)

    summary["EOQ"] = np.sqrt(
        (2 * summary["Avg_Demand"] * ORDERING_COST) /
        (summary["Unit_Cost"] * HOLDING_COST_RATE).replace(0, np.nan)
    ).fillna(0).round(0)

    summary["Safety_Stock"] = (
        SAFETY_STOCK_Z * summary["Std_Demand"] * np.sqrt(LEAD_TIME_DAYS)
    ).round(0)

    summary["Reorder_Point"] = (
        summary["Avg_Demand"] * LEAD_TIME_DAYS + summary["Safety_Stock"]
    ).round(0)

    eoq_path = REPORTS_DIR / "inventory_eoq.csv"
    summary.to_csv(eoq_path, index=False)
    print(f"[main] EOQ results saved -> {eoq_path}")

    plot_inventory_optimization(summary)


def run_model_pipeline(df: pd.DataFrame):
    print("\n=== Supply Chain Delay Prediction ===")
    if TARGET_COL not in df.columns:
        print(f"[main] Target column '{TARGET_COL}' not found — skipping model training.")
        return

    X, y = prepare_Xy(df, TARGET_COL)

    # Drop non-numeric columns if any remain
    X = X.select_dtypes(include="number")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    fitted = train_all_classifiers(X_train, y_train)
    results = evaluate_classifiers(fitted, X_test, y_test, cv_X=X, cv_y=y)

    results_path = REPORTS_DIR / "model_results.csv"
    results.to_csv(results_path, index=False)
    print(f"\n[main] Model results:\n{results.to_string(index=False)}")
    print(f"[main] Results saved -> {results_path}")

    plot_model_comparison(results)
    plot_roc_curves(fitted, X_test, y_test)

    # Best model by AUC
    best_name = results.iloc[0]["Model"]
    best_model = fitted[best_name]
    y_pred = best_model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, best_name)

    imp_df = get_feature_importance(best_model, list(X.columns))
    if not imp_df.empty:
        plot_feature_importance(imp_df, best_name)
        imp_path = REPORTS_DIR / "feature_importance.csv"
        imp_df.to_csv(imp_path, index=False)

    # Save best model
    save_model(best_model, best_name)
    print(f"\n[main] Best model: {best_name} (AUC={results.iloc[0]['AUC-ROC']})")


def main():
    print("=" * 60)
    print("  Supply Chain Analysis and Prediction — Full Pipeline")
    print("=" * 60)

    # Load
    df_raw = load_raw_data()
    quick_summary(df_raw)

    # EDA plots on raw data
    run_eda(df_raw)

    # Preprocess
    df = run_full_preprocessing(df_raw.copy())

    # Analysis modules
    run_logistics_analysis(df)
    run_warehouse_analysis(df)
    run_inventory_optimization(df)

    # ML model
    run_model_pipeline(df)

    print("\n=== Pipeline complete. All outputs saved. ===")


if __name__ == "__main__":
    main()
