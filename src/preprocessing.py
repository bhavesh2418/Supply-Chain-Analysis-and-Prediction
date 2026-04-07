"""
preprocessing.py — Cleaning, encoding, feature engineering for supply chain data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.config import PROCESSED_CSV, TARGET_COL


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    print(f"[preprocessing] Dropped {before - len(df)} duplicates")
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Median for skewed numeric, mode for categorical."""
    for col in df.select_dtypes(include="number").columns:
        if df[col].isnull().any():
            median = df[col].median()
            df[col] = df[col].fillna(median)
            print(f"[preprocessing] Imputed '{col}' with median={median:.2f}")

    for col in df.select_dtypes(include="object").columns:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"[preprocessing] Imputed '{col}' with mode='{mode_val}'")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode all object columns."""
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"[preprocessing] Encoded '{col}'")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features for supply chain analysis:
    - Delay_Label       : binary target (1 = delayed)
    - Lead_Time_Ratio   : actual vs expected lead time ratio
    - Stock_Cover_Days  : days of inventory cover
    - Order_Fulfillment_Rate : units shipped / units ordered
    - Warehouse_Util_Rate    : stock on hand vs warehouse capacity
    """
    # --- Delay Label (target) ---
    # Works if column like 'Delivery_Status', 'Late_delivery_risk', or similar exists
    if "Late_delivery_risk" in df.columns and TARGET_COL not in df.columns:
        df[TARGET_COL] = df["Late_delivery_risk"].astype(int)
    elif "Delivery_Status" in df.columns and TARGET_COL not in df.columns:
        df[TARGET_COL] = (df["Delivery_Status"] != 0).astype(int)

    # --- Lead Time Ratio ---
    if "Days_for_shipment_(scheduled)" in df.columns and "Days_for_shipping_(real)" in df.columns:
        df["Lead_Time_Ratio"] = (
            df["Days_for_shipping_(real)"] /
            df["Days_for_shipment_(scheduled)"].replace(0, np.nan)
        ).fillna(1.0)

    # --- Order Fulfillment Rate ---
    if "Order_Item_Quantity" in df.columns and "Order_Item_Product_Price" in df.columns:
        df["Revenue_Per_Item"] = (
            df["Order_Item_Quantity"] * df["Order_Item_Product_Price"]
        )

    # --- Profit Margin ---
    if "Order_Item_Profit_Ratio" in df.columns:
        df["Is_High_Margin"] = (df["Order_Item_Profit_Ratio"] > 0.2).astype(int)

    print(f"[preprocessing] Feature engineering complete. Shape: {df.shape}")
    return df


def scale_features(df: pd.DataFrame, feature_cols: list) -> tuple[pd.DataFrame, StandardScaler]:
    """StandardScale numeric features for distance-based models."""
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler


def save_processed(df: pd.DataFrame, path=PROCESSED_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[preprocessing] Saved processed data -> {path}")


def run_full_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_duplicates(df)
    df = impute_missing(df)
    df = engineer_features(df)
    df = encode_categoricals(df)
    save_processed(df)
    return df
