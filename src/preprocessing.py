"""
preprocessing.py — Cleaning, encoding, feature engineering for supply chain data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.config import (
    PROCESSED_CSV, TARGET_COL,
    COL_DELIVERY_STATUS, COL_QUANTITY, COL_UNIT_PRICE,
    COL_TOTAL_COST, COL_DELIVERY_DATE, COL_SHIP_METHOD,
    COL_WAREHOUSE, COL_PRODUCT, COL_SUPPLIER, COL_LOGISTICS,
)


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
    - Delay_Label         : binary target (1 = Delayed, 0 = otherwise)
    - Cost_Per_Unit       : Total_Cost / Quantity
    - Is_High_Value       : 1 if Unit_Price > median unit price
    - Delivery_Month      : month extracted from Delivery_Date
    - Delivery_DayOfWeek  : day of week from Delivery_Date
    """
    # --- Delay Label (target) ---
    if COL_DELIVERY_STATUS in df.columns and TARGET_COL not in df.columns:
        df[TARGET_COL] = (df[COL_DELIVERY_STATUS] == "Delayed").astype(int)
        print(f"[preprocessing] Delay_Label: {df[TARGET_COL].sum()} delayed out of {len(df)}")

    # --- Cost Per Unit ---
    if COL_TOTAL_COST in df.columns and COL_QUANTITY in df.columns:
        df["Cost_Per_Unit"] = (
            df[COL_TOTAL_COST] / df[COL_QUANTITY].replace(0, np.nan)
        ).fillna(0)

    # --- High Value Item flag ---
    if COL_UNIT_PRICE in df.columns:
        median_price = df[COL_UNIT_PRICE].median()
        df["Is_High_Value"] = (df[COL_UNIT_PRICE] > median_price).astype(int)

    # --- Date features ---
    if COL_DELIVERY_DATE in df.columns:
        dates = pd.to_datetime(df[COL_DELIVERY_DATE], errors="coerce")
        df["Delivery_Month"]     = dates.dt.month
        df["Delivery_DayOfWeek"] = dates.dt.dayofweek
        df["Delivery_Quarter"]   = dates.dt.quarter

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
