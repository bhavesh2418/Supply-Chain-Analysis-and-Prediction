"""
config.py — Central configuration: all paths, constants, and model parameters.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load credentials from .env in project root
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# ── Credentials ──────────────────────────────────────────────────────────────
GITHUB_TOKEN    = os.getenv("GITHUB_TOKEN")
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME", "bhavesh2418")
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "bhavesh971")
KAGGLE_KEY      = os.getenv("KAGGLE_KEY")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parents[1]
DATA_RAW_DIR   = BASE_DIR / "data" / "raw"
DATA_PROC_DIR  = BASE_DIR / "data" / "processed"
MODELS_DIR     = BASE_DIR / "models"
IMAGES_DIR     = BASE_DIR / "images"
REPORTS_DIR    = BASE_DIR / "reports"
NOTEBOOKS_DIR  = BASE_DIR / "notebooks"

# ── Dataset ───────────────────────────────────────────────────────────────────
KAGGLE_DATASET       = "discovertalent143/supply-chain-dataset"
RAW_CSV              = DATA_RAW_DIR / "Supply_Chain___Logistics_Dataset.csv"
RAW_OPS_CSV          = DATA_RAW_DIR / "Supply_Chain_Operations_Dataset.csv"
PROCESSED_CSV        = DATA_PROC_DIR / "supply_chain_processed.csv"

# ── Column names (actual dataset) ─────────────────────────────────────────────
COL_PRODUCT          = "Product"
COL_SUPPLIER         = "Supplier"
COL_WAREHOUSE        = "Warehouse_Location"
COL_QUANTITY         = "Quantity"
COL_UNIT_PRICE       = "Unit_Price"
COL_TOTAL_COST       = "Total_Cost"
COL_DELIVERY_DATE    = "Delivery_Date"
COL_LOGISTICS        = "Logistics_Partner"
COL_SHIP_METHOD      = "Shipping_Method"
COL_DELIVERY_STATUS  = "Delivery_Status"

# ── Target column ─────────────────────────────────────────────────────────────
# Binary: 1 = Delayed, 0 = On-time (Delivered / In Transit / Pending)
TARGET_COL = "Delay_Label"

# ── Model parameters ──────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.2
CV_FOLDS     = 5

XGBOOST_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
}

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "random_state": RANDOM_STATE,
}

# ── Inventory optimization ────────────────────────────────────────────────────
HOLDING_COST_RATE  = 0.25   # 25% of unit cost per year
ORDERING_COST      = 50.0   # fixed cost per order ($)
LEAD_TIME_DAYS     = 7      # default lead time
SAFETY_STOCK_Z     = 1.645  # 95% service level

# ── Plot style ────────────────────────────────────────────────────────────────
PLOT_STYLE  = "seaborn-v0_8-whitegrid"
FIGURE_DPI  = 150
PALETTE     = "husl"
