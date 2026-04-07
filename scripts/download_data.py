"""
download_data.py — Download supply chain dataset from Kaggle.
"""

import os
import sys
import zipfile
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

sys.path.insert(0, str(PROJECT_ROOT))
from src.config import KAGGLE_DATASET, DATA_RAW_DIR

KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "bhavesh971")
KAGGLE_KEY      = os.getenv("KAGGLE_KEY")

if not KAGGLE_KEY:
    print("[ERROR] KAGGLE_KEY not found in .env — cannot download.")
    sys.exit(1)

# Set kaggle credentials via environment so kaggle CLI picks them up
os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
os.environ["KAGGLE_KEY"]      = KAGGLE_KEY

DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

print(f"[download] Downloading dataset: {KAGGLE_DATASET}")

import kaggle
kaggle.api.authenticate()
kaggle.api.dataset_download_files(KAGGLE_DATASET, path=str(DATA_RAW_DIR), unzip=True)

# List what was downloaded
files = list(DATA_RAW_DIR.glob("*"))
print(f"[download] Files in data/raw/:")
for f in files:
    print(f"  {f.name}  ({f.stat().st_size / 1024:.1f} KB)")

print("[download] Done.")
