"""
generate_pdf.py — Generate project process PDF report using fpdf2.
Run after all notebooks and main.py have completed.
"""

import sys
from pathlib import Path
from fpdf import FPDF

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import REPORTS_DIR, IMAGES_DIR

REPORT_PATH = REPORTS_DIR / "Supply_Chain_Analysis_Process_Report.pdf"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 10, "Supply Chain Analysis and Prediction - Process Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 13)
        self.set_fill_color(30, 80, 150)
        self.set_text_color(255, 255, 255)
        self.cell(0, 9, f"  {title}", new_x="LMARGIN", new_y="NEXT", fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def sub_title(self, title: str):
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def add_image_if_exists(self, img_name: str, caption: str = "", w: int = 170):
        path = IMAGES_DIR / img_name
        if path.exists():
            self.image(str(path), w=w)
            if caption:
                self.set_font("Helvetica", "I", 9)
                self.cell(0, 6, caption, new_x="LMARGIN", new_y="NEXT")
            self.ln(3)
        else:
            self.body_text(f"[Image not available: {img_name}]")

    def key_value_row(self, key: str, value: str):
        self.set_font("Helvetica", "B", 10)
        self.cell(60, 7, key)
        self.set_font("Helvetica", "", 10)
        self.cell(0, 7, value, new_x="LMARGIN", new_y="NEXT")


pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# ── Cover ──────────────────────────────────────────────────────────────────────
pdf.set_font("Helvetica", "B", 20)
pdf.ln(10)
pdf.cell(0, 14, "Supply Chain Analysis and Prediction", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 12)
pdf.cell(0, 8, "End-to-End Data Science Project Report", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(5)
pdf.set_font("Helvetica", "", 10)
pdf.cell(0, 6, "GitHub: bhavesh2418 | Dataset: Kaggle — discovertalent143/supply-chain-dataset", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(10)

# ── Section 1: Project Overview ───────────────────────────────────────────────
pdf.section_title("1. Project Overview")
pdf.body_text(
    "This project performs a comprehensive analysis of supply chain data covering four key "
    "business areas:\n"
    "  1. Logistics Performance Analysis — identifying delay patterns by shipment mode, region, category\n"
    "  2. Warehouse Efficiency Analysis — measuring throughput and fulfillment rates per warehouse\n"
    "  3. Supply Chain Delay Prediction — ML classification models to predict late deliveries\n"
    "  4. Inventory Optimization — EOQ and safety stock calculations per product\n\n"
    "The goal is to surface actionable insights that reduce delays, optimize stock levels, and "
    "improve overall supply chain performance."
)

# ── Section 2: Dataset Description ───────────────────────────────────────────
pdf.section_title("2. Dataset Description")
pdf.body_text("Source: Kaggle — discovertalent143/supply-chain-dataset")
pdf.body_text(
    "The dataset contains supply chain transaction records including order details, shipment "
    "information, warehouse data, product categories, and delivery outcomes. Key columns include "
    "order quantities, scheduled vs actual shipment days, late delivery risk flags, profit ratios, "
    "and shipment modes."
)

# ── Section 3: Workflow ───────────────────────────────────────────────────────
pdf.section_title("3. Project Workflow")
pdf.body_text(
    "Phase 0  -> Project Setup (folder structure, src modules, git init)\n"
    "Phase 1  -> Dataset Download (Kaggle API)\n"
    "Phase 2  -> Dependency Installation\n"
    "Phase 3  -> EDA (01_Data_Preparation.ipynb)\n"
    "Phase 4  -> Feature Engineering (02_Feature_Engineering.ipynb)\n"
    "Phase 5  -> Feature Selection — LASSO + RFE (03_Feature_Selection.ipynb)\n"
    "Phase 6a -> Delay Prediction Models (04_Model_Training.ipynb)\n"
    "Phase 6b -> Model Evaluation (05_Model_Evaluation.ipynb)\n"
    "Phase 7  -> Full Pipeline Run (main.py)\n"
    "Phase 8  -> PDF Report Generation\n"
    "Phase 9  -> README Update"
)

# ── Section 4: EDA Findings ───────────────────────────────────────────────────
pdf.section_title("4. EDA — Key Findings")
pdf.add_image_if_exists("01_missing_values.png", "Fig 1: Missing Values per Column")
pdf.add_image_if_exists("02_numeric_distributions.png", "Fig 2: Numeric Feature Distributions")
pdf.add_image_if_exists("03_correlation_heatmap.png", "Fig 3: Feature Correlation Heatmap")
pdf.add_image_if_exists("04_target_distribution.png", "Fig 4: Target Class Distribution (Delayed vs On-Time)")

# ── Section 5: Logistics Analysis ────────────────────────────────────────────
pdf.section_title("5. Logistics Performance Analysis")
pdf.add_image_if_exists("05_shipment_mode_performance.png", "Fig 5: Delay Rate and Volume by Shipment Mode")

# ── Section 6: Warehouse Efficiency ──────────────────────────────────────────
pdf.section_title("6. Warehouse Efficiency Analysis")
pdf.add_image_if_exists("06_warehouse_efficiency.png", "Fig 6: Warehouse Efficiency Metrics")

# ── Section 7: Inventory Optimization ────────────────────────────────────────
pdf.section_title("7. Inventory Optimization")
pdf.add_image_if_exists("07_inventory_optimization.png", "Fig 7: EOQ and Safety Stock by Product")

# ── Section 8: Model Results ──────────────────────────────────────────────────
pdf.section_title("8. Delay Prediction — Model Results")
pdf.add_image_if_exists("08_model_comparison.png", "Fig 8: Model Comparison — Accuracy & F1 Score")
pdf.add_image_if_exists("10_roc_curves.png", "Fig 9: ROC Curves — All Models")
pdf.add_image_if_exists("11_feature_importance_XGBoost.png", "Fig 10: Feature Importance — XGBoost")

# ── Section 9: Feature Importance ────────────────────────────────────────────
pdf.section_title("9. Feature Engineering Summary")
pdf.body_text(
    "Engineered features created:\n"
    "  - Delay_Label          : Binary target (1=delayed, 0=on-time) from Late_delivery_risk\n"
    "  - Lead_Time_Ratio      : Actual days / Scheduled days — measures shipment efficiency\n"
    "  - Revenue_Per_Item     : Order quantity x product price\n"
    "  - Is_High_Margin       : Binary flag for items with profit ratio > 20%\n"
)

# ── Section 10: Business Recommendations ─────────────────────────────────────
pdf.section_title("10. Business Recommendations")
pdf.body_text(
    "1. Prioritize on-time performance for shipment modes with the highest delay rates.\n"
    "2. Investigate warehouses with below-average throughput — process bottlenecks likely.\n"
    "3. Deploy the XGBoost delay prediction model in the order management system to flag "
       "high-risk shipments proactively.\n"
    "4. Implement EOQ-based reorder policies for top-volume products to reduce stockouts "
       "and overstock costs.\n"
    "5. Focus margin improvement efforts on high-volume, low-margin product categories.\n"
)

# ── Section 11: File Index ────────────────────────────────────────────────────
pdf.section_title("11. File Index")
file_index = [
    ("src/config.py",           "Central configuration: paths, constants, model params"),
    ("src/data_loader.py",      "Load and validate raw CSV data"),
    ("src/preprocessing.py",    "Cleaning, imputation, feature engineering, encoding"),
    ("src/model.py",            "Train, evaluate, save, load all ML models"),
    ("src/visualize.py",        "All plot functions — save to images/"),
    ("scripts/download_data.py","Kaggle API data download"),
    ("scripts/generate_pdf.py", "This report generator"),
    ("main.py",                 "Full end-to-end pipeline runner"),
    ("notebooks/01_*.ipynb",    "EDA — data overview, distributions, correlations"),
    ("notebooks/02_*.ipynb",    "Feature engineering"),
    ("notebooks/03_*.ipynb",    "Feature selection — LASSO + RFE"),
    ("notebooks/04_*.ipynb",    "Model training — 5 classifiers"),
    ("notebooks/05_*.ipynb",    "Model evaluation — confusion matrix, ROC"),
    ("images/",                 "All saved plots — committed to GitHub"),
    ("reports/",                "Results CSVs and this PDF report"),
]
for fname, desc in file_index:
    pdf.key_value_row(fname, desc)

# ── Save ──────────────────────────────────────────────────────────────────────
pdf.output(str(REPORT_PATH))
print(f"[generate_pdf] Report saved -> {REPORT_PATH}")
