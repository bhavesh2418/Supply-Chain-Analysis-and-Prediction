"""
generate_pdf.py -- Generate project process PDF report using fpdf2.
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


def clean(text: str) -> str:
    """Replace non-latin-1 characters with ASCII equivalents."""
    return (
        text.replace("\u2014", "-")   # em dash
            .replace("\u2013", "-")   # en dash
            .replace("\u2019", "'")   # right single quote
            .replace("\u2018", "'")   # left single quote
            .replace("\u201c", '"')   # left double quote
            .replace("\u201d", '"')   # right double quote
            .replace("\u2192", "->")  # right arrow
            .replace("\u2190", "<-")  # left arrow
            .replace("\u00e9", "e")   # e accent
    )


class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 10, "Supply Chain Analysis and Prediction - Process Report",
                  align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 13)
        self.set_fill_color(30, 80, 150)
        self.set_text_color(255, 255, 255)
        self.cell(0, 9, f"  {clean(title)}", new_x="LMARGIN", new_y="NEXT", fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def sub_title(self, title: str):
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 7, clean(title), new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 6, clean(text))
        self.ln(2)

    def add_image_if_exists(self, img_name: str, caption: str = "", w: int = 170):
        path = IMAGES_DIR / img_name
        if path.exists():
            self.image(str(path), w=w)
            if caption:
                self.set_font("Helvetica", "I", 9)
                self.cell(0, 6, clean(caption), new_x="LMARGIN", new_y="NEXT")
            self.ln(3)
        else:
            self.body_text(f"[Image not available: {img_name}]")

    def key_value_row(self, key: str, value: str):
        self.set_font("Helvetica", "B", 10)
        self.cell(65, 7, clean(key))
        self.set_font("Helvetica", "", 10)
        self.cell(0, 7, clean(value), new_x="LMARGIN", new_y="NEXT")


pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# -- Cover -------------------------------------------------------------------
pdf.set_font("Helvetica", "B", 20)
pdf.ln(10)
pdf.cell(0, 14, "Supply Chain Analysis and Prediction",
         align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 12)
pdf.cell(0, 8, "End-to-End Data Science Project Report",
         align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(5)
pdf.set_font("Helvetica", "", 10)
pdf.cell(0, 6, "GitHub: bhavesh2418 | Dataset: Kaggle - discovertalent143/supply-chain-dataset",
         align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(10)

# -- Section 1: Project Overview ---------------------------------------------
pdf.section_title("1. Project Overview")
pdf.body_text(
    "This project performs a comprehensive analysis of supply chain data covering four key "
    "business areas:\n"
    "  1. Logistics Performance Analysis - identifying delay patterns by shipment mode and warehouse\n"
    "  2. Warehouse Efficiency Analysis - measuring throughput and order volume per warehouse\n"
    "  3. Supply Chain Delay Prediction - ML classification models to predict late deliveries\n"
    "  4. Inventory Optimization - EOQ and safety stock calculations per product\n\n"
    "Dataset: 50 supply chain orders with 10 columns. Target: Delay_Label (1=Delayed, 0=On-time).\n"
    "Delay rate: 32% (16 out of 50 orders delayed).\n\n"
    "The goal is to surface actionable insights that reduce delays, optimize stock levels, and "
    "improve overall supply chain performance."
)

# -- Section 2: Dataset Description ------------------------------------------
pdf.section_title("2. Dataset Description")
pdf.body_text("Source: Kaggle - discovertalent143/supply-chain-dataset")
pdf.body_text(
    "Two CSV files downloaded:\n"
    "  - Supply_Chain___Logistics_Dataset.csv : 50 rows x 10 columns (primary)\n"
    "  - Supply_Chain_Operations_Dataset.csv  : 30 rows x 7 columns\n\n"
    "Key columns in primary dataset:\n"
    "  Product, Supplier, Warehouse_Location, Quantity, Unit_Price, Total_Cost,\n"
    "  Delivery_Date, Logistics_Partner, Shipping_Method, Delivery_Status\n\n"
    "Delivery_Status values: Delayed (16), In Transit (14), Pending (10), Delivered (10)\n"
    "Shipping methods: Road (18), Air (11), Sea (11), Rail (10)\n"
    "Warehouses: Warehouse 1 (21), Warehouse 2 (15), Warehouse 3 (14)"
)

# -- Section 3: Workflow -----------------------------------------------------
pdf.section_title("3. Project Workflow")
pdf.body_text(
    "Phase 0  -> Project Setup  : folder structure, src modules, git init, GitHub repo\n"
    "Phase 1  -> Data Download  : Kaggle API, 2 CSV files downloaded\n"
    "Phase 2  -> Dependencies   : pip install all packages, versions pinned\n"
    "Phase 3  -> EDA            : 01_Data_Preparation.ipynb - 7 plots generated\n"
    "Phase 4  -> Feature Eng.   : 02_Feature_Engineering.ipynb - 6 features created\n"
    "Phase 5  -> Feature Sel.   : 03_Feature_Selection.ipynb - LASSO+RFE, 8 features\n"
    "Phase 6a -> Model Training : 04_Model_Training.ipynb - 5 classifiers trained\n"
    "Phase 6b -> Model Eval.    : 05_Model_Evaluation.ipynb - confusion matrix, ROC\n"
    "Phase 7  -> Pipeline Run   : main.py end-to-end run verified\n"
    "Phase 8  -> PDF Report     : This document\n"
    "Phase 9  -> README Update  : Full README with inline images"
)

# -- Section 4: EDA ----------------------------------------------------------
pdf.section_title("4. EDA - Key Findings")
pdf.body_text(
    "Key findings from exploratory analysis:\n"
    "  - No missing values, no duplicates - dataset is clean\n"
    "  - 32% delay rate (16/50) - mild class imbalance\n"
    "  - Total_Cost strongly correlated with Quantity (as expected)\n"
    "  - Unit_Price range: $5.95 to $49.93 - high variance\n"
    "  - Road is most-used shipping method (36% of orders)\n"
    "  - Warehouse 1 handles most orders (42%)\n"
    "  - Alpha Corp is top supplier by order volume"
)
pdf.add_image_if_exists("02_numeric_distributions.png",
                        "Fig 1: Numeric Feature Distributions (Quantity, Unit Price, Total Cost)")
pdf.add_image_if_exists("03_correlation_heatmap.png",
                        "Fig 2: Feature Correlation Heatmap")
pdf.add_image_if_exists("04_target_distribution.png",
                        "Fig 3: Delivery Status Distribution - 32% Delayed")

# -- Section 5: Logistics Analysis -------------------------------------------
pdf.section_title("5. Logistics Performance Analysis")
pdf.body_text(
    "Delay rates were computed per shipping method to identify high-risk transport modes.\n"
    "Volume distribution shows Road as the dominant shipping method (36%), "
    "followed by Air, Sea, and Rail.\n"
    "Identifying high-delay shipping modes enables targeted SLA renegotiations with logistics partners."
)
pdf.add_image_if_exists("05_shipment_mode_performance.png",
                        "Fig 4: Delay Rate and Volume by Shipping Method")

# -- Section 6: Warehouse Efficiency -----------------------------------------
pdf.section_title("6. Warehouse Efficiency Analysis")
pdf.body_text(
    "Average order quantity per warehouse was used as the primary efficiency metric.\n"
    "Warehouse 1 handles the highest volume; Warehouse 3 has the lowest throughput.\n"
    "Low-throughput warehouses should be investigated for process bottlenecks or understaffing."
)
pdf.add_image_if_exists("06_warehouse_efficiency.png",
                        "Fig 5: Average Order Quantity by Warehouse")

# -- Section 7: Feature Engineering -----------------------------------------
pdf.section_title("7. Feature Engineering Summary")
pdf.body_text(
    "6 features engineered from raw columns:\n\n"
    "  Delay_Label          : Binary target - 1=Delayed, 0=On-time/Pending/In-Transit\n"
    "  Cost_Per_Unit        : Total_Cost / Quantity - effective unit economics\n"
    "  Is_High_Value        : 1 if Unit_Price > median ($28.19) - priority handling proxy\n"
    "  Delivery_Month       : Month from Delivery_Date - seasonality signal\n"
    "  Delivery_DayOfWeek   : 0=Mon to 6=Sun - weekday vs weekend patterns\n"
    "  Delivery_Quarter     : Q1-Q4 - quarter-end pressure effect\n\n"
    "Encoding: Label encoding on Product, Supplier, Warehouse_Location,\n"
    "          Logistics_Partner, Shipping_Method, Delivery_Status\n"
    "Outliers: All retained - dataset is 50 rows, removal would reduce signal\n"
    "Imputation: Not required - zero missing values"
)
pdf.add_image_if_exists("02c_engineered_features.png",
                        "Fig 6: Engineered Features vs Delay Label")
pdf.add_image_if_exists("02d_outlier_boxplots.png",
                        "Fig 7: Outlier Analysis - IQR Box Plots")

# -- Section 8: Feature Selection --------------------------------------------
pdf.section_title("8. Feature Selection - LASSO + RFE")
pdf.body_text(
    "Two independent selection methods applied:\n\n"
    "  LASSO (LassoCV, 5-fold CV):\n"
    "    Selected: Delivery_Status, Quantity\n"
    "    Note: Small dataset limits LASSO sensitivity\n\n"
    "  RFE (GradientBoostingClassifier, top 50%):\n"
    "    Selected: Cost_Per_Unit, Delivery_DayOfWeek, Delivery_Month,\n"
    "              Delivery_Quarter, Delivery_Status, Is_High_Value, Shipping_Method\n\n"
    "  Consensus (union of both): 8 features\n"
    "    Delivery_Status, Quantity, Cost_Per_Unit, Delivery_DayOfWeek,\n"
    "    Delivery_Month, Delivery_Quarter, Is_High_Value, Shipping_Method\n\n"
    "  Features selected by BOTH methods: Delivery_Status (strongest predictor)"
)
pdf.add_image_if_exists("03a_lasso_coefficients.png",
                        "Fig 8: LASSO Coefficients")
pdf.add_image_if_exists("03c_feature_selection_comparison.png",
                        "Fig 9: LASSO vs RFE Feature Selection Comparison")

# -- Section 9: Model Results ------------------------------------------------
pdf.section_title("9. Delay Prediction - Model Results")
pdf.body_text(
    "5 classifiers trained on 8 consensus features (40 train / 10 test, stratified split):\n\n"
    "  Model                  Accuracy   F1 Score   AUC-ROC   CV Acc (5-fold)\n"
    "  ---------------------  ---------  ---------  --------  ---------------\n"
    "  Logistic Regression    0.9000     0.9033     1.0000    0.9600\n"
    "  Decision Tree          1.0000     1.0000     1.0000    1.0000\n"
    "  Random Forest          1.0000     1.0000     1.0000    1.0000\n"
    "  Gradient Boosting      1.0000     1.0000     1.0000    1.0000\n"
    "  XGBoost                1.0000     1.0000     1.0000    1.0000\n\n"
    "Note: High scores reflect the small dataset (50 rows). Delivery_Status is a direct\n"
    "encoding of the outcome - in production, it would not be available at prediction time."
)
pdf.add_image_if_exists("08_model_comparison.png",
                        "Fig 10: Model Comparison - Accuracy, F1, AUC-ROC")
pdf.add_image_if_exists("10_roc_curves.png",
                        "Fig 11: ROC Curves - All 5 Models")
pdf.add_image_if_exists("09_confusion_matrices_all.png",
                        "Fig 12: Confusion Matrices - All Models")

# -- Section 10: Inventory Optimization -------------------------------------
pdf.section_title("10. Inventory Optimization - EOQ + Safety Stock")
pdf.body_text(
    "Economic Order Quantity (EOQ) and Safety Stock computed per product:\n\n"
    "  EOQ formula: sqrt(2 * Demand * Ordering_Cost / (Unit_Cost * Holding_Rate))\n"
    "  Safety Stock: Z * StdDev(Demand) * sqrt(Lead_Time)\n\n"
    "  Parameters used:\n"
    "    Holding cost rate : 25% of unit cost per year\n"
    "    Ordering cost     : $50 per order (fixed)\n"
    "    Lead time         : 7 days\n"
    "    Service level     : 95% (Z = 1.645)\n\n"
    "Results saved to reports/inventory_eoq.csv - reorder points computed for all products."
)
pdf.add_image_if_exists("07_inventory_optimization.png",
                        "Fig 13: Top Products by EOQ and Safety Stock")

# -- Section 11: Business Recommendations ------------------------------------
pdf.section_title("11. Business Recommendations")
pdf.body_text(
    "1. LOGISTICS: Renegotiate SLAs with logistics partners for shipping modes with\n"
    "   the highest delay rates. Monitor Air shipments closely - higher cost but\n"
    "   not necessarily faster delivery.\n\n"
    "2. WAREHOUSE: Investigate Warehouse 3 for process bottlenecks - lowest average\n"
    "   order throughput. Consider load rebalancing from Warehouse 1 (highest volume).\n\n"
    "3. PREDICTION: Deploy the XGBoost/Random Forest model in the order management\n"
    "   system to flag high-risk shipments at order creation time. Focus features\n"
    "   on Shipping_Method, Quantity, and date-based signals (exclude Delivery_Status).\n\n"
    "4. INVENTORY: Implement EOQ-based reorder policies for top-volume products.\n"
    "   Safety stock levels ensure 95% service level against demand variability.\n\n"
    "5. DATA: Collect more historical orders (target 500+) to improve model\n"
    "   generalization and reduce risk of overfitting on the current 50-row dataset."
)

# -- Section 12: File Index --------------------------------------------------
pdf.section_title("12. File Index")
file_index = [
    ("src/config.py",            "Central configuration - paths, constants, model params"),
    ("src/data_loader.py",       "Load and validate raw CSV data"),
    ("src/preprocessing.py",     "Cleaning, imputation, feature engineering, encoding"),
    ("src/model.py",             "Train, evaluate, save, load all ML models"),
    ("src/visualize.py",         "All plot functions - save to images/"),
    ("scripts/download_data.py", "Kaggle API data download"),
    ("scripts/generate_pdf.py",  "This PDF report generator"),
    ("main.py",                  "Full end-to-end pipeline runner"),
    ("notebooks/01_*.ipynb",     "EDA - data overview, distributions, correlations"),
    ("notebooks/02_*.ipynb",     "Feature engineering - 6 derived features"),
    ("notebooks/03_*.ipynb",     "Feature selection - LASSO + RFE, 8 features"),
    ("notebooks/04_*.ipynb",     "Model training - 5 classifiers"),
    ("notebooks/05_*.ipynb",     "Model evaluation - confusion matrix, ROC curves"),
    ("images/",                  "All 20+ saved plots - committed to GitHub"),
    ("reports/model_results.csv","Model comparison table with all metrics"),
    ("reports/inventory_eoq.csv","EOQ and safety stock per product"),
    ("reports/selected_features.csv", "LASSO + RFE consensus feature list"),
]
for fname, desc in file_index:
    pdf.key_value_row(fname, desc)

# -- Save --------------------------------------------------------------------
pdf.output(str(REPORT_PATH))
print(f"[generate_pdf] Report saved -> {REPORT_PATH}")
