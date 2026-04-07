# Supply Chain Analysis and Prediction

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-green)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

**End-to-end data science project analyzing supply chain performance, warehouse efficiency, delivery delay prediction, and inventory optimization.**

---

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Workflow](#workflow)
5. [EDA](#eda)
6. [Feature Engineering](#feature-engineering)
7. [Feature Selection](#feature-selection)
8. [Model Results](#model-results)
9. [Inventory Optimization](#inventory-optimization)
10. [Business Insights](#business-insights)
11. [Setup & Usage](#setup--usage)
12. [Tech Stack](#tech-stack)

---

## Problem Statement

Supply chains face constant pressure from delivery delays, poor warehouse utilization, and suboptimal inventory levels. This project addresses four key questions:

1. Which shipment modes and regions have the highest delay rates? *(Logistics Analysis)*
2. Which warehouses underperform on throughput? *(Warehouse Efficiency)*
3. Can we predict whether an order will be delayed before it ships? *(ML Classification)*
4. What are the optimal reorder quantities and safety stock levels? *(Inventory Optimization)*

---

## Dataset

**Source:** [Kaggle — discovertalent143/supply-chain-dataset](https://www.kaggle.com/datasets/discovertalent143/supply-chain-dataset)

| Feature | Type | Description |
|---|---|---|
| Days_for_shipment_(scheduled) | Numeric | Planned shipment duration (days) |
| Days_for_shipping_(real) | Numeric | Actual shipment duration (days) |
| Late_delivery_risk | Binary | 1 = delayed, 0 = on-time |
| Shipment_Mode | Categorical | Mode of transport |
| Warehouse_block | Categorical | Warehouse identifier |
| Order_Item_Quantity | Numeric | Units ordered |
| Order_Item_Product_Price | Numeric | Unit price |
| Order_Item_Profit_Ratio | Numeric | Profit margin ratio |
| Product_Name / Category | Categorical | Product identifier |

---

## Project Structure

```
Supply chain Analysis and Prediction/
├── data/
│   ├── raw/               # Original dataset (gitignored)
│   └── processed/         # Cleaned + engineered data (gitignored)
├── notebooks/             # Jupyter notebooks — one per phase
├── src/
│   ├── config.py          # All paths, constants, model params
│   ├── data_loader.py     # Load + validate raw data
│   ├── preprocessing.py   # Clean, encode, engineer features
│   ├── model.py           # Train, evaluate, save all models
│   └── visualize.py       # All plot functions
├── models/                # Saved .pkl model files (gitignored)
├── images/                # All generated plots (committed)
├── reports/               # Results CSVs + PDF report
├── scripts/
│   ├── download_data.py   # Kaggle API download
│   └── generate_pdf.py    # PDF report generator
├── main.py                # Full pipeline runner
├── requirements.txt
└── README.md
```

---

## Workflow

```
Download Data -> EDA -> Feature Engineering -> Feature Selection
     -> Model Training -> Evaluation -> Inventory Optimization
     -> PDF Report -> README
```

---

## EDA

![Missing Values](images/01_missing_values.png)
![Distributions](images/02_numeric_distributions.png)
![Correlation Heatmap](images/03_correlation_heatmap.png)
![Target Distribution](images/04_target_distribution.png)

---

## Logistics Performance Analysis

![Shipment Mode Performance](images/05_shipment_mode_performance.png)

---

## Warehouse Efficiency Analysis

![Warehouse Efficiency](images/06_warehouse_efficiency.png)

---

## Feature Engineering

| Feature | Description |
|---|---|
| Delay_Label | Binary target: 1=delayed, 0=on-time |
| Lead_Time_Ratio | Actual days / Scheduled days |
| Revenue_Per_Item | Order quantity x product price |
| Is_High_Margin | 1 if profit ratio > 20% |

---

## Feature Selection

*(LASSO + RFE — results added after Phase 5)*

---

## Model Results

*(Results table added after Phase 6)*

![Model Comparison](images/08_model_comparison.png)
![ROC Curves](images/10_roc_curves.png)
![Feature Importance](images/11_feature_importance_XGBoost.png)

---

## Inventory Optimization

![EOQ and Safety Stock](images/07_inventory_optimization.png)

---

## Business Insights

*(Added after full pipeline run)*

---

## Setup & Usage

```bash
# 1. Clone the repository
git clone https://github.com/bhavesh2418/Supply-Chain-Analysis-and-Prediction.git
cd Supply-Chain-Analysis-and-Prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your .env file (never committed)
# Contents:
# GITHUB_TOKEN=...
# GITHUB_USERNAME=bhavesh2418
# KAGGLE_USERNAME=bhavesh971
# KAGGLE_KEY=...

# 4. Download dataset
python scripts/download_data.py

# 5. Run full pipeline
python main.py

# 6. Generate PDF report
python scripts/generate_pdf.py
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| pandas / numpy | Data manipulation |
| scikit-learn | ML models, preprocessing |
| XGBoost | Gradient boosting classifier |
| matplotlib / seaborn | Visualization |
| Jupyter | Notebooks |
| kaggle | Dataset download |
| fpdf2 | PDF report generation |
| GitHub | Version control |
