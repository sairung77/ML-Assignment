# ML-Assignment: Customer Churn Prediction & Retention System

## Business Problem

An e-commerce company is losing customers at a rate of **16.84% (948 out of 5,630 customers)**. Customer acquisition costs significantly exceed retention costs, making churn prevention a high-priority business initiative. The goal is to identify at-risk customers early and deploy targeted retention strategies that maximize ROI while minimizing wasted coupon spend.

## End-to-End Data Pipeline

![Data Pipeline](images/data_pipeline.png)

The project follows a 4-notebook sequential pipeline:

1. **Notebook 1 — EDA & RFM Clustering**: Exploratory analysis, feature engineering, and K-Means clustering to segment customers into behavioral groups.
2. **Notebook 2 — Stacking Model**: Train a 2-layer stacking ensemble that predicts churn probability for every customer. Outputs `predictions.csv`.
3. **Notebook 3 — Loyalty Program**: Use churn probability + cashback amount to create a Value-Risk quadrant and identify RESCUE customers. Outputs `rescue_priority_list.csv`.
4. **Notebook 4 — Coupon Targeting**: Apply ROI scoring + PR curve threshold to select optimal coupon recipients. Outputs `coupon_target_list.csv`.

## Key Results

| Metric | Value |
|---|---|
| ROC-AUC (Stacking Ensemble) | **0.9974** |
| Accuracy | **98.05%** |
| F1 Score | **0.9433** |
| RESCUE customers identified | **314 (5.6%)** |
| Final coupon recipients | **310 (5.5%)** |
| Coupon waste reduction | **94.5%** |
| Coupon precision | **100%** |

### Business Impact
- Without ML targeting: coupons distributed to all 5,630 customers → 83.16% wasted on non-churners
- With ML targeting: 310 high-risk, high-value customers selected → near-zero waste
- Estimated savings: eliminating ~5,320 unnecessary coupons per campaign cycle

## Execution Order

```
Notebook 1 → Notebook 2 → Notebook 3 (or 4, independent)
```

Notebooks 3 and 4 both depend on `predictions.csv` from Notebook 2 but are independent of each other.

## Tech Stack

| Library | Version | Role |
|---|---|---|
| numpy | 1.24+ | Numerical computation |
| pandas | 2.0+ | Data manipulation |
| scikit-learn | 1.3+ | ML pipeline, CV, metrics |
| xgboost | 2.0+ | Base model (gradient boosting) |
| lightgbm | 4.0+ | Base model (gradient boosting) |
| matplotlib | 3.7+ | Visualizations |
| seaborn | 0.12+ | Statistical plots |

## Dataset

- **Source**: E-Commerce Customer Churn dataset
- **Size**: 5,630 customers × 20 features
- **Target**: `Churn` (binary: 0 = Non-Churn, 1 = Churn)
- **Train/Test Split**: 80/20 → 4,504 training / 1,126 test
- **Location**: `data/` directory

## Project Structure

```
ML-Assignment/
├── data/                          # Raw dataset
├── outputs/                       # Intermediate outputs
├── 1_eda_churn.ipynb             # EDA & RFM Clustering
├── 2_prediction_stacking.ipynb   # Stacking Model
├── 3_prediction_loyalty.ipynb    # Loyalty Program
├── 4_prediction_coupon.ipynb     # Coupon Targeting
├── predictions.csv               # Churn probabilities (5,630 rows)
├── rescue_priority_list.csv      # RESCUE segment (314 rows)
├── coupon_target_list.csv        # Coupon recipients (310 rows)
└── notebook-lm-source/           # This documentation
```
