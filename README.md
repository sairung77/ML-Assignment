# Ecommerce Customer Churn — EDA, Segmentation & ML Prediction

โปรเจกต์นี้วิเคราะห์ข้อมูลลูกค้า E-commerce เพื่อหา pattern การ churn โดยใช้เทคนิค EDA, RFM Analysis, K-Means Clustering และ Stacking CV Ensemble

---

## โครงสร้างโปรเจกต์

```
ML-Assignment/
├── data/
│   └── Ecommerce Customer Churn.csv     # ชุดข้อมูลหลัก
├── 1_eda_churn.ipynb                    # EDA, RFM, K-Means Clustering
├── 2_prediction_stacking.ipynb          # Stacking CV Ensemble → predictions.csv
├── 3_prediction_loyalty.ipynb           # Loyalty Program — Value-Risk Matrix
├── 4_prediction_coupon.ipynb            # Coupon Targeting — ROI Score + PR Curve
├── requirements.txt                     # รายการ packages
└── README.md
```

---

## ความต้องการของระบบ

| รายการ | เวอร์ชัน |
|---|---|
| **Python** | **3.11 หรือ 3.12** (บังคับ — xgboost 2.x ไม่รองรับ Python 3.9/3.10) |
| OS | macOS / Linux / Windows |

---

## ขั้นตอนการติดตั้งและรัน

### 1. ตรวจสอบ Python version

```bash
python3 --version
# ต้องได้ Python 3.11.x หรือ 3.12.x
```

หาก Python เก่ากว่า 3.11 ให้ติดตั้งใหม่ก่อน:

**macOS (Homebrew):**
```bash
brew install python@3.11
# หรือ
brew install python@3.12
```

**Windows / Linux:** ดาวน์โหลดจาก [python.org](https://www.python.org/downloads/)

---

### 2. ติดตั้ง System Dependency: OpenMP (macOS เท่านั้น)

XGBoost และ LightGBM ต้องการ OpenMP runtime (`libomp`) รันคำสั่งนี้ **ครั้งเดียว** บนเครื่อง:

```bash
brew install libomp
ln -sf /opt/homebrew/opt/libomp/lib/libomp.dylib /opt/homebrew/lib/libomp.dylib
```

> ต้องมี [Homebrew](https://brew.sh) ก่อน — ถ้ายังไม่ได้ติดตั้งให้รัน `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"` ก่อน

---

### 3. สร้าง Virtual Environment ด้วย Python 3.11+

**macOS / Linux:**
```bash
python3.11 -m venv .venv
# หรือ python3.12 -m venv .venv
```

**Windows:**
```bash
py -3.11 -m venv .venv
```

---

### 4. Activate venv

**macOS / Linux:**
```bash
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
.venv\Scripts\activate.bat
```

> เมื่อ activate สำเร็จ จะเห็น `(.venv)` นำหน้า prompt

---

### 5. ติดตั้ง packages

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 6. เพิ่ม venv เป็น Jupyter Kernel

```bash
python -m ipykernel install --user --name=ml-churn --display-name "Python (ml-churn)"
```

---

### 7. เปิด Notebook

**VS Code:**
- เปิดไฟล์ `.ipynb` ที่ต้องการ
- คลิก kernel selector มุมขวาบน → เลือก **Python (ml-churn)**
- รัน cells ตามลำดับ

**JupyterLab:**
```bash
jupyter lab
```

---

## ลำดับการรัน Notebooks

```
1_eda_churn.ipynb              ← รันก่อน (EDA ทั้งหมด)
         ↓
2_prediction_stacking.ipynb   ← Train model → ได้ predictions.csv
         ↓              ↓
3_prediction_loyalty.ipynb    4_prediction_coupon.ipynb
   (รันได้อิสระ หลังจากมี predictions.csv)
```

---

## สรุปเนื้อหาใน Notebooks

### `1_eda_churn.ipynb` — EDA

| Section | หัวข้อ |
|---|---|
| 1–2 | Import libraries และโหลดข้อมูล |
| 3–4 | ตรวจสอบและแสดงภาพค่าที่หายไป (Missing Values) |
| 5–6 | ทำความสะอาดข้อมูลด้วย Median Imputation |
| 7 | เปรียบเทียบ Churn vs Non-Churn ใน Features ต่างๆ |
| 8 | Scatter Plots คู่ Features |
| 9 | Pairplot KDE แยกตาม Churn |
| 10.1 | RFM Analysis — แบ่งกลุ่มลูกค้าด้วย Recency / Frequency / Monetary |
| 10.2 | K-Means Clustering — แบ่ง 4 กลุ่มด้วย Elbow Method |
| 10.3 | Dimensionality Reduction ด้วย PCA และ t-SNE |
| 10.4–10.7 | Business Insights และกลยุทธ์ตาม Segment |

### `2_prediction_stacking.ipynb` — ML Prediction

| Section | หัวข้อ |
|---|---|
| 1–2 | Load data + EDA (correlation, feature distributions) |
| 3 | Preprocessing — Median Imputation + OneHotEncoding |
| 4 | Base Model Comparison (5-Fold CV) |
| 5 | Train Stacking CV Ensemble |
| 6 | Evaluation — Confusion Matrix, ROC Curve |
| 7 | Feature Importance (MDI + Permutation) |
| 8 | Export `predictions.csv` |

### `3_prediction_loyalty.ipynb` — Loyalty Program

| Section | หัวข้อ |
|---|---|
| 1–2 | Load predictions + EDA (cashback, tenure, value tier) |
| 3 | Value-Risk Quadrant (PROTECT / RESCUE / MAINTAIN / LET GO) |
| 4 | Priority Retention List — กลุ่ม RESCUE |
| 5 | Loyalty Program Design ตาม Quadrant |

### `4_prediction_coupon.ipynb` — Coupon Targeting

| Section | หัวข้อ |
|---|---|
| 1–2 | Load predictions + EDA (coupon usage, complain, product category) |
| 3 | Value-Risk Quadrant (กรอง LET GO ออก) |
| 4 | Coupon ROI Score + Budget Efficiency Curve |
| 5 | Precision-Recall Curve + Optimal Threshold |
| 6 | Final Coupon Target List |
| 7 | Export `coupon_target_list.csv` |

---

## Packages ที่ใช้

| Package | Version | วัตถุประสงค์ |
|---|---|---|
| `numpy` | 2.1.3 | คำนวณเชิงตัวเลข |
| `pandas` | 2.2.3 | จัดการ DataFrame |
| `matplotlib` | 3.9.2 | วาดกราฟพื้นฐาน |
| `seaborn` | 0.13.2 | วาดกราฟ statistical |
| `scikit-learn` | 1.5.2 | ML algorithms, preprocessing |
| `xgboost` | 2.1.3 | Gradient Boosting (XGBoost) |
| `lightgbm` | 4.5.0 | Gradient Boosting (LightGBM) |
| `jupyterlab` | 4.3.3 | รัน Notebook |
| `ipykernel` | 6.29.5 | เชื่อม venv กับ Jupyter |

---

## ปิดการใช้งาน venv

```bash
deactivate
```

---

## ข้อมูลชุดข้อมูล (Dataset)

- **ไฟล์:** `data/Ecommerce Customer Churn.csv`
- **จำนวนแถว:** ~5,630 ลูกค้า
- **Target column:** `Churn` (0 = ไม่ churn, 1 = churn)
- **Features หลัก:** Tenure, OrderCount, CashbackAmount, DaySinceLastOrder, SatisfactionScore, Complain, CouponUsed ฯลฯ
