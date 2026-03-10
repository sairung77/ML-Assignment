# คู่มือ Dataset: E-Commerce Customer Churn

## ภาพรวม Dataset

![ภาพรวม Dataset](images/dataset_overview.png)

| รายการ | ค่า |
|---|---|
| จำนวนลูกค้าทั้งหมด | 5,630 ราย |
| จำนวน Feature | 20 คอลัมน์ |
| ตัวแปรเป้าหมาย | `Churn` (0 = ไม่ Churn, 1 = Churn) |
| สัดส่วน Churn | 83.16% ไม่ Churn / 16.84% Churn |
| แหล่งข้อมูล | `data/Ecommerce Customer Churn.csv` |

## ตารางคำอธิบาย Feature ทั้ง 20 คอลัมน์

| # | คอลัมน์ | ประเภท | คำอธิบาย | ค่าหายไป |
|---|---|---|---|---|
| 1 | CustomerID | ID | รหัสลูกค้าเฉพาะ | 0% |
| 2 | Churn | Target (Binary) | 1 = Churn, 0 = ไม่ Churn | 0% |
| 3 | Tenure | ตัวเลข | จำนวนเดือนที่เป็นลูกค้า | **4.69%** |
| 4 | PreferredLoginDevice | หมวดหมู่ | อุปกรณ์ที่ใช้ Login บ่อยสุด (Mobile/Computer/Tablet) | 0% |
| 5 | CityTier | Ordinal | ระดับเมือง 1/2/3 | 0% |
| 6 | WarehouseToHome | ตัวเลข | ระยะทาง (กม.) จากคลังสินค้าถึงบ้าน | **4.46%** |
| 7 | PreferredPaymentMode | หมวดหมู่ | วิธีชำระเงินที่ชอบ (Credit/Debit/UPI/COD/E-wallet) | 0% |
| 8 | Gender | Binary | เพศ (Male/Female) | 0% |
| 9 | HourSpendOnApp | ตัวเลข | ชั่วโมงที่ใช้งาน App ต่อเดือน | **4.53%** |
| 10 | NumberOfDeviceRegistered | ตัวนับ | จำนวนอุปกรณ์ที่ลงทะเบียน (1–6) | 0% |
| 11 | PreferedOrderCat | หมวดหมู่ | หมวดสินค้าที่ซื้อบ่อยสุด | 0% |
| 12 | SatisfactionScore | Ordinal | คะแนนความพึงพอใจ 1–5 | 0% |
| 13 | MaritalStatus | หมวดหมู่ | สถานภาพสมรส (Single/Married/Divorced) | 0% |
| 14 | NumberOfAddress | ตัวนับ | จำนวนที่อยู่จัดส่งที่บันทึกไว้ | 0% |
| 15 | Complain | Binary | 1 = เคยร้องเรียนเดือนที่ผ่านมา | 0% |
| 16 | OrderAmountHikeFromlastYear | ตัวเลข | % การเพิ่มขึ้นของยอดสั่งซื้อเทียบปีที่แล้ว | **4.71%** |
| 17 | CouponUsed | ตัวนับ | จำนวนคูปองที่ใช้เดือนที่ผ่านมา | **4.55%** |
| 18 | OrderCount | ตัวนับ | จำนวนคำสั่งซื้อเดือนที่ผ่านมา | **4.58%** |
| 19 | DaySinceLastOrder | ตัวเลข | จำนวนวันนับจากสั่งซื้อครั้งล่าสุด | **5.45%** |
| 20 | CashbackAmount | ตัวเลข | จำนวนเงิน Cashback เฉลี่ย (บาท/เดือน) | 0% |

## การจัดการค่าที่หายไป

7 คอลัมน์มีค่าหายไป โดยเรียงจากมากไปน้อย:

```
DaySinceLastOrder              5.45%  →  เติมด้วย Median
OrderAmountHikeFromlastYear    4.71%  →  เติมด้วย Median
Tenure                         4.69%  →  เติมด้วย Median
OrderCount                     4.58%  →  เติมด้วย Median
CouponUsed                     4.55%  →  เติมด้วย Median
HourSpendOnApp                 4.53%  →  เติมด้วย Median
WarehouseToHome                4.46%  →  เติมด้วย Median
```

**เหตุผลที่เลือก Median**: ตัวแปรตัวเลขเหล่านี้มีการกระจายแบบเบ้ (Skewed) จึงใช้ Median แทน Mean เพื่อไม่ให้ค่าผิดปกติ (Outlier) บิดเบือนการเติมค่า

## ความไม่สมดุลของคลาส (Class Imbalance)

| คลาส | จำนวน | ร้อยละ |
|---|---|---|
| ไม่ Churn (0) | 4,682 | 83.16% |
| Churn (1) | 948 | 16.84% |
| อัตราส่วน | 4.93 : 1 | — |

วิธีรับมือกับ Class Imbalance:
- ใช้ `StratifiedKFold` (CV=5) รักษาสัดส่วน 83:17 ในทุก Fold
- ตั้งค่า `class_weight='balanced'` ใน Random Forest และ Extra Trees
- ประเมินด้วย F1 Score และ ROC-AUC แทนที่จะใช้ Accuracy อย่างเดียว

## การแบ่ง Train/Test

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# ผลลัพธ์: 4,504 training / 1,126 test
```

หลัง One-Hot Encoding มีทั้งหมด **29 Feature** (จาก 20 คอลัมน์ดิบ)

## ไฟล์ Output ที่ระบบสร้างออกมา

### `outputs/csv/predictions.csv` (5,630 แถว × 16 คอลัมน์)

| คอลัมน์สำคัญ | คำอธิบาย |
|---|---|
| `CustomerID` | รหัสลูกค้า |
| `Churn` | ค่าจริง (Actual Label) |
| `Churn_Prob` | ความน่าจะเป็น Churn จากโมเดล (0.0–1.0) |
| `Churn_Pred` | การทำนายแบบ Binary (threshold = 0.5) |
| `CashbackAmount` | ใช้สำหรับแบ่งกลุ่ม Value-Risk |

### `outputs/csv/rescue_priority_list.csv` (314 แถว)
ลูกค้ากลุ่ม RESCUE เรียงตาม `Churn_Prob` จากสูงไปต่ำ:
- Churn Risk ≥ 35% และ CashbackAmount ≥ 163 บาท
- Avg Churn Prob: **96.4%** | Avg Cashback: **202.3 บาท** | Avg Tenure: **5.58 เดือน**

### `outputs/csv/coupon_target_list.csv` (310 แถว)
ผู้รับคูปองสุดท้ายหลังกรองด้วย ROI Score + Threshold = 0.777:
- คอลัมน์: `Coupon_Priority`, `CustomerID`, `Churn_Prob`, `ROI_Score`, `CashbackAmount`, `Tenure`, `Complain`, `SatisfactionScore`, `Value_Risk`, `Actual_Churn`
- Precision: **100%** (ทุกคนเป็น Churn จริง)

## สิ่งที่ค้นพบจาก EDA

| ข้อค้นพบ | ตัวเลข |
|---|---|
| ลูกค้าที่ร้องเรียน vs ไม่ร้องเรียน | Churn 40.5% vs 13.4% |
| Tenure 0–3 เดือน | Churn rate ~59.3% (สูงสุด) |
| มูลค่าสหสัมพันธ์สูงสุด (Negative) | Tenure (−0.55), CashbackAmount (−0.48) |
| มูลค่าสหสัมพันธ์สูงสุด (Positive) | Complain (0.45), DaySinceLastOrder (0.42) |
| กลุ่มที่เสี่ยงที่สุด | Mobile Phone users + Complained = highest risk |
