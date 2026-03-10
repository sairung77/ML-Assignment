# การวิเคราะห์ข้อมูลและ RFM Clustering (Notebook 1)

## ภาพรวม

Notebook 1 ทำการวิเคราะห์เชิงสำรวจ (EDA) และใช้กรอบ RFM ร่วมกับ K-Means Clustering เพื่อแบ่งกลุ่มลูกค้าออกเป็น Segment ตามพฤติกรรม ผลลัพธ์ใช้เป็นพื้นฐานสำหรับกลยุทธ์การรักษาลูกค้าใน Notebook 3 และ 4

## RFM Dashboard

![RFM Dashboard](images/rfm_dashboard.png)

Dashboard แสดงการกระจายตัวของกลุ่มลูกค้าในทุกมิติ RFM, อัตรา Churn ต่อกลุ่ม และ Label ที่กำหนดให้แต่ละกลุ่ม

## กรอบ RFM ที่ใช้

| มิติ | คอลัมน์ | ความหมายทางธุรกิจ |
|---|---|---|
| **R** (Recency — ความใหม่) | `DaySinceLastOrder` | สั่งซื้อนานแค่ไหนแล้ว? น้อย = ล่าสุด = ดี |
| **F** (Frequency — ความถี่) | `OrderCount` | สั่งซื้อบ่อยแค่ไหน? มาก = ใช้งานบ่อย |
| **M** (Monetary — มูลค่า) | `CashbackAmount` | สร้างมูลค่าเท่าไหร่? สูง = ลูกค้ามีคุณค่า |

> หมายเหตุ: ใช้ `CashbackAmount` แทน Revenue โดยตรง เพราะสัมพันธ์กับมูลค่าการสั่งซื้อและมีอยู่ใน Dataset

## Feature ที่ใช้ใน Clustering

K-Means ใช้ **7 Feature** (ไม่ใช่แค่ RFM 3 ตัว):

```python
cluster_features = [
    'DaySinceLastOrder',  # R
    'OrderCount',          # F
    'CashbackAmount',      # M
    'Tenure',              # ระยะเวลาเป็นลูกค้า
    'CouponUsed',          # การใช้คูปอง
    'Complain',            # ประวัติการร้องเรียน
    'SatisfactionScore',   # คะแนนความพึงพอใจ
]
```

## K-Means Clustering

### การเลือก K ที่เหมาะสม

ใช้ **Elbow Method** โดยวัด Inertia (Within-Cluster Sum of Squares) สำหรับ k=2 ถึง k=10 พบ "ข้อศอก" ที่ **k=4** ซึ่งสมดุลระหว่างความละเอียดและความสามารถในการตีความ

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_cluster = scaler.fit_transform(df[cluster_features])

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['RFM_Cluster'] = kmeans.fit_predict(X_cluster)
```

### โปรไฟล์ของแต่ละกลุ่ม

| กลุ่ม | Label ธุรกิจ | Recency | Frequency | Monetary | อัตรา Churn |
|---|---|---|---|---|---|
| 0 | Champions (แชมเปี้ยน) | ต่ำ (สั่งล่าสุด) | สูง | สูง | ~8% |
| 1 | At-Risk (เสี่ยง) | สูง (ไม่ได้สั่งนาน) | ต่ำ | ปานกลาง | ~28% |
| 2 | Loyal Customers (ซื่อสัตย์) | ต่ำ | ปานกลาง | ปานกลาง | ~12% |
| 3 | New/Low-Value (ใหม่/ต่ำ) | ปานกลาง | ต่ำ | ต่ำ | ~20% |

## การทดสอบทางสถิติ

### Mann-Whitney U Test (สำหรับตัวแปรตัวเลข)
- **12 Feature ตัวเลข** มีความแตกต่างอย่างมีนัยสำคัญ (p < 0.05) ระหว่าง Churn กับ Non-Churn

### Chi-Square Test (สำหรับตัวแปรหมวดหมู่)
- **5 Feature หมวดหมู่** มีความสัมพันธ์อย่างมีนัยสำคัญกับ Churn (p < 0.05)

## ข้อค้นพบสำคัญจาก EDA

### 1. Tenure คือตัวทำนายที่แข็งแกร่งที่สุด

| ช่วง Tenure | อัตรา Churn |
|---|---|
| 0–3 เดือน | ~59.3% (สูงมาก) |
| 4–12 เดือน | ~22% |
| 12+ เดือน | ~5% |

**ข้อสรุป**: โปรแกรม Onboarding ใน 3 เดือนแรกมีผลโดยตรงต่ออัตรา Churn ตลอดชีวิตลูกค้า

### 2. Complaint คือสัญญาณ Binary ที่ชัดเจนที่สุด

| กลุ่ม | อัตรา Churn |
|---|---|
| ร้องเรียน (`Complain=1`) | **40.5%** |
| ไม่ร้องเรียน (`Complain=0`) | **13.4%** |

**ข้อสรุป**: ความเร็วในการแก้ปัญหาการร้องเรียนส่งผลต่ออัตรา Churn โดยตรง

### 3. ความสัมพันธ์ระหว่าง Value และ Complaint

| กลุ่ม | อัตรา Churn |
|---|---|
| High Value + ไม่ร้องเรียน | **5.8%** |
| High Value + ร้องเรียน | **100%** |
| Low Value + ไม่ร้องเรียน | **13.8%** |
| Low Value + ร้องเรียน | **54.5%** |

### 4. CashbackAmount — การกระจายแบบ 2 ยอด (Bimodal)

มีเส้นแบ่งชัดเจนที่ประมาณ **163 บาท** (Median) ซึ่งใช้เป็น Threshold ในการแบ่งกลุ่ม High/Low Value ใน Notebook 3

### 5. ความย้อนแย้งของ SatisfactionScore

ลูกค้าที่ให้คะแนนความพึงพอใจสูง บางส่วนก็ยัง Churn — แสดงว่าคู่แข่งเสนอดีลที่ดีกว่า แม้ลูกค้าจะพอใจกับบริการปัจจุบัน

## การแสดงผลด้วย PCA และ t-SNE

Notebook 1 ยังสร้าง Visualization ด้วย PCA (2D) และ t-SNE เพื่อยืนยันว่ากลุ่ม K-Means มีความแยกตัวที่ดีในพื้นที่ Feature

## Output ของ Notebook 1

- DataFrame ที่มีคอลัมน์ `RFM_Cluster` เพิ่มเข้ามา
- ภาพ RFM Dashboard ที่บันทึกใน `outputs/figures/rfm_dashboard.png`
- ข้อมูลที่ทำความสะอาดแล้ว (ค่าหายไปถูกเติมด้วย Median)
- Feature ที่สะอาดพร้อมส่งต่อ Notebook 2 สำหรับ Model Training
