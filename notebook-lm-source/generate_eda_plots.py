"""
generate_eda_plots.py
สร้างภาพ EDA ทั้งหมดจากข้อมูลจริง — ใช้ font Sarabun รองรับภาษาไทย
รัน: python generate_eda_plots.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np
import pandas as pd
import os
from pathlib import Path

# ─── Thai Font Setup ───────────────────────────────────────────────────────────
def setup_thai_font():
    preferred = ['Sarabun', 'Thonburi', 'Ayuthaya', 'Krungthep', 'Silom']
    available = {f.name for f in fm.fontManager.ttflist}
    for font in preferred:
        if font in available:
            matplotlib.rcParams['font.family'] = font
            matplotlib.rcParams['axes.unicode_minus'] = False
            print(f'Font: {font}')
            return
    print('Warning: Thai font not found')

setup_thai_font()
sns.set_theme(style='whitegrid', font=matplotlib.rcParams['font.family'])

# ─── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / 'data' / 'Ecommerce Customer Churn.csv'
OUT_DIR = Path(__file__).parent / 'images' / 'eda'
OUT_DIR.mkdir(parents=True, exist_ok=True)

def save(name):
    path = OUT_DIR / name
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close('all')
    print(f'  saved: {name}')

# ─── Load & Clean Data ─────────────────────────────────────────────────────────
print('\nโหลดข้อมูล...')
df = pd.read_csv(DATA_PATH)
print(f'  Shape: {df.shape}')

# Median imputation
na_cols = [c for c in df.columns if df[c].isnull().any() and c != 'CustomerID']
df_clean = df.copy()
for col in na_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

print(f'  Cleaned shape: {df_clean.shape}')


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 01 — Churn Class Distribution
# ══════════════════════════════════════════════════════════════════════════════
print('\n[01] Churn Distribution...')
churn_counts = df['Churn'].value_counts()
churn_rate   = df['Churn'].mean()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

bars = axes[0].bar(['Non-Churn (0)', 'Churn (1)'], churn_counts.values,
                    color=['steelblue', 'salmon'], edgecolor='black')
for bar in bars:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                 f'{int(bar.get_height()):,}', ha='center', fontsize=12, fontweight='bold')
axes[0].set_title('จำนวนลูกค้าแต่ละคลาส', fontsize=13, fontweight='bold')
axes[0].set_ylabel('จำนวน (Count)')

axes[1].pie(churn_counts.values,
            labels=['ไม่ Churn', 'Churn'],
            colors=['steelblue', 'salmon'],
            autopct='%1.1f%%', startangle=90,
            wedgeprops=dict(edgecolor='black'),
            textprops={'fontsize': 12})
axes[1].set_title('สัดส่วนคลาส', fontsize=13, fontweight='bold')

ratio = churn_counts[0] / churn_counts[1]
axes[2].barh(['อัตราส่วน Imbalance\n(Non-Churn : Churn)'], [ratio],
             color='#FF9800', edgecolor='black', height=0.4)
axes[2].text(ratio + 0.1, 0, f'{ratio:.2f} : 1', va='center', fontsize=14, fontweight='bold')
axes[2].set_xlim(0, ratio + 1.5)
axes[2].set_title('Class Imbalance Ratio', fontsize=13, fontweight='bold')

plt.suptitle(f'Churn Rate = {churn_rate:.2%}  |  Dataset ไม่สมดุล (Imbalanced) — ต้องใช้ class_weight หรือ Stratified CV',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
save('eda_01_churn_distribution.png')


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 02 — Missing Values Heatmap & Bar
# ══════════════════════════════════════════════════════════════════════════════
print('[02] Missing Values...')
na_summary = pd.DataFrame({
    'NA Count': df[na_cols].isnull().sum(),
    'NA %': df[na_cols].isnull().mean() * 100
}).sort_values('NA %', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sample = df.sample(500, random_state=42)
sns.heatmap(sample[na_cols + ['Churn']].isnull(), cbar=False, yticklabels=False,
            cmap='YlOrRd', ax=axes[0])
axes[0].set_title('รูปแบบค่าที่หายไป (500 rows ตัวอย่าง)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('คอลัมน์')

na_summary['NA %'].sort_values().plot(kind='barh', color='steelblue', edgecolor='black', ax=axes[1])
axes[1].set_title('ร้อยละค่าที่หายไปแต่ละคอลัมน์', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Missing %')
for p in axes[1].patches:
    axes[1].annotate(f'{p.get_width():.2f}%',
                     (p.get_width() + 0.05, p.get_y() + p.get_height()/2),
                     va='center', fontsize=9)

plt.suptitle('การวิเคราะห์ค่าที่หายไป (Missing Values Analysis)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
save('eda_02_missing_values.png')


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 03 — Before / After Imputation Distribution
# ══════════════════════════════════════════════════════════════════════════════
print('[03] Before/After Imputation...')
fig, axes = plt.subplots(len(na_cols), 2, figsize=(12, len(na_cols) * 2.8))

for i, col in enumerate(na_cols):
    axes[i, 0].hist(df[col].dropna(), bins=30, color='salmon', edgecolor='black')
    axes[i, 0].set_title(f'{col} — ก่อนเติมค่า (dropna)', fontsize=10)
    axes[i, 0].set_ylabel('Count')

    axes[i, 1].hist(df_clean[col], bins=30, color='steelblue', edgecolor='black')
    axes[i, 1].set_title(f'{col} — หลังเติมค่า (Median fill)', fontsize=10)
    axes[i, 1].set_ylabel('Count')

plt.suptitle('Distribution ก่อนและหลังการเติมค่าที่หายไปด้วย Median', fontsize=13, y=1.01)
plt.tight_layout()
save('eda_03_before_after_imputation.png')


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 04 — Correlation Heatmap & Churn Correlation Bar
# ══════════════════════════════════════════════════════════════════════════════
print('[04] Correlation...')
num_for_corr = ['Tenure','CityTier','WarehouseToHome','HourSpendOnApp',
                'NumberOfDeviceRegistered','SatisfactionScore','NumberOfAddress',
                'Complain','OrderAmountHikeFromlastYear','CouponUsed',
                'OrderCount','DaySinceLastOrder','CashbackAmount','Churn']

corr = df_clean[num_for_corr].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, cbar_kws={'shrink': 0.8}, ax=axes[0],
            linewidths=0.5, linecolor='white')
axes[0].set_title('Correlation Matrix (Feature ทั้งหมด)', fontsize=13, fontweight='bold')

churn_corr = corr['Churn'].drop('Churn').sort_values()
colors_bar = ['#F44336' if v > 0 else '#2196F3' for v in churn_corr]
axes[1].barh(churn_corr.index, churn_corr.values, color=colors_bar, edgecolor='black', alpha=0.85)
axes[1].axvline(0, color='black', lw=0.8)
axes[1].set_title('ความสัมพันธ์ของ Feature กับ Churn', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Pearson Correlation')
for i, (feat, val) in enumerate(churn_corr.items()):
    axes[1].text(val + (0.005 if val >= 0 else -0.005), i,
                 f'{val:.3f}', va='center',
                 ha='left' if val >= 0 else 'right', fontsize=9)

plt.suptitle('Correlation Analysis — Feature ไหนส่งผลต่อ Churn มากที่สุด?',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
save('eda_04_correlation.png')


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 05 — Box Plots by Churn
# ══════════════════════════════════════════════════════════════════════════════
print('[05] Box Plots...')
box_cols = ['Tenure','WarehouseToHome','HourSpendOnApp',
            'OrderAmountHikeFromlastYear','CouponUsed','OrderCount',
            'DaySinceLastOrder','CashbackAmount','NumberOfAddress']

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()

for i, col in enumerate(box_cols):
    data0 = df_clean[df_clean['Churn']==0][col].dropna()
    data1 = df_clean[df_clean['Churn']==1][col].dropna()
    bp = axes[i].boxplot([data0, data1], tick_labels=['Non-Churn','Churn'],
                          patch_artist=True, notch=True,
                          medianprops=dict(color='crimson', lw=2))
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('salmon')
    bp['boxes'][1].set_alpha(0.7)
    axes[i].set_title(col, fontsize=11, fontweight='bold')

plt.suptitle('Box Plots: Feature ตัวเลข แยกตาม Churn (พร้อม Outlier Detection)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
save('eda_05_boxplots.png')


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 06 — Density Histograms by Churn
# ══════════════════════════════════════════════════════════════════════════════
print('[06] Density Histograms...')
num_cols = ['Tenure','WarehouseToHome','HourSpendOnApp',
            'OrderAmountHikeFromlastYear','CouponUsed','OrderCount',
            'DaySinceLastOrder','CashbackAmount',
            'NumberOfDeviceRegistered','NumberOfAddress','SatisfactionScore']

n = len(num_cols)
rows = (n + 2) // 3
fig, axes = plt.subplots(rows, 3, figsize=(16, rows * 3.8))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    for churn_val, label, color in [(0, 'Non-Churn', 'steelblue'), (1, 'Churn', 'salmon')]:
        subset = df_clean[df_clean['Churn'] == churn_val][col]
        axes[i].hist(subset, bins=30, alpha=0.6, label=label,
                     color=color, edgecolor='none', density=True)
    axes[i].set_title(col, fontsize=11, fontweight='bold')
    axes[i].set_ylabel('Density')
    axes[i].legend(fontsize=8)

for j in range(n, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribution ของ Feature ตัวเลข: Churn vs Non-Churn (Density)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
save('eda_06_density_histograms.png')


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 07 — Categorical Features by Churn
# ══════════════════════════════════════════════════════════════════════════════
print('[07] Categorical Features...')
cat_cols = ['PreferredLoginDevice','PreferredPaymentMode',
            'Gender','PreferedOrderCat','MaritalStatus','CityTier']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    ct = (df_clean.groupby([col, 'Churn'])
          .size().unstack(fill_value=0)
          .rename(columns={0: 'Non-Churn', 1: 'Churn'}))
    ct.plot(kind='bar', ax=axes[i], color=['steelblue', 'salmon'],
            edgecolor='black', width=0.7)
    axes[i].set_title(col, fontsize=12, fontweight='bold')
    axes[i].set_ylabel('จำนวน')
    axes[i].tick_params(axis='x', rotation=30)
    axes[i].legend(title='Churn')

plt.suptitle('Feature หมวดหมู่: Churn vs Non-Churn', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
save('eda_07_categorical_features.png')


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 08 — Churn Rate by Key Factors (6 subplots)
# ══════════════════════════════════════════════════════════════════════════════
print('[08] Churn Rate Analysis...')
df_temp = df_clean.copy()
df_temp['Tenure_Grp'] = pd.cut(df_temp['Tenure'], bins=[0,3,6,12,24,200],
                                labels=['0-3 เดือน','3-6 เดือน','6-12 เดือน','12-24 เดือน','>24 เดือน'])

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# 1) Complain
complain_churn = df_temp.groupby('Complain')['Churn'].mean() * 100
bars = axes[0,0].bar(['ไม่ร้องเรียน (0)','ร้องเรียน (1)'], complain_churn.values,
                      color=['#4CAF50','#F44336'], edgecolor='black', alpha=0.88)
for bar in bars:
    axes[0,0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                   f'{bar.get_height():.1f}%', ha='center', fontsize=13, fontweight='bold')
axes[0,0].set_title('อัตรา Churn ตามการร้องเรียน', fontsize=13, fontweight='bold')
axes[0,0].set_ylabel('Churn Rate (%)')

# 2) Tenure Group
tenure_churn = df_temp.groupby('Tenure_Grp', observed=True)['Churn'].mean() * 100
bars = axes[0,1].bar(tenure_churn.index.astype(str), tenure_churn.values,
                      color='#E91E63', edgecolor='black', alpha=0.88)
for bar in bars:
    axes[0,1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                   f'{bar.get_height():.1f}%', ha='center', fontsize=10, fontweight='bold')
axes[0,1].set_title('อัตรา Churn ตามช่วง Tenure', fontsize=13, fontweight='bold')
axes[0,1].set_ylabel('Churn Rate (%)')
axes[0,1].tick_params(axis='x', rotation=15)

# 3) City Tier
city_churn = df_temp.groupby('CityTier')['Churn'].mean() * 100
bars = axes[0,2].bar(['Tier 1','Tier 2','Tier 3'], city_churn.values,
                      color=['#2196F3','#FF9800','#9C27B0'], edgecolor='black', alpha=0.88)
for bar in bars:
    axes[0,2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                   f'{bar.get_height():.1f}%', ha='center', fontsize=10, fontweight='bold')
axes[0,2].set_title('อัตรา Churn ตาม City Tier', fontsize=13, fontweight='bold')
axes[0,2].set_ylabel('Churn Rate (%)')

# 4) Satisfaction Score
sat_churn = df_temp.groupby('SatisfactionScore')['Churn'].mean() * 100
bars = axes[1,0].bar([str(x) for x in sat_churn.index], sat_churn.values,
                      color='#FF5722', edgecolor='black', alpha=0.88)
for bar in bars:
    axes[1,0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                   f'{bar.get_height():.1f}%', ha='center', fontsize=10, fontweight='bold')
axes[1,0].set_title('อัตรา Churn ตามคะแนนความพึงพอใจ', fontsize=13, fontweight='bold')
axes[1,0].set_ylabel('Churn Rate (%)')
axes[1,0].set_xlabel('SatisfactionScore (1=ต่ำสุด, 5=สูงสุด)')

# 5) Tenure × Complain Heatmap
df_temp['Complain_Label'] = df_temp['Complain'].map({0:'ไม่ร้องเรียน', 1:'ร้องเรียน'})
pivot = df_temp.groupby(['Tenure_Grp','Complain_Label'], observed=True)['Churn'].mean().unstack() * 100
sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1,1],
            cbar_kws={'label':'Churn Rate (%)'}, linewidths=1, linecolor='white')
axes[1,1].set_title('Churn Rate: Tenure x Complaint', fontsize=13, fontweight='bold')
axes[1,1].set_ylabel('กลุ่ม Tenure')

# 6) Marital Status
marital_churn = df_temp.groupby('MaritalStatus')['Churn'].mean() * 100
bars = axes[1,2].bar(marital_churn.index, marital_churn.values,
                      color=['#3F51B5','#009688','#795548'], edgecolor='black', alpha=0.88)
for bar in bars:
    axes[1,2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                   f'{bar.get_height():.1f}%', ha='center', fontsize=10, fontweight='bold')
axes[1,2].set_title('อัตรา Churn ตามสถานภาพสมรส', fontsize=13, fontweight='bold')
axes[1,2].set_ylabel('Churn Rate (%)')

plt.suptitle('การวิเคราะห์อัตรา Churn ตามปัจจัยสำคัญ', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
save('eda_08_churn_rate_by_factors.png')


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 09 — Multivariate Interaction Heatmaps
# ══════════════════════════════════════════════════════════════════════════════
print('[09] Interaction Heatmaps...')
df_temp2 = df_clean.copy()
df_temp2['Tenure_Grp']    = pd.cut(df_temp2['Tenure'], bins=[0,3,6,12,24,200],
                                    labels=['0-3m','3-6m','6-12m','12-24m','>24m'])
df_temp2['Cashback_Grp']  = pd.qcut(df_temp2['CashbackAmount'], q=4,
                                     labels=['Q1\n(ต่ำ)','Q2','Q3','Q4\n(สูง)'],
                                     duplicates='drop')
df_temp2['DaysSince_Grp'] = pd.cut(df_temp2['DaySinceLastOrder'], bins=[-1,2,5,10,100],
                                    labels=['0-2 วัน','3-5 วัน','6-10 วัน','>10 วัน'])
df_temp2['Complain_Label'] = df_temp2['Complain'].map({0:'ไม่ร้องเรียน', 1:'ร้องเรียน'})

fig, axes = plt.subplots(1, 3, figsize=(22, 6))

# 1) Tenure × SatisfactionScore
piv1 = df_temp2.groupby(['Tenure_Grp','SatisfactionScore'], observed=True)['Churn'].mean().unstack() * 100
sns.heatmap(piv1, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[0],
            linewidths=1, linecolor='white', cbar_kws={'label':'Churn %'})
axes[0].set_title('Churn%: Tenure x SatisfactionScore', fontsize=12, fontweight='bold')
axes[0].set_ylabel('กลุ่ม Tenure')

# 2) DaySinceLastOrder × Complain
piv2 = df_temp2.groupby(['DaysSince_Grp','Complain_Label'], observed=True)['Churn'].mean().unstack() * 100
sns.heatmap(piv2, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1],
            linewidths=1, linecolor='white', cbar_kws={'label':'Churn %'})
axes[1].set_title('Churn%: DaySinceLastOrder x Complaint', fontsize=12, fontweight='bold')
axes[1].set_ylabel('ช่วงเวลาตั้งแต่สั่งซื้อล่าสุด')

# 3) PreferedOrderCat × Complain
piv3 = df_temp2.groupby(['PreferedOrderCat','Complain_Label'])['Churn'].mean().unstack() * 100
sns.heatmap(piv3, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[2],
            linewidths=1, linecolor='white', cbar_kws={'label':'Churn %'})
axes[2].set_title('Churn%: หมวดสินค้า x Complaint', fontsize=12, fontweight='bold')
axes[2].set_ylabel('หมวดสินค้าที่ชอบ')

plt.suptitle('ผลกระทบเชิงปฏิสัมพันธ์ (Interaction Effects) ต่อ Churn Rate',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
save('eda_09_interaction_heatmaps.png')


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 10 — Scatter Plots: Feature Pairs
# ══════════════════════════════════════════════════════════════════════════════
print('[10] Scatter Plots...')
scatter_pairs = [
    ('Tenure', 'CashbackAmount'),
    ('HourSpendOnApp', 'OrderCount'),
    ('WarehouseToHome', 'DaySinceLastOrder'),
    ('OrderAmountHikeFromlastYear', 'CouponUsed'),
    ('Tenure', 'OrderCount'),
    ('CashbackAmount', 'OrderAmountHikeFromlastYear'),
]

colors_s = {0: 'steelblue', 1: 'salmon'}
labels_s  = {0: 'Non-Churn', 1: 'Churn'}

fig, axes = plt.subplots(2, 3, figsize=(17, 10))
axes = axes.flatten()

for i, (x_col, y_col) in enumerate(scatter_pairs):
    for cv in [0, 1]:
        subset = df_clean[df_clean['Churn'] == cv]
        axes[i].scatter(subset[x_col], subset[y_col],
                        alpha=0.3, s=14, color=colors_s[cv], label=labels_s[cv])
    axes[i].set_xlabel(x_col)
    axes[i].set_ylabel(y_col)
    axes[i].set_title(f'{x_col}  vs  {y_col}', fontsize=10, fontweight='bold')
    axes[i].legend(title='Churn', markerscale=2, fontsize=8)

plt.suptitle('Scatter Plots: คู่ Feature จำแนกตาม Churn', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
save('eda_10_scatter_pairs.png')


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 11 — Pairplot (KDE) — Key Features
# ══════════════════════════════════════════════════════════════════════════════
print('[11] Pairplot KDE...')
pairplot_cols = ['Tenure','WarehouseToHome','HourSpendOnApp',
                 'OrderCount','DaySinceLastOrder','CashbackAmount','Churn']

base = df_clean[pairplot_cols].dropna()
pairplot_data = pd.concat(
    [grp.sample(n=min(350, len(grp)), random_state=42)
     for _, grp in base.groupby('Churn')],
    ignore_index=True
)
pairplot_data['Churn'] = pairplot_data['Churn'].map({0:'Non-Churn', 1:'Churn'})

g = sns.pairplot(
    data=pairplot_data,
    vars=[c for c in pairplot_cols if c != 'Churn'],
    hue='Churn', corner=True, kind='kde', diag_kind='kde',
    plot_kws={'fill': True, 'levels': 5, 'thresh': 0.06, 'alpha': 0.4},
    diag_kws={'fill': True, 'common_norm': False, 'alpha': 0.35},
    palette={'Non-Churn': 'steelblue', 'Churn': 'salmon'}
)
g.fig.suptitle('Pairplot Density (KDE) ของ Feature หลักจำแนกตาม Churn', y=1.02, fontsize=13)
g.fig.savefig(OUT_DIR / 'eda_11_pairplot_kde.png', dpi=120, bbox_inches='tight')
plt.close('all')
print('  saved: eda_11_pairplot_kde.png')


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 12 — RFM Segmentation Dashboard
# ══════════════════════════════════════════════════════════════════════════════
print('[12] RFM Dashboard...')
df_rfm = df_clean[['CustomerID','DaySinceLastOrder','OrderCount',
                    'CashbackAmount','Tenure','Churn']].copy()

df_rfm['R'] = pd.qcut(df_rfm['DaySinceLastOrder'], q=5,
                       labels=[5,4,3,2,1], duplicates='drop').astype(int)
df_rfm['F'] = pd.qcut(df_rfm['OrderCount'].rank(method='first'), q=5,
                       labels=[1,2,3,4,5]).astype(int)
df_rfm['M'] = pd.qcut(df_rfm['CashbackAmount'].rank(method='first'), q=5,
                       labels=[1,2,3,4,5]).astype(int)
df_rfm['RFM_Score'] = df_rfm['R'] + df_rfm['F'] + df_rfm['M']

def rfm_segment(row):
    s, r = row['RFM_Score'], row['R']
    if s >= 13:               return 'Champions'
    elif s >= 10:             return 'Loyal Customers'
    elif r >= 4 and s >= 7:  return 'Potential Loyalists'
    elif r >= 4 and s < 7:   return 'New Customers'
    elif r <= 2 and s >= 10: return 'At Risk'
    elif r <= 2 and s >= 7:  return 'Need Attention'
    else:                     return 'Lost / Hibernating'

df_rfm['Segment'] = df_rfm.apply(rfm_segment, axis=1)

rfm_summary = df_rfm.groupby('Segment').agg(
    CustomerCount=('CustomerID','count'),
    Churn_Rate=('Churn','mean'),
    Avg_R=('R','mean'), Avg_F=('F','mean'), Avg_M=('M','mean'),
    Avg_Recency=('DaySinceLastOrder','mean'),
    Avg_Cashback=('CashbackAmount','mean'),
).sort_values('Churn_Rate', ascending=False)

seg_order = rfm_summary.index.tolist()
palette = sns.color_palette('RdYlGn_r', len(seg_order))
color_map = dict(zip(seg_order, palette))

fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# 1. Count
ax1 = fig.add_subplot(gs[0, 0])
counts = rfm_summary['CustomerCount']
bars = ax1.barh(seg_order, counts, color=[color_map[s] for s in seg_order], edgecolor='white')
ax1.set_title('จำนวนลูกค้าต่อกลุ่ม', fontsize=12, fontweight='bold')
ax1.set_xlabel('จำนวน')
for bar, val in zip(bars, counts):
    ax1.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
             f'{val:,}', va='center', fontsize=9)

# 2. Churn Rate
ax2 = fig.add_subplot(gs[0, 1])
churn_r = rfm_summary['Churn_Rate'] * 100
bars2 = ax2.barh(seg_order, churn_r, color=[color_map[s] for s in seg_order], edgecolor='white')
ax2.set_title('อัตรา Churn ต่อกลุ่ม (%)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Churn Rate (%)')
for bar, val in zip(bars2, churn_r):
    ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', fontsize=9)

# 3. RFM Score Distribution
ax3 = fig.add_subplot(gs[0, 2])
for seg in seg_order:
    subset = df_rfm[df_rfm['Segment'] == seg]['RFM_Score']
    ax3.hist(subset, bins=range(3, 16), alpha=0.55, label=seg, color=color_map[seg], edgecolor='none')
ax3.set_title('การกระจาย RFM Score ต่อกลุ่ม', fontsize=12, fontweight='bold')
ax3.set_xlabel('RFM Score')
ax3.set_ylabel('Count')
ax3.legend(fontsize=7)

# 4. R vs F Scatter
ax4 = fig.add_subplot(gs[1, 0])
for seg in seg_order:
    sub = df_rfm[df_rfm['Segment'] == seg]
    ax4.scatter(sub['F'], sub['R'], alpha=0.4, s=18, label=seg, color=color_map[seg])
ax4.set_title('Recency vs Frequency Score', fontsize=12, fontweight='bold')
ax4.set_xlabel('Frequency Score')
ax4.set_ylabel('Recency Score')
ax4.legend(fontsize=7)

# 5. F vs M Scatter
ax5 = fig.add_subplot(gs[1, 1])
for seg in seg_order:
    sub = df_rfm[df_rfm['Segment'] == seg]
    ax5.scatter(sub['F'], sub['M'], alpha=0.4, s=18, label=seg, color=color_map[seg])
ax5.set_title('Frequency vs Monetary Score', fontsize=12, fontweight='bold')
ax5.set_xlabel('Frequency Score')
ax5.set_ylabel('Monetary Score')
ax5.legend(fontsize=7)

# 6. Avg R/F/M per segment
ax6 = fig.add_subplot(gs[1, 2])
x = np.arange(len(seg_order))
w = 0.26
ax6.bar(x - w, [df_rfm[df_rfm['Segment']==s]['R'].mean() for s in seg_order],
        w, label='R Score', color='#e74c3c', alpha=0.85)
ax6.bar(x,     [df_rfm[df_rfm['Segment']==s]['F'].mean() for s in seg_order],
        w, label='F Score', color='#3498db', alpha=0.85)
ax6.bar(x + w, [df_rfm[df_rfm['Segment']==s]['M'].mean() for s in seg_order],
        w, label='M Score', color='#2ecc71', alpha=0.85)
ax6.set_xticks(x)
ax6.set_xticklabels(seg_order, rotation=30, ha='right', fontsize=8)
ax6.set_title('R / F / M Score เฉลี่ยต่อกลุ่ม', fontsize=12, fontweight='bold')
ax6.set_ylabel('Score (1-5)')
ax6.legend(fontsize=9)

plt.suptitle('RFM Customer Segmentation Dashboard', fontsize=16, fontweight='bold', y=1.01)
save('eda_12_rfm_dashboard.png')


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 13 — K-Means Clustering: Elbow Method
# ══════════════════════════════════════════════════════════════════════════════
print('[13] Elbow Method...')
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

cluster_cols = ['DaySinceLastOrder','OrderCount','CashbackAmount',
                'Tenure','CouponUsed','Complain','SatisfactionScore']

df_cluster = df_clean[cluster_cols + ['CustomerID','Churn']].dropna().copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster[cluster_cols])

inertia = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(K_range, inertia, marker='o', linestyle='--', color='steelblue', lw=2.2, markersize=8)
ax.axvline(4, color='red', linestyle=':', lw=1.8, label='เลือก k=4 (Elbow)')
ax.scatter([4], [inertia[2]], s=120, color='red', zorder=5)
ax.set_xlabel('จำนวน Cluster (k)', fontsize=12)
ax.set_ylabel('Inertia (Within-Cluster SS)', fontsize=12)
ax.set_title('Elbow Method: หา k ที่เหมาะสมสำหรับ K-Means', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.text(4.15, inertia[2], 'k=4\n(Elbow)', color='red', fontsize=10, va='center')
plt.tight_layout()
save('eda_13_elbow_method.png')


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 14 — K-Means Clustering: Fit k=4 + PCA + t-SNE
# ══════════════════════════════════════════════════════════════════════════════
print('[14] K-Means + PCA + t-SNE (อาจใช้เวลาสักครู่)...')
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

km4 = KMeans(n_clusters=4, random_state=42, n_init=10)
df_cluster['Cluster'] = km4.fit_predict(X_scaled)

cluster_labels = {0:'Loyalists', 1:'At-Risk', 2:'Champions', 3:'New/Low'}
churn_per_cluster = df_cluster.groupby('Cluster')['Churn'].mean()
sorted_clusters = churn_per_cluster.sort_values(ascending=False)
label_map = {}
predefined = ['At-Risk', 'Potential\nChurners', 'Stable', 'Loyalists']
for rank, cid in enumerate(sorted_clusters.index):
    label_map[cid] = predefined[rank]

df_cluster['Cluster_Label'] = df_cluster['Cluster'].map(label_map)

pca = PCA(n_components=2, random_state=42)
pca_res = pca.fit_transform(X_scaled)
df_cluster['PCA1'] = pca_res[:, 0]
df_cluster['PCA2'] = pca_res[:, 1]

# t-SNE บน sample เพื่อความเร็ว
sample_idx = df_cluster.sample(n=min(2000, len(df_cluster)), random_state=42).index
X_tsne = X_scaled[df_cluster.index.get_indexer(sample_idx)]
tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', n_iter=500)
tsne_res = tsne.fit_transform(X_tsne)
df_cluster.loc[sample_idx, 'tSNE1'] = tsne_res[:, 0]
df_cluster.loc[sample_idx, 'tSNE2'] = tsne_res[:, 1]

cluster_palette = {lbl: c for lbl, c in zip(
    df_cluster['Cluster_Label'].unique(),
    ['#F44336','#FF9800','#4CAF50','#2196F3'])}

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# PCA
for lbl, grp in df_cluster.groupby('Cluster_Label'):
    axes[0].scatter(grp['PCA1'], grp['PCA2'], alpha=0.55, s=16,
                    label=lbl, color=cluster_palette.get(lbl, 'gray'))
axes[0].set_title('การแบ่งกลุ่มลูกค้า (PCA)', fontsize=13, fontweight='bold')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
axes[0].legend(title='กลุ่ม', fontsize=9)

# t-SNE
df_tsne_plot = df_cluster.dropna(subset=['tSNE1','tSNE2'])
for lbl, grp in df_tsne_plot.groupby('Cluster_Label'):
    axes[1].scatter(grp['tSNE1'], grp['tSNE2'], alpha=0.55, s=16,
                    label=lbl, color=cluster_palette.get(lbl, 'gray'))
axes[1].set_title('การแบ่งกลุ่มลูกค้า (t-SNE)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')
axes[1].legend(title='กลุ่ม', fontsize=9)

plt.suptitle('K-Means Clustering (k=4): Visualization ด้วย PCA และ t-SNE',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
save('eda_14_kmeans_pca_tsne.png')


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 15 — Cluster Profile: Feature Means per Cluster
# ══════════════════════════════════════════════════════════════════════════════
print('[15] Cluster Profiles...')
profile = df_cluster.groupby('Cluster_Label')[cluster_cols + ['Churn']].mean()

fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flatten()
colors_cl = ['#F44336','#FF9800','#4CAF50','#2196F3']

for i, col in enumerate(cluster_cols + ['Churn']):
    vals = [profile.loc[lbl, col] if lbl in profile.index else 0
            for lbl in predefined]
    bars = axes[i].bar(predefined, vals,
                        color=colors_cl[:len(predefined)], edgecolor='white', alpha=0.88)
    axes[i].set_title(col, fontsize=11, fontweight='bold')
    axes[i].set_ylabel('ค่าเฉลี่ย')
    axes[i].tick_params(axis='x', rotation=15)
    for bar in bars:
        axes[i].text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                     f'{bar.get_height():.2f}', ha='center', fontsize=8)

plt.suptitle('โปรไฟล์ของแต่ละกลุ่ม K-Means (Feature Means per Cluster)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
save('eda_15_cluster_profiles.png')


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 16 — Product Category by Cluster (Stacked Bar)
# ══════════════════════════════════════════════════════════════════════════════
print('[16] Product Category by Cluster...')
df_cluster['PreferedOrderCat'] = df_clean.loc[df_cluster.index, 'PreferedOrderCat']
cat_dist = pd.crosstab(df_cluster['Cluster_Label'], df_cluster['PreferedOrderCat'],
                        normalize='index') * 100

fig, ax = plt.subplots(figsize=(13, 6))
cat_dist.plot(kind='bar', stacked=True, ax=ax, colormap='tab20', alpha=0.88)
ax.set_title('ความชอบหมวดสินค้าของแต่ละกลุ่มลูกค้า (%)',
             fontsize=13, fontweight='bold')
ax.set_ylabel('ร้อยละ (%)')
ax.set_xlabel('กลุ่มลูกค้า')
ax.legend(title='หมวดสินค้า', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax.tick_params(axis='x', rotation=0)
plt.tight_layout()
save('eda_16_category_by_cluster.png')


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 17 — Tenure vs CashbackAmount (Loyalists vs At-Risk)
# ══════════════════════════════════════════════════════════════════════════════
print('[17] Tenure vs Cashback by Cluster...')
highlight_clusters = [lbl for lbl in predefined if lbl in df_cluster['Cluster_Label'].values]
df_hi = df_cluster[df_cluster['Cluster_Label'].isin(highlight_clusters[:2])].copy()

fig, ax = plt.subplots(figsize=(10, 6))
for lbl, grp in df_hi.groupby('Cluster_Label'):
    color = cluster_palette.get(lbl, 'gray')
    ax.scatter(grp['Tenure'], grp['CashbackAmount'],
               alpha=0.5, s=60, color=color, label=lbl,
               edgecolors='white', linewidths=0.4)

ax.set_title(f'Tenure vs CashbackAmount: {highlight_clusters[0]} vs {highlight_clusters[1]}',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Tenure (เดือน)', fontsize=12)
ax.set_ylabel('CashbackAmount (บาท)', fontsize=12)
ax.legend(title='กลุ่ม', fontsize=10, bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
save('eda_17_tenure_cashback_clusters.png')


# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*55)
print(f'สร้างภาพทั้งหมดเสร็จแล้ว! บันทึกใน: {OUT_DIR}')
files = sorted(OUT_DIR.glob('eda_*.png'))
print(f'จำนวนรูป: {len(files)} ไฟล์')
for f in files:
    size_kb = f.stat().st_size // 1024
    print(f'  {f.name}  ({size_kb} KB)')
