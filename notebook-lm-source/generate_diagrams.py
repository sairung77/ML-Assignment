import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.font_manager as fm
import numpy as np
import os

# ─── Thai Font Setup ───────────────────────────────────────────────────────────
def setup_thai_font():
    """ตั้งค่า font ภาษาไทย — ลองตามลำดับจนกว่าจะเจอ"""
    preferred = ['Sarabun', 'Thonburi', 'Ayuthaya', 'Krungthep', 'Silom', 'Tahoma']
    available = {f.name for f in fm.fontManager.ttflist}
    for font in preferred:
        if font in available:
            matplotlib.rcParams['font.family'] = font
            matplotlib.rcParams['axes.unicode_minus'] = False
            print(f'✓ ใช้ font: {font}')
            return font
    print('⚠ ไม่พบ Thai font — ข้อความภาษาไทยอาจแสดงผลไม่ถูกต้อง')
    return None

THAI_FONT = setup_thai_font()
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Helper ───────────────────────────────────────────────────────────────────
def savefig(name):
    plt.savefig(os.path.join(OUTPUT_DIR, name), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ {name}')


# ─────────────────────────────────────────────────────────────────────────────
# 1. data_pipeline.png — ขั้นตอนการทำงานทั้งโปรเจกต์
# ─────────────────────────────────────────────────────────────────────────────
def make_data_pipeline():
    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 5.5)
    ax.axis('off')
    fig.patch.set_facecolor('#F8F9FA')

    stages = [
        ('ข้อมูลดิบ\n(E-Commerce\n5,630 ราย)', '#4C72B0', 1.1),
        ('Notebook 1\nวิเคราะห์ข้อมูล\n& RFM Clustering', '#DD8452', 3.8),
        ('Notebook 2\nโมเดล Stacking\nทำนาย Churn', '#55A868', 6.5),
        ('Notebook 3\nโปรแกรม\nสมาชิก', '#C44E52', 9.2),
        ('Notebook 4\nเลือกกลุ่ม\nส่งคูปอง', '#8172B2', 11.9),
        ('ผลลัพธ์\nทางธุรกิจ', '#937860', 14.2),
    ]

    for label, color, x in stages:
        box = FancyBboxPatch((x - 1.0, 1.8), 2.0, 2.0,
                             boxstyle='round,pad=0.12',
                             linewidth=1.5, edgecolor='white',
                             facecolor=color, alpha=0.92)
        ax.add_patch(box)
        ax.text(x, 2.8, label, ha='center', va='center',
                fontsize=9, color='white', fontweight='bold', linespacing=1.4)

    arrow_xs = [(2.1, 2.8), (4.8, 5.5), (7.5, 8.2), (10.2, 10.9), (12.9, 13.2)]
    for x1, x2 in arrow_xs:
        ax.annotate('', xy=(x2, 2.8), xytext=(x1, 2.8),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=2.2))

    outputs = [
        'กลุ่มลูกค้า K-Means\nRFM Segments',
        'predictions.csv\n(ความน่าจะเป็น Churn)',
        'rescue_priority\n_list.csv (314 ราย)',
        'coupon_target\n_list.csv (310 ราย)',
    ]
    ox = [3.8, 6.5, 9.2, 11.9]
    for label, x in zip(outputs, ox):
        ax.text(x, 1.1, label, ha='center', va='center', fontsize=8,
                color='#333',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor='#ccc', alpha=0.85))
        ax.annotate('', xy=(x, 1.8), xytext=(x, 1.45),
                    arrowprops=dict(arrowstyle='->', color='#aaa', lw=1.2))

    ax.set_title('ML-Assignment: กระบวนการทำงานตั้งแต่ต้นจนจบ (End-to-End Pipeline)',
                 fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    savefig('data_pipeline.png')


# ─────────────────────────────────────────────────────────────────────────────
# 2. stacking_architecture.png — โครงสร้างโมเดล Stacking 2 ชั้น
# ─────────────────────────────────────────────────────────────────────────────
def make_stacking_architecture():
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis('off')
    fig.patch.set_facecolor('#F8F9FA')

    # Input
    inp = FancyBboxPatch((0.2, 3.0), 2.0, 1.8, boxstyle='round,pad=0.12',
                         linewidth=1.5, edgecolor='white', facecolor='#4C72B0', alpha=0.88)
    ax.add_patch(inp)
    ax.text(1.2, 3.9, 'ข้อมูล Input\n29 Features\n(หลัง Encode)', ha='center', va='center',
            fontsize=9, color='white', fontweight='bold')

    # Layer 0 — base models (แก้ไข: ExtraTrees แทน Logistic Regression)
    base_models = [
        ('Random\nForest', '#2196F3', 'n=300, balanced'),
        ('Extra\nTrees', '#4CAF50', 'n=300, balanced'),
        ('XGBoost', '#FF9800', 'n=300, lr=0.05'),
        ('LightGBM', '#9C27B0', 'n=300, lr=0.05'),
        ('Gradient\nBoosting', '#F44336', 'n=200, lr=0.05'),
    ]
    layer0_y = [7.0, 5.6, 4.2, 2.8, 1.4]

    for (name, color, params), y in zip(base_models, layer0_y):
        box = FancyBboxPatch((3.2, y - 0.55), 2.6, 1.1,
                             boxstyle='round,pad=0.1',
                             linewidth=1, edgecolor='white', facecolor=color, alpha=0.88)
        ax.add_patch(box)
        ax.text(4.5, y + 0.15, name, ha='center', va='center',
                fontsize=9, color='white', fontweight='bold')
        ax.text(4.5, y - 0.25, params, ha='center', va='center',
                fontsize=7, color='white', alpha=0.9)
        ax.annotate('', xy=(3.2, y), xytext=(2.2, 3.9),
                    arrowprops=dict(arrowstyle='->', color='#888', lw=1.3,
                                   connectionstyle='arc3,rad=0'))

    ax.text(4.5, 7.65, 'ชั้นที่ 0 — Base Models (StratifiedKFold CV=5)',
            ha='center', fontsize=9.5, fontweight='bold', color='#1a237e')

    # CV Results labels
    cv_results = ['AUC 0.9675', 'AUC 0.9712', 'AUC 0.9652', 'AUC 0.9706', 'AUC 0.9468']
    for result, y in zip(cv_results, layer0_y):
        ax.text(6.1, y, result, ha='left', va='center', fontsize=7.5,
                color='#555', style='italic')

    # OOF arrows
    for y in layer0_y:
        ax.annotate('', xy=(8.0, 3.9), xytext=(5.8, y),
                    arrowprops=dict(arrowstyle='->', color='#888', lw=1.2,
                                   connectionstyle='arc3,rad=0'))

    # OOF box
    oof_box = FancyBboxPatch((8.0, 2.9), 2.0, 2.0, boxstyle='round,pad=0.12',
                              linewidth=1.5, edgecolor='white', facecolor='#607D8B', alpha=0.88)
    ax.add_patch(oof_box)
    ax.text(9.0, 3.9, 'OOF\nPredictions\n5 คอลัมน์\n(ไม่รั่วข้อมูล)', ha='center', va='center',
            fontsize=8.5, color='white', fontweight='bold')

    # Layer 1
    ax.annotate('', xy=(11.2, 3.9), xytext=(10.0, 3.9),
                arrowprops=dict(arrowstyle='->', color='#555', lw=2.2))

    meta_box = FancyBboxPatch((11.2, 2.95), 2.4, 1.9, boxstyle='round,pad=0.12',
                               linewidth=2, edgecolor='#FFD700', facecolor='#E91E63', alpha=0.92)
    ax.add_patch(meta_box)
    ax.text(12.4, 3.9, 'Meta Learner\nLogistic\nRegression\n(ชั้นที่ 1)', ha='center', va='center',
            fontsize=9, color='white', fontweight='bold')
    ax.text(12.4, 7.65, 'ชั้นที่ 1 — Meta Model',
            ha='center', fontsize=9.5, fontweight='bold', color='#880e4f')

    # Output
    ax.annotate('', xy=(14.0, 3.9), xytext=(13.6, 3.9),
                arrowprops=dict(arrowstyle='->', color='#555', lw=2.2))
    out_box = FancyBboxPatch((14.0, 3.15), 1.0, 1.5, boxstyle='round,pad=0.1',
                              linewidth=1.5, edgecolor='white', facecolor='#4CAF50', alpha=0.92)
    ax.add_patch(out_box)
    ax.text(14.5, 3.9, 'Churn\nProb', ha='center', va='center',
            fontsize=8.5, color='white', fontweight='bold')

    # Metrics bar
    ax.text(7.5, 0.5,
            'ROC-AUC: 0.9974   |   Accuracy: 98.05%   |   F1 Score: 0.9433   |   Precision: 92%   |   Recall: 96%',
            ha='center', va='center', fontsize=10, fontweight='bold', color='#1a237e',
            bbox=dict(boxstyle='round,pad=0.45', facecolor='#E3F2FD',
                      edgecolor='#1a237e', alpha=0.92))

    ax.set_title('สถาปัตยกรรมโมเดล Stacking Ensemble 2 ชั้น', fontsize=13, fontweight='bold', pad=10)
    plt.tight_layout()
    savefig('stacking_architecture.png')


# ─────────────────────────────────────────────────────────────────────────────
# 3. dataset_overview.png — ภาพรวม Dataset
# ─────────────────────────────────────────────────────────────────────────────
def make_dataset_overview():
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('#F8F9FA')
    fig.suptitle('ภาพรวม Dataset: E-Commerce Churn  (5,630 ลูกค้า × 20 Features)',
                 fontsize=14, fontweight='bold', y=0.97)

    # ── Churn Pie ──────────────────────────────────────────────────────────────
    ax1 = fig.add_axes([0.02, 0.52, 0.22, 0.4])
    sizes = [83.16, 16.84]
    colors = ['#4CAF50', '#F44336']
    wedges, texts, autotexts = ax1.pie(
        sizes,
        labels=['ไม่ Churn\n(4,682 ราย)', 'Churn\n(948 ราย)'],
        colors=colors, autopct='%1.1f%%',
        startangle=90, pctdistance=0.72,
        wedgeprops=dict(width=0.62))
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight('bold')
    ax1.set_title('สัดส่วน Churn', fontsize=11, fontweight='bold')

    # ── Missing Values ─────────────────────────────────────────────────────────
    ax2 = fig.add_axes([0.27, 0.52, 0.36, 0.4])
    # ข้อมูลจริงจาก notebook
    cols_missing = ['DaySinceLastOrder', 'OrderAmountHike\nFromlastYear',
                    'Tenure', 'OrderCount',
                    'CouponUsed', 'HourSpendOnApp', 'WarehouseToHome']
    miss_pct = [5.45, 4.71, 4.69, 4.58, 4.55, 4.53, 4.46]
    bars = ax2.barh(cols_missing[::-1], miss_pct[::-1],
                    color='#FF7043', edgecolor='white', height=0.6)
    ax2.set_xlim(0, 7.0)
    ax2.axvline(5.0, color='red', linestyle='--', alpha=0.45, linewidth=1.2)
    for bar, pct in zip(bars, miss_pct[::-1]):
        ax2.text(pct + 0.06, bar.get_y() + bar.get_height() / 2,
                 f'{pct}%', va='center', fontsize=8.5)
    ax2.set_xlabel('ร้อยละที่หายไป (%)', fontsize=9)
    ax2.set_title('ค่าที่หายไป (→ เติมด้วย Median)', fontsize=10, fontweight='bold')
    ax2.tick_params(labelsize=8)

    # ── Feature Types ──────────────────────────────────────────────────────────
    ax3 = fig.add_axes([0.66, 0.52, 0.14, 0.4])
    types = ['ตัวเลข\n(ต่อเนื่อง)', 'ตัวเลข\n(นับได้)', 'หมวดหมู่\n/ Binary']
    counts = [8, 9, 3]
    ax3.bar(types, counts,
            color=['#42A5F5', '#26C6DA', '#AB47BC'], edgecolor='white')
    for i, v in enumerate(counts):
        ax3.text(i, v + 0.08, str(v), ha='center', fontsize=11, fontweight='bold')
    ax3.set_ylim(0, 13)
    ax3.set_title('ประเภท Feature', fontsize=10, fontweight='bold')
    ax3.tick_params(labelsize=8)

    # ── Top 5 Feature Importance ───────────────────────────────────────────────
    ax4 = fig.add_axes([0.02, 0.05, 0.46, 0.40])
    features = ['Tenure', 'Complain', 'CashbackAmount',
                'DaySinceLastOrder', 'SatisfactionScore']
    importance = [0.312, 0.218, 0.156, 0.134, 0.089]
    colors4 = ['#1565C0', '#1976D2', '#1E88E5', '#2196F3', '#42A5F5']
    bars4 = ax4.barh(features[::-1], importance[::-1], color=colors4, edgecolor='white', height=0.6)
    for bar, val in zip(bars4, importance[::-1]):
        ax4.text(val + 0.004, bar.get_y() + bar.get_height() / 2,
                 f'{val:.3f}', va='center', fontsize=8.5)
    ax4.set_xlabel('ความสำคัญ MDI', fontsize=9)
    ax4.set_title('Top 5 ตัวแปรสำคัญ (MDI Importance)', fontsize=10, fontweight='bold')
    ax4.set_xlim(0, 0.40)
    ax4.tick_params(labelsize=9)

    # ── Stats Table ────────────────────────────────────────────────────────────
    ax5 = fig.add_axes([0.52, 0.05, 0.46, 0.40])
    ax5.axis('off')
    table_data = [
        ['ตัวชี้วัด', 'ค่า'],
        ['จำนวนลูกค้าทั้งหมด', '5,630 ราย'],
        ['ชุดฝึกสอน (80%)', '4,504 ราย'],
        ['ชุดทดสอบ (20%)', '1,126 ราย'],
        ['Cross-Validation', '5-Fold StratifiedKFold'],
        ['ROC-AUC (Stacking)', '0.9974'],
        ['Accuracy', '98.05%'],
        ['F1 Score', '0.9433'],
        ['กลุ่ม RESCUE', '314 ราย (5.6%)'],
        ['ผู้รับคูปอง', '310 ราย (5.5%)'],
        ['ประหยัดคูปอง', '94.5%'],
    ]
    table = ax5.table(cellText=table_data[1:], colLabels=table_data[0],
                      cellLoc='center', loc='center',
                      colWidths=[0.58, 0.42])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.32)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#1565C0')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#E3F2FD')
        cell.set_edgecolor('#BBDEFB')
    ax5.set_title('สถิติสำคัญของโปรเจกต์', fontsize=10, fontweight='bold', pad=8)

    savefig('dataset_overview.png')


# ─────────────────────────────────────────────────────────────────────────────
# 4. value_risk_quadrant_detail.png — ตาราง 4 กลุ่ม Value-Risk
# ─────────────────────────────────────────────────────────────────────────────
def make_value_risk_quadrant():
    fig, ax = plt.subplots(figsize=(13, 10))
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    quads = [
        # (x, y, bg, label_th, label_en, count, pct, strategy lines)
        (0.2, 5.2, '#E8F5E9', 'PROTECT', 'ลูกค้า VIP รักษาไว้', 2537, '45.1%',
         ['มูลค่าสูง / ความเสี่ยงต่ำ',
          '• รางวัล VIP & สิทธิพิเศษ',
          '• เพิ่มยอดขาย premium',
          '• คะแนนสะสม Loyalty']),
        (5.2, 5.2, '#FFEBEE', 'RESCUE', 'กู้คืนด่วน', 314, '5.6%',
         ['มูลค่าสูง / ความเสี่ยงสูง',
          '• ติดต่อส่วนตัวใน 48 ชม.',
          '• แก้ปัญหา Complaint',
          '• ส่งคูปองอิงค่า ROI']),
        (0.2, 0.2, '#E3F2FD', 'MAINTAIN', 'ดูแลพื้นฐาน', 2114, '37.6%',
         ['มูลค่าต่ำ / ความเสี่ยงต่ำ',
          '• โปรแกรม Loyalty มาตรฐาน',
          '• กระตุ้นให้ใช้จ่ายมากขึ้น',
          '• แคมเปญอัตโนมัติ']),
        (5.2, 0.2, '#FFF9C4', 'LET GO', 'ไม่ลงทุนเพิ่ม', 665, '11.8%',
         ['มูลค่าต่ำ / ความเสี่ยงสูง',
          '• ไม่ส่งคูปอง',
          '• ยอมรับการสูญเสีย',
          '• ประหยัดงบค่าใช้จ่าย']),
    ]
    quad_colors = {'PROTECT': '#2E7D32', 'RESCUE': '#C62828',
                   'MAINTAIN': '#1565C0', 'LET GO': '#F57F17'}

    for x, y, bg, label, label_th, count, pct, strategy in quads:
        box = FancyBboxPatch((x, y), 4.6, 4.6, boxstyle='round,pad=0.15',
                             linewidth=2.2, edgecolor=quad_colors[label], facecolor=bg, alpha=0.92)
        ax.add_patch(box)
        ax.text(x + 2.3, y + 4.15, label, ha='center', va='center',
                fontsize=16, fontweight='bold', color=quad_colors[label])
        ax.text(x + 2.3, y + 3.65, f'({label_th})', ha='center', va='center',
                fontsize=9, color=quad_colors[label], style='italic')
        ax.text(x + 2.3, y + 3.15, f'{count:,} ราย  ({pct})', ha='center', va='center',
                fontsize=10.5, color='#444', fontweight='bold')
        for i, line in enumerate(strategy):
            style = 'italic' if i == 0 else 'normal'
            ax.text(x + 2.3, y + 2.6 - i * 0.65, line,
                    ha='center', va='center', fontsize=9, color='#333', style=style)

    # Dividers
    ax.plot([5.0, 5.0], [0.1, 9.9], color='#777', lw=2, linestyle='--')
    ax.plot([0.1, 9.9], [5.0, 5.0], color='#777', lw=2, linestyle='--')

    # Axis labels
    ax.text(5.0, -0.35, '← CashbackAmount →', ha='center', fontsize=11,
            fontweight='bold', color='#333')
    ax.text(-0.35, 5.0, '← Churn\nProbability →', ha='center', va='center',
            fontsize=9.5, color='#333', rotation=90)

    # Threshold labels
    ax.text(2.5, 9.8, 'ความเสี่ยงต่ำ (< 35%)', ha='center', fontsize=8.5, color='#666')
    ax.text(7.5, 9.8, 'ความเสี่ยงสูง (≥ 35%)', ha='center', fontsize=8.5, color='#666')
    ax.text(0.08, 7.5, 'Cashback\nสูง\n(≥163)', ha='center', va='center',
            fontsize=8, color='#666', rotation=90)
    ax.text(0.08, 2.5, 'Cashback\nต่ำ\n(<163)', ha='center', va='center',
            fontsize=8, color='#666', rotation=90)

    # เส้น threshold annotations
    ax.text(5.0, 9.5, '← Median CashbackAmount = 163 บาท →',
            ha='center', fontsize=8.5, color='#555', style='italic')

    ax.set_title('การแบ่งกลุ่มลูกค้า Value-Risk Quadrant  (ทั้งหมด 5,630 ราย)',
                 fontsize=13, fontweight='bold', pad=14)
    plt.tight_layout()
    savefig('value_risk_quadrant_detail.png')


# ─────────────────────────────────────────────────────────────────────────────
# 5. coupon_flowchart.png — กระบวนการเลือกกลุ่มคูปอง
# ─────────────────────────────────────────────────────────────────────────────
def make_coupon_flowchart():
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 9)
    ax.axis('off')
    fig.patch.set_facecolor('#F8F9FA')

    def box(x, y, w, h, text, color, tcolor='white', fs=9):
        b = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                           boxstyle='round,pad=0.15', linewidth=1.5,
                           edgecolor='white', facecolor=color, alpha=0.92)
        ax.add_patch(b)
        ax.text(x, y, text, ha='center', va='center', fontsize=fs,
                color=tcolor, fontweight='bold')

    def diamond(x, y, w, h, text, color, fs=8.5):
        d = plt.Polygon([[x, y + h / 2], [x + w / 2, y],
                         [x, y - h / 2], [x - w / 2, y]],
                        facecolor=color, edgecolor='white', lw=1.5, alpha=0.92)
        ax.add_patch(d)
        ax.text(x, y, text, ha='center', va='center', fontsize=fs,
                color='white', fontweight='bold')

    def arrow(x1, y1, x2, y2, label='', lx=None, ly=None):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.8))
        if label:
            mx = lx if lx else (x1 + x2) / 2 + 0.1
            my = ly if ly else (y1 + y2) / 2
            ax.text(mx, my, label, fontsize=8, color='#444', style='italic')

    # Step 1
    box(1.5, 7.5, 2.2, 1.0, 'predictions.csv\n5,630 ราย', '#4C72B0')
    arrow(1.5, 7.0, 1.5, 6.1)

    # Step 2
    diamond(1.5, 5.5, 2.8, 1.2, 'แบ่งกลุ่ม\nValue-Risk\nQuadrant', '#7B1FA2')
    arrow(1.5, 4.9, 1.5, 4.0, 'RESCUE\n(314 ราย)')

    # Step 3
    box(1.5, 3.4, 2.2, 1.0, 'ตัดกลุ่ม\nLET GO (665 ราย)\nออก', '#C62828')
    arrow(1.5, 2.9, 1.5, 2.0)

    box(1.5, 1.5, 2.2, 0.9, 'กลุ่ม RESCUE\n314 ราย', '#E65100')

    # Step 4
    arrow(2.6, 1.5, 4.5, 5.5, 'คำนวณ\nROI Score')

    box(5.5, 5.5, 2.6, 1.1,
        'ROI Score\n= Churn_Prob\n× CashbackAmount', '#1565C0')
    arrow(5.5, 4.95, 5.5, 4.1)

    box(5.5, 3.5, 2.4, 1.0, 'Normalize\n0 – 100', '#0277BD')
    arrow(5.5, 3.0, 5.5, 2.1)

    # Step 5
    diamond(5.5, 1.5, 3.0, 1.2,
            'Precision-Recall\nThreshold\n= 0.777', '#00695C')

    # Step 6
    arrow(7.0, 1.5, 9.0, 2.5)
    box(9.5, 2.8, 2.4, 1.0, 'Churn_Prob\n≥ 0.777?', '#2E7D32')
    arrow(9.5, 2.3, 9.5, 1.3, 'ใช่')
    box(9.5, 0.8, 2.2, 0.75, 'เข้ารายการ (KEEP)', '#388E3C')
    arrow(10.7, 2.8, 12.5, 1.3, 'ไม่ใช่')
    box(13.0, 0.8, 2.0, 0.75, 'ตัดออก (SKIP)', '#C62828')

    # Output
    arrow(9.5, 0.42, 12.5, 4.5)
    box(13.5, 5.2, 2.2, 2.0,
        'ผลลัพธ์\ncoupon_target\n_list.csv\n310 ราย\n(5.5%)', '#1A237E')

    # Metrics bar ด้านบน
    ax.text(7.5, 8.4,
            'Precision = 100%   |   ลดของเสีย (Waste) = 94.5%   |   Threshold = 0.777',
            ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.45', facecolor='#E8F5E9',
                      edgecolor='#2E7D32', alpha=0.92))

    ax.set_title('กระบวนการเลือกกลุ่มลูกค้าเพื่อส่งคูปอง (Coupon Targeting Flow)',
                 fontsize=13, fontweight='bold', pad=10)
    plt.tight_layout()
    savefig('coupon_flowchart.png')


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('กำลัง Generate ภาพ...\n')
    make_data_pipeline()
    make_stacking_architecture()
    make_dataset_overview()
    make_value_risk_quadrant()
    make_coupon_flowchart()
    print(f'\n✅ เสร็จแล้ว — บันทึกใน: {OUTPUT_DIR}')
