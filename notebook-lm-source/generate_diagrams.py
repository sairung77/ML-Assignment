import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyArrow
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'images')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1. data_pipeline.png
# ─────────────────────────────────────────────
def make_data_pipeline():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis('off')
    fig.patch.set_facecolor('#F8F9FA')

    stages = [
        ('Raw Data\n(E-Commerce\nDataset)', '#4C72B0', 1.0),
        ('Notebook 1\nEDA & RFM\nClustering', '#DD8452', 3.5),
        ('Notebook 2\nStacking Model\nTraining', '#55A868', 6.0),
        ('Notebook 3\nLoyalty\nProgram', '#C44E52', 8.5),
        ('Notebook 4\nCoupon\nTargeting', '#8172B2', 11.0),
        ('Business\nOutputs', '#937860', 13.5),
    ]

    for label, color, x in stages:
        box = FancyBboxPatch((x - 0.9, 1.5), 1.8, 2.0,
                             boxstyle='round,pad=0.1',
                             linewidth=1.5, edgecolor='white',
                             facecolor=color, alpha=0.9)
        ax.add_patch(box)
        ax.text(x, 2.5, label, ha='center', va='center',
                fontsize=9, color='white', fontweight='bold')

    arrow_xs = [(1.9, 2.6), (4.4, 5.1), (6.9, 7.6), (9.4, 10.1), (11.9, 12.6)]
    for x1, x2 in arrow_xs:
        ax.annotate('', xy=(x2, 2.5), xytext=(x1, 2.5),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=2))

    outputs = ['K-Means Clusters\nRFM Segments', 'predictions.csv\n(Churn Prob)', 'rescue_priority\n_list.csv', 'coupon_target\n_list.csv']
    ox = [3.5, 6.0, 8.5, 11.0]
    for label, x in zip(outputs, ox):
        ax.text(x, 0.9, label, ha='center', va='center', fontsize=7.5,
                color='#333', style='italic',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#ccc', alpha=0.8))
        ax.annotate('', xy=(x, 1.5), xytext=(x, 1.1),
                    arrowprops=dict(arrowstyle='->', color='#aaa', lw=1.2))

    ax.set_title('ML-Assignment: End-to-End Data Pipeline', fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'data_pipeline.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('✓ data_pipeline.png')


# ─────────────────────────────────────────────
# 2. stacking_architecture.png
# ─────────────────────────────────────────────
def make_stacking_architecture():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')
    fig.patch.set_facecolor('#F8F9FA')

    # Input
    inp = FancyBboxPatch((0.3, 2.8), 1.8, 1.4, boxstyle='round,pad=0.1',
                         linewidth=1.5, edgecolor='#555', facecolor='#4C72B0', alpha=0.85)
    ax.add_patch(inp)
    ax.text(1.2, 3.5, 'Input\nFeatures\n(20 cols)', ha='center', va='center',
            fontsize=8.5, color='white', fontweight='bold')

    # Layer 0
    base_models = [
        ('Random\nForest', '#2196F3'),
        ('XGBoost', '#4CAF50'),
        ('LightGBM', '#FF9800'),
        ('Gradient\nBoosting', '#9C27B0'),
        ('Logistic\nRegression', '#F44336'),
    ]
    layer0_y = [6.0, 4.8, 3.6, 2.4, 1.2]
    for (name, color), y in zip(base_models, layer0_y):
        box = FancyBboxPatch((3.0, y - 0.45), 2.2, 0.9, boxstyle='round,pad=0.08',
                             linewidth=1, edgecolor='white', facecolor=color, alpha=0.85)
        ax.add_patch(box)
        ax.text(4.1, y, name, ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        # arrow from input
        ax.annotate('', xy=(3.0, y), xytext=(2.1, 3.5),
                    arrowprops=dict(arrowstyle='->', color='#888', lw=1.2,
                                   connectionstyle='arc3,rad=0'))

    ax.text(4.1, 6.8, 'Layer 0 — Base Models\n(StratifiedKFold CV=5)', ha='center', va='center',
            fontsize=9, fontweight='bold', color='#333')

    # OOF predictions
    for y in layer0_y:
        ax.annotate('', xy=(7.5, 3.5), xytext=(5.2, y),
                    arrowprops=dict(arrowstyle='->', color='#888', lw=1.2,
                                   connectionstyle='arc3,rad=0'))

    oof_box = FancyBboxPatch((7.0, 2.6), 1.8, 1.8, boxstyle='round,pad=0.1',
                              linewidth=1.5, edgecolor='#555', facecolor='#607D8B', alpha=0.85)
    ax.add_patch(oof_box)
    ax.text(7.9, 3.5, 'OOF\nPredictions\n(5 cols)', ha='center', va='center',
            fontsize=8.5, color='white', fontweight='bold')

    # Layer 1
    ax.annotate('', xy=(10.2, 3.5), xytext=(8.8, 3.5),
                arrowprops=dict(arrowstyle='->', color='#555', lw=2))

    meta_box = FancyBboxPatch((10.2, 2.65), 2.2, 1.7, boxstyle='round,pad=0.1',
                               linewidth=2, edgecolor='#FFD700', facecolor='#E91E63', alpha=0.9)
    ax.add_patch(meta_box)
    ax.text(11.3, 3.5, 'Meta Learner\nLogistic\nRegression\n(Layer 1)', ha='center', va='center',
            fontsize=8.5, color='white', fontweight='bold')
    ax.text(11.3, 6.8, 'Layer 1 — Meta Model', ha='center', fontsize=9, fontweight='bold', color='#333')

    # Output
    ax.annotate('', xy=(13.0, 3.5), xytext=(12.4, 3.5),
                arrowprops=dict(arrowstyle='->', color='#555', lw=2))
    out_box = FancyBboxPatch((13.0, 2.8), 1.0, 1.4, boxstyle='round,pad=0.1',
                              linewidth=1.5, edgecolor='#555', facecolor='#4CAF50', alpha=0.9)
    ax.add_patch(out_box)
    ax.text(13.5, 3.5, 'Churn\nProb', ha='center', va='center',
            fontsize=8, color='white', fontweight='bold')

    # Metrics
    metrics_text = 'ROC-AUC: 0.9974   Accuracy: 98.05%   F1: 0.9433'
    ax.text(7.0, 0.4, metrics_text, ha='center', va='center', fontsize=10,
            fontweight='bold', color='#1a237e',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD', edgecolor='#1a237e', alpha=0.9))

    ax.set_title('Stacking Ensemble Architecture', fontsize=13, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'stacking_architecture.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('✓ stacking_architecture.png')


# ─────────────────────────────────────────────
# 3. dataset_overview.png
# ─────────────────────────────────────────────
def make_dataset_overview():
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('#F8F9FA')
    fig.suptitle('E-Commerce Churn Dataset Overview  (5,630 customers × 20 features)',
                 fontsize=14, fontweight='bold', y=0.97)

    # ── Churn Pie ──
    ax1 = fig.add_axes([0.02, 0.52, 0.22, 0.38])
    sizes = [83.16, 16.84]
    colors = ['#4CAF50', '#F44336']
    wedges, texts, autotexts = ax1.pie(sizes, labels=['Non-Churn\n(4,682)', 'Churn\n(948)'],
                                        colors=colors, autopct='%1.1f%%',
                                        startangle=90, pctdistance=0.75,
                                        wedgeprops=dict(width=0.6))
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight('bold')
    ax1.set_title('Class Distribution', fontsize=10, fontweight='bold')

    # ── Missing Values ──
    ax2 = fig.add_axes([0.27, 0.52, 0.35, 0.38])
    cols_missing = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp',
                    'OrderAmountHikeFromLastYear', 'CouponUsed',
                    'OrderCount', 'DaySinceLastOrder']
    miss_pct = [4.46, 5.45, 4.96, 4.78, 5.09, 4.96, 5.27]
    colors_bar = ['#FF7043'] * len(cols_missing)
    bars = ax2.barh(cols_missing, miss_pct, color=colors_bar, edgecolor='white', height=0.6)
    ax2.set_xlim(0, 7)
    ax2.axvline(5.0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    for bar, pct in zip(bars, miss_pct):
        ax2.text(pct + 0.05, bar.get_y() + bar.get_height() / 2,
                 f'{pct}%', va='center', fontsize=8.5)
    ax2.set_xlabel('Missing %', fontsize=9)
    ax2.set_title('Missing Values (→ Median Imputation)', fontsize=10, fontweight='bold')
    ax2.tick_params(labelsize=8.5)

    # ── Feature Types ──
    ax3 = fig.add_axes([0.65, 0.52, 0.15, 0.38])
    types = ['Numerical\n(continuous)', 'Numerical\n(discrete)', 'Binary /\nCategorical']
    counts = [8, 9, 3]
    colors3 = ['#42A5F5', '#26C6DA', '#AB47BC']
    ax3.bar(types, counts, color=colors3, edgecolor='white')
    for i, v in enumerate(counts):
        ax3.text(i, v + 0.05, str(v), ha='center', fontsize=10, fontweight='bold')
    ax3.set_ylim(0, 12)
    ax3.set_title('Feature Types', fontsize=10, fontweight='bold')
    ax3.tick_params(labelsize=8)

    # ── Top 5 Feature Importance ──
    ax4 = fig.add_axes([0.02, 0.06, 0.45, 0.38])
    features = ['Tenure', 'Complain', 'CashbackAmount',
                'DaySinceLastOrder', 'SatisfactionScore']
    importance = [0.312, 0.218, 0.156, 0.134, 0.089]
    bars4 = ax4.barh(features[::-1], importance[::-1],
                     color=['#1565C0', '#1976D2', '#1E88E5', '#2196F3', '#42A5F5'],
                     edgecolor='white', height=0.6)
    for bar, val in zip(bars4, importance[::-1]):
        ax4.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                 f'{val:.3f}', va='center', fontsize=8.5)
    ax4.set_xlabel('MDI Importance', fontsize=9)
    ax4.set_title('Top 5 Feature Importance (MDI)', fontsize=10, fontweight='bold')
    ax4.set_xlim(0, 0.38)
    ax4.tick_params(labelsize=9)

    # ── Stats Table ──
    ax5 = fig.add_axes([0.52, 0.06, 0.46, 0.38])
    ax5.axis('off')
    table_data = [
        ['Metric', 'Value'],
        ['Total Customers', '5,630'],
        ['Training Set', '4,504 (80%)'],
        ['Test Set', '1,126 (20%)'],
        ['CV Folds', '5 (StratifiedKFold)'],
        ['ROC-AUC (Stacking)', '0.9974'],
        ['Accuracy', '98.05%'],
        ['F1 Score', '0.9433'],
        ['RESCUE customers', '314 (5.6%)'],
        ['Coupon recipients', '310 (5.5%)'],
    ]
    col_widths = [0.55, 0.45]
    table = ax5.table(cellText=table_data[1:], colLabels=table_data[0],
                       cellLoc='center', loc='center',
                       colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#1565C0')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#E3F2FD')
        cell.set_edgecolor('#BBDEFB')
    ax5.set_title('Key Statistics', fontsize=10, fontweight='bold', pad=6)

    plt.savefig(os.path.join(OUTPUT_DIR, 'dataset_overview.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('✓ dataset_overview.png')


# ─────────────────────────────────────────────
# 4. value_risk_quadrant_detail.png
# ─────────────────────────────────────────────
def make_value_risk_quadrant():
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Quadrant boxes
    quads = [
        # (x, y, w, h, color, label, count, pct, strategy)
        (0.2, 5.2, 4.6, 4.6, '#E8F5E9', 'PROTECT', 2537, '45.1%',
         'High Value, Low Churn Risk\n• VIP rewards & exclusive benefits\n• Upsell premium products\n• Loyalty point multipliers'),
        (5.2, 5.2, 4.6, 4.6, '#FFEBEE', 'RESCUE', 314, '5.6%',
         'High Value, High Churn Risk\n• Immediate personal outreach\n• Targeted retention offers\n• ROI-optimized coupons'),
        (0.2, 0.2, 4.6, 4.6, '#E3F2FD', 'MAINTAIN', 2114, '37.6%',
         'Low Value, Low Churn Risk\n• Standard loyalty program\n• Encourage higher spend\n• Automated engagement'),
        (5.2, 0.2, 4.6, 4.6, '#FFF9C4', 'LET GO', 665, '11.8%',
         'Low Value, High Churn Risk\n• Minimal retention investment\n• Natural attrition accepted\n• Exclude from coupon campaign'),
    ]
    quad_colors = {'PROTECT': '#2E7D32', 'RESCUE': '#C62828',
                   'MAINTAIN': '#1565C0', 'LET GO': '#F57F17'}

    for x, y, w, h, bg, label, count, pct, strategy in quads:
        box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.15',
                             linewidth=2, edgecolor=quad_colors[label], facecolor=bg, alpha=0.9)
        ax.add_patch(box)
        # Label
        ax.text(x + w / 2, y + h - 0.5, label, ha='center', va='center',
                fontsize=15, fontweight='bold', color=quad_colors[label])
        # Count & pct
        ax.text(x + w / 2, y + h - 1.1, f'{count:,} customers ({pct})',
                ha='center', va='center', fontsize=10, color='#555')
        # Strategy
        for i, line in enumerate(strategy.split('\n')):
            ax.text(x + w / 2, y + h - 1.8 - i * 0.65, line,
                    ha='center', va='center', fontsize=8.5, color='#333')

    # Dividing lines
    ax.plot([5.0, 5.0], [0.1, 9.9], color='#777', lw=2, linestyle='--')
    ax.plot([0.1, 9.9], [5.0, 5.0], color='#777', lw=2, linestyle='--')

    # Axis labels
    ax.text(5.0, -0.3, 'CashbackAmount  →', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#333')
    ax.text(0.25, 5.2, 'Low Cashback\n(< 163 Baht)', ha='center', va='center',
            fontsize=8.5, color='#666', rotation=90)
    ax.text(9.75, 5.2, 'High Cashback\n(≥ 163 Baht)', ha='center', va='center',
            fontsize=8.5, color='#666', rotation=90)
    ax.text(2.65, 9.9, 'Low Churn Risk\n(< 35%)', ha='center', va='center', fontsize=8.5, color='#666')
    ax.text(7.35, 9.9, 'High Churn Risk\n(≥ 35%)', ha='center', va='center', fontsize=8.5, color='#666')

    # Threshold annotations
    ax.text(5.0, 9.5, '← Cashback Median = 163 Baht →', ha='center', fontsize=9,
            color='#555', style='italic')
    ax.text(9.85, 5.0, 'Churn\nThreshold\n= 0.35', ha='center', va='center',
            fontsize=8, color='#555', style='italic')

    ax.set_title('Value-Risk Quadrant Analysis  (Total: 5,630 customers)',
                 fontsize=13, fontweight='bold', pad=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'value_risk_quadrant_detail.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('✓ value_risk_quadrant_detail.png')


# ─────────────────────────────────────────────
# 5. coupon_flowchart.png
# ─────────────────────────────────────────────
def make_coupon_flowchart():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    fig.patch.set_facecolor('#F8F9FA')

    def draw_box(x, y, w, h, text, color, text_color='white', style='round,pad=0.15'):
        box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                             boxstyle=style, linewidth=1.5,
                             edgecolor='white', facecolor=color, alpha=0.9)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=8.5,
                color=text_color, fontweight='bold', wrap=True)

    def draw_diamond(x, y, w, h, text, color):
        diamond = plt.Polygon([[x, y + h / 2], [x + w / 2, y],
                                [x, y - h / 2], [x - w / 2, y]],
                               facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.9)
        ax.add_patch(diamond)
        ax.text(x, y, text, ha='center', va='center', fontsize=8, color='white', fontweight='bold')

    def arrow(x1, y1, x2, y2, label=''):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.8))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.1, my, label, fontsize=8, color='#555', style='italic')

    # Nodes
    draw_box(1.3, 6.5, 2.0, 0.9, 'predictions.csv\n(5,630 rows)', '#4C72B0')
    draw_diamond(1.3, 4.8, 2.4, 1.2, 'Value-Risk\nSegmentation', '#7B1FA2')
    draw_box(1.3, 3.0, 2.0, 0.9, 'Remove\nLET GO (665)', '#C62828')
    draw_box(1.3, 1.4, 2.0, 0.9, 'RESCUE pool\n(314 customers)', '#E65100')

    draw_box(5.0, 4.8, 2.2, 0.9, 'ROI Score\n= Churn_Prob ×\nCashbackAmount', '#1565C0')
    draw_box(5.0, 3.0, 2.2, 0.9, 'Normalize\nROI Score\n→ 0–100', '#0277BD')
    draw_diamond(5.0, 1.4, 2.6, 1.2, 'Apply PR Curve\nThreshold\n(0.777)', '#00695C')

    draw_box(9.0, 2.2, 2.4, 0.9, 'Churn_Prob\n≥ 0.777?', '#2E7D32', style='round,pad=0.1')
    draw_box(9.0, 0.7, 2.2, 0.7, 'YES → Keep', '#388E3C')
    draw_box(11.5, 0.7, 2.2, 0.7, 'NO → Exclude', '#C62828')

    draw_box(12.5, 4.0, 2.0, 1.6, 'Final Output\ncoupon_target\n_list.csv\n310 customers\n(5.5%)', '#1A237E')

    # Arrows
    arrow(1.3, 6.05, 1.3, 5.4)
    arrow(1.3, 4.2, 1.3, 3.45)
    ax.text(1.65, 3.8, 'RESCUE\n+PROTECT', fontsize=7.5, color='#555', style='italic')
    arrow(1.3, 2.55, 1.3, 1.85)
    arrow(2.3, 1.4, 3.9, 4.8)
    arrow(5.0, 4.35, 5.0, 3.45)
    arrow(5.0, 2.55, 5.0, 1.95)
    arrow(6.3, 1.4, 7.7, 1.4)
    ax.text(6.8, 1.55, 'Filter', fontsize=8, color='#555', style='italic')
    arrow(8.8, 1.4, 8.0, 2.2)
    arrow(9.0, 1.75, 9.0, 1.05)
    ax.text(9.15, 1.4, 'Yes', fontsize=8, color='#2E7D32', fontweight='bold')
    arrow(10.1, 2.2, 11.5, 1.05)
    ax.text(10.6, 1.7, 'No', fontsize=8, color='#C62828', fontweight='bold')
    arrow(9.0, 0.35, 12.0, 3.2)

    # Metrics
    ax.text(7.0, 6.8, 'Key Metrics: Precision=100%   Waste Reduction=94.5%   Threshold=0.777',
            ha='center', fontsize=9.5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9', edgecolor='#2E7D32', alpha=0.9))

    ax.set_title('Coupon Targeting Decision Flow', fontsize=13, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'coupon_flowchart.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('✓ coupon_flowchart.png')


if __name__ == '__main__':
    print('Generating diagrams...')
    make_data_pipeline()
    make_stacking_architecture()
    make_dataset_overview()
    make_value_risk_quadrant()
    make_coupon_flowchart()
    print(f'\nAll 5 diagrams saved to: {OUTPUT_DIR}')
