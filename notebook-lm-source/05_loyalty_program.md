# Loyalty Program & Value-Risk Segmentation (Notebook 3)

## Overview

Notebook 3 takes `predictions.csv` from Notebook 2 and applies a 2-dimensional segmentation framework to categorize all 5,630 customers into four strategic quadrants. This allows the business to allocate retention resources efficiently based on both customer value and churn risk.

## Value-Risk Quadrant (Detailed)

![Value-Risk Quadrant Detail](images/value_risk_quadrant_detail.png)

## Value-Risk Scatter Plot

![Value-Risk Matrix](images/value_risk_matrix.png)

Each point represents one customer, colored by actual churn label, positioned by their cashback amount and churn probability.

## Segmentation Framework

### Thresholds

| Parameter | Value | Rationale |
|---|---|---|
| `VALUE_MEDIAN` (CashbackAmount) | **163 Baht** | Median split — above median = "High Value" |
| `CHURN_THRESHOLD` (Churn_Prob) | **0.35** | Business-defined risk threshold — above 35% = "High Risk" |

```python
VALUE_MEDIAN = df['CashbackAmount'].median()   # = 163 Baht
CHURN_THRESHOLD = 0.35

df['Value_Segment'] = np.where(df['CashbackAmount'] >= VALUE_MEDIAN, 'High', 'Low')
df['Risk_Segment'] = np.where(df['Churn_Prob'] >= CHURN_THRESHOLD, 'High', 'Low')
df['Quadrant'] = df['Value_Segment'] + '_' + df['Risk_Segment']
```

## Quadrant Definitions & Counts

| Quadrant | Label | Count | % of Total | Churn Risk | Value |
|---|---|---|---|---|---|
| High Value + Low Risk | **PROTECT** | 2,537 | 45.1% | < 35% | ≥ 163 Baht |
| High Value + High Risk | **RESCUE** | 314 | 5.6% | ≥ 35% | ≥ 163 Baht |
| Low Value + Low Risk | **MAINTAIN** | 2,114 | 37.6% | < 35% | < 163 Baht |
| Low Value + High Risk | **LET GO** | 665 | 11.8% | ≥ 35% | < 163 Baht |

**Total**: 2,537 + 314 + 2,114 + 665 = **5,630** ✓

## RESCUE Segment Profile

The 314 RESCUE customers are the primary targets for intensive retention:

| Metric | RESCUE Segment | All Customers |
|---|---|---|
| Avg Churn Probability | **96.4%** | 16.84% |
| Avg CashbackAmount | **202 Baht** | 163 Baht |
| Avg Tenure | **5.58 months** | ~10 months |
| % with Complaint | ~45% | ~17% |

**Key finding**: RESCUE customers are disproportionately newer customers (short tenure) who have registered complaints. They generate above-average revenue but are leaving early in their lifecycle.

## Strategy Per Quadrant

### PROTECT (2,537 customers — 45.1%)
**Goal**: Maintain loyalty, prevent future drift into High Risk

- VIP tier upgrades and exclusive member benefits
- Loyalty point multipliers on premium categories
- Early access to new products and sales
- Proactive satisfaction surveys
- *Do NOT assume they are safe forever — monitor churn probability quarterly*

### RESCUE (314 customers — 5.6%)
**Goal**: Immediate retention intervention before churn occurs

- Personal outreach from customer success team within 48 hours
- Complaint resolution fast-track (if `Complain=1`)
- Personalized retention offers (coupons, cashback bonuses)
- ROI-optimized coupon targeting (→ Notebook 4)
- Re-engagement campaigns for lapsed orderers

**Offer examples**:
- "We miss you" cashback bonus (20–30% on next order)
- Free shipping for next 3 orders
- Personalized category discount based on `PreferedOrderCat`

### MAINTAIN (2,114 customers — 37.6%)
**Goal**: Gradually increase value, prevent sliding into LET GO

- Standard loyalty program enrollment
- Behavioral nudges to increase order frequency
- Category recommendations to raise average order value
- Automated email/app engagement campaigns

### LET GO (665 customers — 11.8%)
**Goal**: Minimize spend, accept natural attrition

- No coupon investment in this segment
- Remove from intensive marketing campaigns
- If they re-engage organically, welcome back
- Cost savings: eliminating 665 unnecessary coupons per cycle

## Cost-Benefit Analysis

| Action | Without Segmentation | With Segmentation |
|---|---|---|
| Coupons distributed | 5,630 (all customers) | 310–314 (RESCUE only) |
| Wasted on non-churners | ~4,682 coupons | ~0–4 coupons |
| Efficiency | 16.84% relevant | ~100% relevant |
| Cost savings | Baseline | ~94.5% reduction |

## Output

`rescue_priority_list.csv` — 314 rows sorted by `Churn_Prob` descending:
- All original features + `Churn_Prob` + `Quadrant` label
- Ready for Notebook 4 coupon optimization
- Also usable directly for manual outreach campaigns
