# Stacking Ensemble Model (Notebook 2)

## Overview

Notebook 2 builds a 2-layer stacking ensemble that achieves near-perfect churn prediction. The model combines 5 diverse base learners in Layer 0 and a Logistic Regression meta-learner in Layer 1. The output is `predictions.csv` containing churn probabilities for all 5,630 customers.

## Architecture

![Stacking Architecture](images/stacking_architecture.png)

The stacking model uses Out-Of-Fold (OOF) predictions to train the meta-learner, preventing data leakage between layers.

## Base Model Comparison

![Base Model CV](images/base_model_cv.png)

Cross-validation results across all 5 base models, showing individual ROC-AUC and F1 scores before stacking.

## Model Evaluation

![Stacking Evaluation](images/stacking_evaluation.png)

Confusion matrix, ROC curve, and churn probability distribution for the final stacking model on the test set.

## Feature Importance

![Feature Importance](images/feature_importance.png)

MDI (Mean Decrease in Impurity) importance from Random Forest and permutation importance, both confirming the same top-5 ranking.

## Layer 0: Base Models

| # | Model | Key Hyperparameters |
|---|---|---|
| 1 | Random Forest | `n_estimators=300, max_depth=None, min_samples_leaf=1` |
| 2 | XGBoost | `n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8` |
| 3 | LightGBM | `n_estimators=300, learning_rate=0.05, num_leaves=31` |
| 4 | Gradient Boosting | `n_estimators=200, learning_rate=0.1, max_depth=5` |
| 5 | Logistic Regression | `C=1.0, max_iter=1000, solver='lbfgs'` |

## Layer 1: Meta-Learner

```python
meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
```

The meta-learner receives a 5-column matrix of OOF churn probabilities from Layer 0 and learns the optimal linear combination. Logistic Regression was chosen for:
- Interpretability (coefficients show which base model contributes most)
- Calibrated probability outputs
- Preventing overfitting from complex meta-learners

## Cross-Validation Strategy

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

- **5-fold StratifiedKFold** preserves the 83.16/16.84% class ratio in each fold
- Base models generate OOF predictions across all 5 folds
- Meta-learner is trained on the full OOF matrix
- Final evaluation on held-out test set (20% = 1,126 rows)

## OOF Prediction Generation

```python
oof_preds = np.zeros((len(X_train), n_base_models))

for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    for model_idx, model in enumerate(base_models):
        model.fit(X_train[train_idx], y_train[train_idx])
        oof_preds[val_idx, model_idx] = model.predict_proba(X_train[val_idx])[:, 1]

# Train meta-learner on OOF predictions
meta_model.fit(oof_preds, y_train)
```

## Performance Metrics

| Metric | Value |
|---|---|
| **ROC-AUC** | **0.9974** |
| **Accuracy** | **98.05%** |
| **F1 Score** | **0.9433** |
| Precision | ~0.97 |
| Recall | ~0.92 |

### Confusion Matrix (Test Set, n=1,126)

|  | Predicted Non-Churn | Predicted Churn |
|---|---|---|
| **Actual Non-Churn** | ~934 (TN) | ~3 (FP) |
| **Actual Churn** | ~19 (FN) | ~170 (TP) |

> False Negatives (missed churners) are more costly than False Positives in this business context.

## Top 5 Feature Importance

| Rank | Feature | MDI Importance | Business Interpretation |
|---|---|---|---|
| 1 | **Tenure** | 0.312 | Longer-tenured customers are far more stable |
| 2 | **Complain** | 0.218 | Single strongest binary signal for churn |
| 3 | **CashbackAmount** | 0.156 | High cashback = high value = higher retention effort |
| 4 | **DaySinceLastOrder** | 0.134 | Recency signal — lapsed customers are at-risk |
| 5 | **SatisfactionScore** | 0.089 | Moderate predictor; paradox at high scores |

## Why Stacking Outperforms Individual Models

1. **Diversity**: Each base model captures different patterns (trees vs linear vs boosted)
2. **OOF prevents leakage**: Meta-learner never sees predictions on data it was trained on
3. **Error correction**: Meta-learner learns when each base model is reliable
4. **Calibrated probabilities**: LR meta-learner produces well-calibrated churn probabilities critical for downstream threshold-based decisions

## Output

`predictions.csv` — 5,630 rows with columns including:
- `CustomerID`, `Churn` (actual), `Churn_Prob` (0.0–1.0), `Churn_Pred` (0/1 at 0.5 threshold)
- All original features preserved for downstream segmentation
