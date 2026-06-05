# PromoWise — Technical Documentation

## Table of Contents

1. [Application Overview](#1-application-overview)
2. [Architecture](#2-architecture)
3. [Data](#3-data)
4. [Model Training Pipeline](#4-model-training-pipeline)
5. [Prediction & Recommendation Logic](#5-prediction--recommendation-logic)
6. [Feature Reference](#6-feature-reference)
7. [Model Performance](#7-model-performance)
8. [Known Limitations](#8-known-limitations)
9. [Deployment Guide](#9-deployment-guide)

---

## 1. Application Overview

PromoWise is a Streamlit-based decision-support tool for Sunshine Grocery's orange juice category. Given a store, brand, forecast week, price, and promotion configuration, it:

1. Predicts weekly demand in units sold using a trained LightGBM model.
2. Computes the predicted lift over a no-promotion baseline.
3. Translates the lift and margin into one of five campaign recommendations.

The tool is designed for weekly use by store managers or category planners. It does not replace human judgement — it informs it.

---

## 2. Architecture

```
promowise_streamlit_app/
├── app.py                  # Streamlit UI and prediction logic
├── train_model.py          # Model training script
├── requirements.txt        # Python dependencies
├── README.md               # Quick-start guide
├── TECHNICAL_DOCUMENTATION.md
├── model/
│   ├── promowise_demand_model.pkl   # Serialised sklearn Pipeline (preprocess + LightGBM)
│   └── model_metadata.json          # Feature columns, metrics, training notes
└── data/
    ├── sunshine_grocery_powerbi_ready.csv  # Full historical dataset
    ├── reference_data.csv                  # Most-recent-week snapshot per store/brand
    └── model_validation_results.csv        # Per-model validation metrics
```

**Runtime flow:**

```
User input (sidebar)
    │
    ▼
input_row built from reference_data.csv defaults
    │
    ▼
model.predict(input_df)  ─── sklearn Pipeline ───► LightGBM
    │                          (impute → encode)
    ▼
pred_log_sales  ──► np.exp()  ──► pred_units
    │
    ▼
baseline_units (same pipeline, promotions zeroed out)
    │
    ▼
lift% + profit_per_unit  ──► make_recommendation()  ──► action + explanation
```

---

## 3. Data

### 3.1 `sunshine_grocery_powerbi_ready.csv`

The full historical dataset. Each row is one store–brand–week observation.

| Column group | Examples | Notes |
|---|---|---|
| Identity | `store_id`, `brand_name`, `week_id`, `brand_id` | `brand_id` is dropped (redundant with `brand_name`) |
| Prices | `brand_price`, `price_tropicana_premium_64oz`, … | All competitor SKU prices included as features |
| Promotion flags | `is_coupon_promotion`, `is_feature_ad`, `has_any_promotion`, `promotion_type` | Binary flags + derived string label |
| Financials | `profit_per_unit` | Entered by the user at runtime; also present in historical data |
| Demographics | `pct_age_over_60`, `pct_college_educated`, `median_income`, … | Store catchment area characteristics |
| Competition | `avg_distance_to_5_competitors`, `sales_ratio_vs_5_competitors`, … | Competitive context per store |
| Targets | `units_sold`, `log_sales` | `log_sales = log(units_sold)`; only `log_sales` is used as the ML target |
| Constant | `constant` | Zero-variance column; always dropped |

### 3.2 `reference_data.csv`

A per-store/brand snapshot of the most recent available week. Used by `app.py` to populate default sidebar values so the user always starts from a realistic baseline.

Built by `train_model.py`:
```python
df.sort_values("week_id")
  .groupby(["store_id", "brand_name"])
  .tail(1)
```

### 3.3 `model_validation_results.csv`

Contains the validation-set MAE, RMSE, and R² for each candidate model. Displayed in the app's "Model performance" expander.

---

## 4. Model Training Pipeline

### 4.1 Data splits (chronological)

| Split | Week range | Purpose |
|---|---|---|
| Train | week_id ≤ 112 | Model fitting |
| Validation | 113 ≤ week_id ≤ 142 | Model selection |
| Test | week_id ≥ 143 | Final held-out evaluation |

Splits are chronological to prevent temporal leakage — the model never sees future weeks during training.

### 4.2 Sampling (MVP setting)

```python
main(sample_train=12000, sample_eval=5000)
```

Training uses a 12 000-row sample for speed. Remove or increase `sample_train` in `train_model.py` before a production retrain.

### 4.3 Preprocessing (`ColumnTransformer`)

| Step | Columns | Strategy |
|---|---|---|
| Numeric imputation | All numeric features | Median (robust to outliers) |
| Categorical imputation | `store_id`, `promotion_type`, `brand_name`, `income_segment` | Most frequent |
| Ordinal encoding | Same categorical columns | Unknown categories → `-1` at inference time |

### 4.4 Candidate models

| Model | Key hyperparameters |
|---|---|
| Decision Tree Baseline | `max_depth=10`, `min_samples_leaf=20` |
| LightGBM Gradient Boosting | `n_estimators=45`, `learning_rate=0.08`, `max_depth=5`, `num_leaves=20` |

### 4.5 Model selection

The model with the highest R² on the validation set is automatically selected and saved as `promowise_demand_model.pkl`.

### 4.6 Output artefacts

| File | Contents |
|---|---|
| `model/promowise_demand_model.pkl` | Full sklearn Pipeline (preprocessor + chosen model) |
| `model/model_metadata.json` | Selected model name, metrics, feature column lists, business use description |
| `data/reference_data.csv` | Inference baseline snapshot |
| `data/model_validation_results.csv` | Per-model validation scores |

---

## 5. Prediction & Recommendation Logic

### 5.1 Prediction

The model predicts `log_sales`. The app back-transforms to units:

```python
pred_log_sales = model.predict(input_df)[0]
pred_units = np.exp(pred_log_sales)
```

A **baseline prediction** is computed with all promotion flags set to zero, keeping all other inputs identical. This isolates the promotional effect.

### 5.2 Lift calculation

```python
lift = (pred_units - baseline_units) / max(baseline_units, 1)
```

### 5.3 Recommendation rules

| Condition | Action | Rationale |
|---|---|---|
| No promotion active AND predicted ≥ 105% of baseline | **Protect Margin** | Demand is already healthy — discounting is unnecessary |
| lift ≥ 20% AND profit_per_unit ≥ 20 | **Promote** | Strong lift and acceptable margin justify the campaign |
| lift ≥ 5% AND profit_per_unit ≥ 15 | **Continue** | Useful but should be monitored for diminishing returns |
| lift ≥ 5% AND profit_per_unit < 15 | **Redesign** | Lift exists but margin is too thin — reduce discount depth |
| lift < 5% | **Stop** | Predicted lift too small to justify promotional cost |

### 5.4 Promotion type inference

```
is_coupon AND is_feature  →  "Coupon + Feature"
is_coupon only            →  "Coupon Only"
is_feature only           →  "Feature Only"
neither                   →  "No Promotion"
```

---

## 6. Feature Reference

The model uses 32 input features. Key groups:

| Group | Features | Why included |
|---|---|---|
| Store identity | `store_id` | Captures store-level fixed effects |
| Time | `week_id` | Captures seasonal and trend patterns |
| Own-brand price | `brand_price` | Primary own-brand price elasticity signal |
| Competitor prices | `price_tropicana_*`, `price_minute_maid_*`, etc. | Cross-price elasticity |
| Promotion | `is_coupon_promotion`, `is_feature_ad`, `has_any_promotion`, `promotion_type` | Promotional uplift drivers |
| Profit | `profit_per_unit` | Passed through to the recommendation layer |
| Demographics | `pct_age_over_60`, `pct_college_educated`, `median_income`, … | Store catchment demand sensitivity |
| Competition | `avg_distance_to_5_competitors`, `sales_ratio_vs_5_competitors`, … | Competitive vulnerability |
| Derived | `income_segment` | Bucketed income band for the store catchment |

---

## 7. Model Performance

Evaluated on `log_sales`. Lower MAE/RMSE and higher R² are better.

### Validation set

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Decision Tree Baseline | 0.581 | 0.793 | 0.507 |
| **LightGBM Gradient Boosting** | **0.540** | **0.706** | **0.609** |

### Held-out test set (sampled)

| Metric | Value |
|---|---:|
| MAE | 0.552 |
| RMSE | 0.732 |
| R² | 0.564 |

**Interpreting R² = 0.61:** The model explains about 61% of the variance in log-sales on unseen validation data. This is reasonable for a sampled MVP, but real-world unit-level accuracy depends on the back-transformation and high-volume SKU behaviour.

---

## 8. Known Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| Training on a 12 000-row sample | Likely underfitting; metrics are optimistic for full data | Remove sampling cap in `train_model.py` before production retrain |
| Only 45 LightGBM estimators | Model may not have converged | Increase `n_estimators` to 300–600 for production |
| Metrics are on log scale | A small log-error can be a large unit-error on high-volume SKUs | Evaluate `np.exp(pred)` vs `units_sold` for business-facing accuracy |
| Single validation split | Validation score carries sampling variance | Use walk-forward cross-validation for a more robust estimate |
| Hardcoded week_id boundaries | Breaks if the dataset is extended | Parameterise split boundaries or use a percentage-based split |
| No model monitoring | Silent drift as promotions and pricing evolve | Add prediction logging and periodic retraining schedule |
| No authentication | Anyone with the URL can use the app | Add Streamlit authentication or deploy behind an identity proxy |

---

## 9. Deployment Guide

### Local

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Streamlit Community Cloud (recommended for quick deployment)

1. Push the full project (including `model/` and `data/` folders) to a public or private GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repository.
3. Set the main file path to `app.py`.
4. Streamlit Cloud installs `requirements.txt` automatically and serves the app.

**Memory limit:** 1 GB. The current model and data files are well within this limit.

### Other platforms

| Platform | Start command | Notes |
|---|---|---|
| Railway | `streamlit run app.py --server.port $PORT` | Free tier; deploy from GitHub |
| Render | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` | Free tier; slightly more config |
| Hugging Face Spaces | Select "Streamlit" SDK when creating the Space | Free; good for public visibility |

### Retraining

```bash
python train_model.py
```

This overwrites `model/promowise_demand_model.pkl`, `model/model_metadata.json`, `data/reference_data.csv`, and `data/model_validation_results.csv`. Commit the updated artefacts to trigger a redeploy on Streamlit Cloud.
