# PromoWise Streamlit MVP

PromoWise is a simple machine-learning application for Sunshine Grocery's orange juice category.  
It predicts weekly demand and converts the prediction into a campaign recommendation: **Promote, Continue, Redesign, Stop, or Protect Margin**.

## Model

Selected model: **LightGBM Gradient Boosting**

Validation results:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Decision Tree Baseline | 0.581 | 0.793 | 0.507 |
| LightGBM Gradient Boosting | 0.540 | 0.706 | 0.609 |

Sampled test performance for the selected model:

- MAE: 0.552
- RMSE: 0.732
- R²: 0.564

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How to retrain the model

```bash
python train_model.py
```

## Folder structure

```text
promowise_streamlit_app/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── model/
│   ├── promowise_demand_model.pkl
│   └── model_metadata.json
└── data/
    ├── sunshine_grocery_powerbi_ready.csv
    ├── reference_data.csv
    └── model_validation_results.csv
```

## Production note

This is a **production-style MVP**, not a full enterprise production system.  
For real production, Sunshine Grocery would still need authentication, cloud hosting, database connection, model monitoring, retraining automation, governance controls, and user access management.
