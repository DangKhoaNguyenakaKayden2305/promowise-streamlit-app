# PromoWise: ML-Powered Promotion Recommendation App

**PromoWise** is a Streamlit-based machine learning MVP built for **Sunshine Grocery’s orange juice category**.
The app predicts weekly product demand and converts the forecast into a simple campaign recommendation: **Promote, Continue, Redesign, Stop, or Protect Margin**.

Instead of only showing a prediction number, PromoWise helps retail managers understand what action they should take next.

---

## Live Demo

**Deployed app:** https://promowise-app.streamlit.app/

---

## App Preview

<img width="2553" height="1293" alt="image" src="https://github.com/user-attachments/assets/a5b767be-929e-4c0e-8948-c58cebd97add" />
Promotional Impact: 
<img width="2545" height="1267" alt="image" src="https://github.com/user-attachments/assets/145f996b-299a-4d7e-a47e-3bd251a3086d" />



---

## Business Problem

Sunshine Grocery needs a better way to plan orange juice promotions across different stores and brands.
Promotion decisions are often affected by price, discounts, competitor pricing, store demographics and previous sales patterns.

The main business questions are:

* Which products are likely to have stronger weekly demand?
* When should the business promote, continue, redesign or stop a campaign?
* How can prediction results be translated into simple actions for non-technical users?

PromoWise addresses this by combining machine learning prediction with business-rule recommendations.

---

## What the App Does

PromoWise allows users to input or select product and store-level information, then generates:

* Predicted weekly demand
* Campaign recommendation
* Business interpretation of the result
* Clear guidance for category and promotion planning

The goal is to make machine learning more practical for retail decision-making, not just technically accurate.

---

## Recommendation Logic

The model predicts weekly demand first. The app then converts the prediction into a business recommendation:

| Recommendation     | Meaning                                                                |
| ------------------ | ---------------------------------------------------------------------- |
| **Promote**        | Demand looks strong enough to support a campaign                       |
| **Continue**       | Current settings appear reasonable                                     |
| **Redesign**       | Campaign may need changes in pricing, targeting or promotion structure |
| **Stop**           | Demand signal is weak and campaign may not be worth continuing         |
| **Protect Margin** | Demand may be healthy, but discounting could reduce profitability      |

---

## Machine Learning Model

Selected model: **LightGBM Gradient Boosting**

LightGBM was selected because it performed better than the Decision Tree baseline and is suitable for structured retail data with numerical and categorical features.

### Validation Results

| Model                      |   MAE |  RMSE |    R² |
| -------------------------- | ----: | ----: | ----: |
| Decision Tree Baseline     | 0.581 | 0.793 | 0.507 |
| LightGBM Gradient Boosting | 0.540 | 0.706 | 0.609 |

### Sampled Test Performance

| Metric | Value |
| ------ | ----: |
| MAE    | 0.552 |
| RMSE   | 0.732 |
| R²     | 0.564 |

The selected model explains a meaningful portion of demand variation and provides a stronger predictive baseline than the simple decision tree model.

---

## Key Features

* Interactive Streamlit interface
* Weekly demand prediction
* Promotion recommendation engine
* LightGBM machine learning model
* Model validation comparison
* Business-friendly output for non-technical users
* Local retraining script
* Production-style folder structure

---

## Tech Stack

| Area             | Tools                  |
| ---------------- | ---------------------- |
| App              | Streamlit              |
| Language         | Python                 |
| Data Processing  | Pandas, NumPy          |
| Machine Learning | LightGBM, Scikit-learn |
| Model Storage    | Pickle                 |
| Deployment       | Streamlit Cloud        |
| Reporting Data   | CSV                    |

---

## Project Structure

```text
promowise_streamlit_app/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── model/
│   ├── promowise_demand_model.pkl
│   └── model_metadata.json
├── data/
│   ├── sunshine_grocery_powerbi_ready.csv
│   ├── reference_data.csv
│   └── model_validation_results.csv
└── images/
    └── promowise_dashboard.png
```

---

## How to Run Locally

Clone the repository and install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

---

## How to Retrain the Model

To retrain the model using the latest prepared dataset:

```bash
python train_model.py
```

The retraining script updates the saved model and metadata files inside the `model/` folder.

---

## Why This Project Matters

PromoWise demonstrates how machine learning can be connected to real business decision-making.
The project does not stop at model training. It turns prediction outputs into campaign recommendations that a retail manager can understand and act on.

This makes the project useful as a portfolio example for:

* Data Analyst roles
* Business Analyst roles
* Retail analytics roles
* Machine learning MVP projects
* Dashboard and reporting projects

---

## Production Note

This is a **production-style MVP**, not a full enterprise production system.

For a real production environment, Sunshine Grocery would still need:

* User authentication
* Cloud database connection
* Automated data pipeline
* Scheduled model retraining
* Model monitoring
* Access control
* Data governance controls
* Audit logging
* Business approval workflow for campaign decisions

---

## Future Improvements

Planned improvements include:

* Add store and brand-level performance charts
* Add confidence bands around demand predictions
* Connect the app to a live database
* Add model drift monitoring
* Add campaign history tracking
* Add user login for category managers
* Add Power BI dashboard integration
* Add explainability output for key prediction drivers

---

## Author

Built as a machine learning MVP for Sunshine Grocery’s orange juice category, with a focus on turning prediction into practical retail campaign decisions.
