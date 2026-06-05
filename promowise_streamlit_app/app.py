import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="PromoWise | Sunshine Grocery",
    page_icon="🍊",
    layout="wide",
)

MODEL_PATH = Path("model/promowise_demand_model.pkl")
METADATA_PATH = Path("model/model_metadata.json")
REFERENCE_PATH = Path("data/reference_data.csv")
RESULTS_PATH = Path("data/model_validation_results.csv")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    reference = pd.read_csv(REFERENCE_PATH)
    results = pd.read_csv(RESULTS_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    return reference, results, metadata


def infer_promotion_type(coupon: int, feature: int) -> str:
    if coupon and feature:
        return "Coupon + Feature"
    if coupon:
        return "Coupon Only"
    if feature:
        return "Feature Only"
    return "No Promotion"


def make_recommendation(predicted_units, baseline_units, coupon, feature, profit_per_unit):
    lift = (predicted_units - baseline_units) / max(baseline_units, 1)

    if coupon == 0 and feature == 0 and predicted_units >= baseline_units * 1.05:
        return "Protect Margin", "Predicted demand is already healthy without discounting."
    if lift >= 0.20 and profit_per_unit >= 20:
        return "Promote", "Strong predicted lift with acceptable unit profit."
    if lift >= 0.05 and profit_per_unit >= 15:
        return "Continue", "Promotion appears useful, but should be monitored."
    if lift >= 0.05 and profit_per_unit < 15:
        return "Redesign", "Demand lift exists, but margin is weak. Consider lower discounting or feature-only promotion."
    return "Stop", "Predicted lift is too small to justify promotional effort."


def format_metric(value):
    return f"{value:,.0f}"


model = load_model()
reference, results, metadata = load_data()

st.title("🍊 PromoWise: Orange Juice Campaign Recommendation App")
st.caption(
    "A machine-learning MVP for Sunshine Grocery. It predicts weekly orange juice demand and translates the result into a simple promotion action."
)

with st.sidebar:
    st.header("Campaign Input")

    stores = sorted(reference["store_id"].unique())
    brands = sorted(reference["brand_name"].unique())

    store_id = st.selectbox("Store", stores)
    brand_name = st.selectbox("Brand", brands)

    base_rows = reference[
        (reference["store_id"] == store_id) & (reference["brand_name"] == brand_name)
    ]

    if base_rows.empty:
        base_row = reference[reference["store_id"] == store_id].iloc[0].copy()
        base_row["brand_name"] = brand_name
    else:
        base_row = base_rows.iloc[0].copy()

    week_id = st.slider(
        "Forecast week",
        int(reference["week_id"].min()),
        int(reference["week_id"].max()) + 12,
        int(min(reference["week_id"].max() + 1, reference["week_id"].max() + 12)),
    )

    brand_price = st.number_input(
        "Own brand price",
        min_value=0.00,
        max_value=1.00,
        value=float(base_row["brand_price"]),
        step=0.001,
        format="%.3f",
    )

    coupon = st.toggle("Coupon promotion", value=bool(base_row["is_coupon_promotion"]))
    feature = st.toggle("Feature advertising", value=bool(base_row["is_feature_ad"]))

    profit_per_unit = st.number_input(
        "Profit per unit",
        min_value=0.00,
        max_value=150.00,
        value=float(base_row["profit_per_unit"]),
        step=1.0,
    )

input_row = base_row.copy()
input_row["week_id"] = week_id
input_row["brand_price"] = brand_price
input_row["is_coupon_promotion"] = int(coupon)
input_row["is_feature_ad"] = int(feature)
input_row["has_any_promotion"] = int(coupon or feature)
input_row["promotion_type"] = infer_promotion_type(int(coupon), int(feature))
input_row["profit_per_unit"] = profit_per_unit

feature_cols = metadata["feature_columns"]
input_df = pd.DataFrame([input_row])[feature_cols]

pred_log_sales = float(model.predict(input_df)[0])
pred_units = float(np.exp(pred_log_sales))

# Baseline scenario: no coupon, no feature.
baseline_row = input_row.copy()
baseline_row["is_coupon_promotion"] = 0
baseline_row["is_feature_ad"] = 0
baseline_row["has_any_promotion"] = 0
baseline_row["promotion_type"] = "No Promotion"
baseline_df = pd.DataFrame([baseline_row])[feature_cols]
baseline_units = float(np.exp(model.predict(baseline_df)[0]))

lift_pct = (pred_units - baseline_units) / max(baseline_units, 1) * 100
recommendation, reason = make_recommendation(
    pred_units, baseline_units, int(coupon), int(feature), profit_per_unit
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Predicted units sold", format_metric(pred_units))
col2.metric("Baseline units", format_metric(baseline_units))
col3.metric("Predicted lift", f"{lift_pct:.1f}%")
col4.metric("Recommended action", recommendation)

st.subheader("Recommendation Explanation")
st.write(
    f"For **Store {store_id}** and **{brand_name}**, PromoWise recommends **{recommendation}**. "
    f"The selected campaign is **{input_row['promotion_type']}**. {reason}"
)

chart_df = pd.DataFrame(
    {
        "Scenario": ["No Promotion Baseline", input_row["promotion_type"]],
        "Predicted Units": [baseline_units, pred_units],
    }
)
st.bar_chart(chart_df.set_index("Scenario"))

with st.expander("Model performance and selected model"):
    st.write(f"Selected model: **{metadata['model_selected']}**")
    st.dataframe(results, use_container_width=True)
    st.write(
        "The model is trained as an MVP using a chronological split to reduce temporal leakage. "
        "For a final enterprise deployment, retrain on the full data pipeline, add monitoring, and validate recommendations with store-manager feedback."
    )

with st.expander("Input data used for this prediction"):
    st.dataframe(input_df.T.rename(columns={0: "value"}), use_container_width=True)

st.info(
    "Manager-in-the-loop design: PromoWise supports weekly campaign decisions but does not replace store-manager judgement."
)
