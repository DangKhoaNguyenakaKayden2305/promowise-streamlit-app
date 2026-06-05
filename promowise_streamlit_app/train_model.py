import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor


DATA_PATH = Path("data/sunshine_grocery_powerbi_ready.csv")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

# Predict log-transformed sales to stabilize variance across high/low-volume SKUs.
TARGET = "log_sales"
# Drop the raw sales columns (would leak the target), a zero-variance constant, and an ID that adds no signal.
DROP_COLS = ["log_sales", "units_sold", "constant", "brand_id"]


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main(sample_train=12000, sample_eval=5000):
    df = pd.read_csv(DATA_PATH)

    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    cat_cols = ["store_id", "promotion_type", "brand_name", "income_segment"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    # Chronological split to avoid temporal leakage.
    train_full = df[df["week_id"] <= 112].copy()
    val_full = df[(df["week_id"] >= 113) & (df["week_id"] <= 142)].copy()
    test_full = df[df["week_id"] >= 143].copy()

    # Sampled training keeps the MVP fast. Increase or remove sampling for final training.
    train = train_full.sample(n=min(sample_train, len(train_full)), random_state=42)
    val = val_full.sample(n=min(sample_eval, len(val_full)), random_state=42)
    test = test_full.sample(n=min(sample_eval, len(test_full)), random_state=42)

    preprocess = ColumnTransformer(
        transformers=[
            # Median is robust to outliers in skewed sales/price features.
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                # Map unseen stores/brands at inference to -1 instead of raising an error.
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    models = {
        "Decision Tree Baseline": DecisionTreeRegressor(
            max_depth=10, random_state=42, min_samples_leaf=20
        ),
        "LightGBM Gradient Boosting": LGBMRegressor(
            n_estimators=45,
            learning_rate=0.08,
            max_depth=5,
            num_leaves=20,
            random_state=42,
            n_jobs=2,
            verbosity=-1,
        ),
    }

    results = []
    pipelines = {}

    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
        pipe.fit(train[feature_cols], train[TARGET])
        pred = pipe.predict(val[feature_cols])

        results.append(
            {
                "model": name,
                "MAE": float(mean_absolute_error(val[TARGET], pred)),
                "RMSE": rmse(val[TARGET], pred),
                "R2": float(r2_score(val[TARGET], pred)),
            }
        )
        pipelines[name] = pipe

    # R2 is used as the selection criterion: highest explained variance wins.
    best_name = max(results, key=lambda row: row["R2"])["model"]
    best_pipe = pipelines[best_name]

    pred_test = best_pipe.predict(test[feature_cols])
    test_metrics = {
        "MAE": float(mean_absolute_error(test[TARGET], pred_test)),
        "RMSE": rmse(test[TARGET], pred_test),
        "R2": float(r2_score(test[TARGET], pred_test)),
    }

    joblib.dump(best_pipe, MODEL_DIR / "promowise_demand_model.pkl")

    metadata = {
        "target": TARGET,
        "model_selected": best_name,
        "validation_results": results,
        "test_metrics_sampled": test_metrics,
        "feature_columns": feature_cols,
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "business_use": "Predict OJ demand and convert the prediction into a weekly campaign recommendation.",
    }

    with open(MODEL_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Build a per-store/brand snapshot using the most recent week as the inference baseline.
    reference = (
        df.sort_values("week_id")
        .groupby(["store_id", "brand_name"], as_index=False)
        .tail(1)[feature_cols + ["units_sold", "log_sales"]]
        .reset_index(drop=True)
    )
    reference.to_csv("data/reference_data.csv", index=False)
    pd.DataFrame(results).to_csv("data/model_validation_results.csv", index=False)

    print("Training complete.")
    print("Best model:", best_name)
    print("Validation results:", results)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
