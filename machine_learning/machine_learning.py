"""
machine_learning.py
===================
Load and use the already-trained Random Forest model for load forecasting.

The heaviest step - TRAINING - is already done: a model was trained elsewhere
(see the cyrus1 notebook / book2 pipeline) and saved to disk. This file only
*loads* that saved model and uses it to make predictions.

The full training code is included below but COMMENTED OUT, as a reference for
how the model was built - in case you ever want to retrain or understand it.
"""
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
# The notebook saves the model DIRECTLY (not wrapped in a metadata dict),
# so we load it and use it as-is. Because it is large (~1 GB), only load
# when you actually need a prediction.
MODEL_PATH   = Path(__file__).resolve().parents[1] / "models" / "best_rf_clean.joblib"
FEAT_COL_PATH = Path(__file__).resolve().parents[1] / "models" / "feature_cols_clean.joblib"

TARGET_COL = "MW"  # the load in Megawatts - what the model predicts


# ---------------------------------------------------------------------------
# ACTIVE: load the saved model + its feature list
# ---------------------------------------------------------------------------
def load_model():
    """Load the trained RandomForestRegressor from disk.

    Returns the model object directly (not a dict) - joblib.load returns the
    exact object that was saved.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def load_feature_cols():
    """Load the exact column names the model was trained on.

    Order matters: predictions must use columns in the SAME order.
    """
    if not FEAT_COL_PATH.exists():
        raise FileNotFoundError(f"Feature columns file not found: {FEAT_COL_PATH}")
    return joblib.load(FEAT_COL_PATH)


def predict(model, features: pd.DataFrame) -> np.ndarray:
    """Predict load (MW) for rows of pre-built features.

    features: DataFrame with exactly the columns in feature_cols_clean, in order.
    Returns:  array of predicted MW values.
    """
    features = features[load_feature_cols()]  # enforce correct column order
    return model.predict(features)


def predict_single(input_data: dict) -> float:
    """Predict the load for ONE hour given as a dict of features.

    Example:
        row = {"hour": 14, "day_of_week": 2, ..., "load_lag_1h": 11935.0}
        mw = predict_single(row)
    """
    model = load_model()
    df = pd.DataFrame([input_data])
    return float(predict(model, df)[0])


def batch_predict(features_df: pd.DataFrame) -> pd.Series:
    """Convenience wrapper: full model load + predict on a DataFrame of rows."""
    model = load_model()
    preds = predict(model, features_df)
    return pd.Series(preds, name="Predicted_MW")


# ---------------------------------------------------------------------------
# COMMENTED OUT - TRAINING (already done; kept for reference)
# ---------------------------------------------------------------------------
# The model is ALREADY trained and saved. You do NOT need to run any of this.
# It is here only to document how the RF was built (RandomForestRegressor,
# regression, no one-hot pipeline needed - all features are numeric).
#
# from sklearn.ensemble import RandomForestRegressor
# from scripts.clean import CLEAN_PATH
# from sklearn.model_selection import TimeSeriesSplit
#
# def get_X_y(df: pd.DataFrame):
#     # x = feature columns only; y = the load to predict (MW).
#     feat_cols = load_feature_cols()
#     X = df[feat_cols]
#     y = df[TARGET_COL]
#     return X, y
#
# def train_model(df: pd.DataFrame):
#     # Split by time (not random) - load forecasting must not leak the future.
#     X, y = get_X_y(df)
#     split_idx = int(len(X) * 0.8)
#     X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
#     y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
#
#     # Random Forest - a robust regressor, good on tabular data out of the box.
#     model = RandomForestRegressor(
#         n_estimators=120,
#         random_state=42,
#         n_jobs=-1,
#     )
#     model.fit(X_train, y_train)
#
#     from sklearn.metrics import mean_absolute_error
#     mae = mean_absolute_error(y_test, model.predict(X_test))
#     print(f"Test MAE: {mae:.2f} MW")
#
#     MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
#     joblib.dump(model, MODEL_PATH)
#     joblib.dump(list(X.columns), FEAT_COL_PATH)
#     print(f"Model saved to {MODEL_PATH}")
#     return model, X_test, y_test
#
#
# if __name__ == "__main__":
#     # Uncomment only if you intentionally want to retrain.
#     # df = pd.read_csv(CLEAN_PATH)
#     # train_model(df)
#     pass


# ---------------------------------------------------------------------------
# ACTIVE: quick demo when run as a script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = load_model()
    feat_cols = load_feature_cols()
    print(f"Loaded model: RandomForestRegressor")
    print(f"Expected feature count: {len(feat_cols)}")

    # Build ONE example row with the right columns, fill lags with real-ish values
    # so you can see a prediction come out. NaN lags would pollute the result.
    example = {c: 0.0 for c in feat_cols}
    example.update({
        "hour": 14, "day_of_week": 2, "month": 6, "year": 2018,
        "is_weekend": 0, "MW": 12000.0,
    })
    df_demo = pd.DataFrame([example])
    pred = predict(model, df_demo)[0]
    print(f"Demo prediction: {pred:.2f} MW")
