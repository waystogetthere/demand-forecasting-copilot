import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")
import shap

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-series features for LightGBM."""
    df = df.copy()
    df = df.sort_values(["item_id", "date"]).reset_index(drop=True)

    grp = df.groupby("item_id")["sales"]

    # Lag features
    for lag in [7, 14, 28]:
        df[f"lag_{lag}"] = grp.shift(lag)

    # Rolling mean features
    for window in [7, 14, 28]:
        df[f"roll_mean_{window}"] = grp.shift(1).rolling(window).mean().reset_index(0, drop=True)

    # Rolling std
    df["roll_std_7"] = grp.shift(1).rolling(7).std().reset_index(0, drop=True)

    # Calendar features
    df["dayofweek"] = df["date"].dt.dayofweek
    df["dayofmonth"] = df["date"].dt.day
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Event features
    df["has_event"] = df["event_name_1"].notna().astype(int)

    # SNAP (food assistance program — relevant for FOODS category)
    df["snap"] = df["snap_CA"].fillna(0)

    # Price feature
    df["sell_price"] = df["sell_price"].fillna(df["sell_price"].median())

    return df


FEATURE_LABELS = {
    "roll_mean_7": "Recent 7-day sales trend",
    "roll_mean_14": "Recent 14-day sales trend",
    "roll_mean_28": "Monthly sales baseline",
    "roll_std_7": "Recent demand volatility",
    "lag_7": "Sales same day last week",
    "lag_14": "Sales same day 2 weeks ago",
    "lag_28": "Sales same day last month",
    "has_event": "Holiday / promotion effect",
    "snap": "Food assistance program day",
    "sell_price": "Product price",
    "is_weekend": "Weekend effect",
    "dayofweek": "Day of week",
    "dayofmonth": "Day of month",
    "weekofyear": "Week of year",
    "month": "Month",
    "year": "Year"
}

FEATURE_COLS = [
    "lag_7", "lag_14", "lag_28",
    "roll_mean_7", "roll_mean_14", "roll_mean_28",
    "roll_std_7",
    "dayofweek", "dayofmonth", "weekofyear", "is_weekend",
    "has_event", "snap", "sell_price",
    "month", "year"
]


def train_forecast(df: pd.DataFrame, horizon: int = 14):
    """
    Train a LightGBM model and generate forecasts.
    Uses last `horizon` days as test set.
    Returns: forecast_df, mae, feature_importance
    """
    df = make_features(df)
    df = df.dropna(subset=FEATURE_COLS)

    # Split: last `horizon` days per item = test
    cutoff = df["date"].max() - pd.Timedelta(days=horizon)
    train = df[df["date"] <= cutoff]
    test = df[df["date"] > cutoff]

    X_train = train[FEATURE_COLS]
    y_train = train["sales"]
    X_test = test[FEATURE_COLS]
    y_test = test["sales"]

    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              callbacks=[])

    preds = model.predict(X_test).clip(0)
    mae = mean_absolute_error(y_test, preds)

    # Build forecast dataframe
    forecast_df = test[["item_id", "date", "sales"]].copy()
    forecast_df["predicted"] = preds.round(2)

    # Feature importance
    importance = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLS
    ).sort_values(ascending=False)

    # return forecast_df, mae, importance
    return forecast_df, mae, importance, model


def summarise_forecast(forecast_df: pd.DataFrame, full_df: pd.DataFrame, horizon: int) -> dict:
    """
    Aggregate forecast stats across all items.
    pct_change compares predicted window vs the equally-long prior window of actuals,
    so it measures demand trend rather than model error.
    """
    summary = {}
    for item_id, grp in forecast_df.groupby("item_id"):
        pred_mean = grp["predicted"].mean()

        forecast_start = grp["date"].min()
        prior_start = forecast_start - pd.Timedelta(days=horizon)
        prior_window = full_df[
            (full_df["item_id"] == item_id) &
            (full_df["date"] >= prior_start) &
            (full_df["date"] < forecast_start)
        ]
        prior_mean = prior_window["sales"].mean() if len(prior_window) > 0 else pred_mean

        pct_change = ((pred_mean - prior_mean) / (prior_mean + 1e-6)) * 100
        pct_change = max(min(pct_change, 999.0), -999.0)

        summary[item_id] = {
            "prior_avg_daily": round(prior_mean, 2),
            "predicted_avg_daily": round(pred_mean, 2),
            "pct_change": round(pct_change, 1),
            "total_predicted": round(grp["predicted"].sum(), 1),
        }
    return summary

def compute_shap(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute SHAP values for the test set.
    Returns a DataFrame with SHAP values per feature per row.
    """
    df_feat = make_features(df)
    df_feat = df_feat.dropna(subset=FEATURE_COLS)

    cutoff = df_feat["date"].max() - pd.Timedelta(days=14)
    test = df_feat[df_feat["date"] > cutoff]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test[FEATURE_COLS])

    shap_df = pd.DataFrame(shap_values, columns=FEATURE_COLS)
    shap_df["item_id"] = test["item_id"].values
    shap_df["date"] = test["date"].values
    return shap_df


def summarise_shap(shap_df: pd.DataFrame) -> dict:
    """
    For each item, return the top 3 features driving its forecast.
    Format ready for LLM consumption.
    """
    result = {}
    for item_id, grp in shap_df.groupby("item_id"):
        mean_shap = grp[FEATURE_COLS].mean()
        top3 = mean_shap.abs().nlargest(3).index.tolist()
        result[item_id] = {
            feat: round(float(mean_shap[feat]), 3)
            for feat in top3
        }
    return result