"""
STEP 3 - Forecasting Models + Walk-Forward Evaluation
Run: python scripts/03_train_models.py
Input : data/features_uac_data.csv
Output: outputs/model_metrics.csv, outputs/forecasts_<target>.csv, models/*.pkl
"""
import os
import warnings
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

FEATURES_PATH = os.path.join("data", "features_uac_data.csv")
OUT_DIR = "outputs"
MODEL_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TARGETS = ["hhs_care", "hhs_discharged"]
HORIZONS = [1, 7, 14]           # days ahead evaluated
TEST_SIZE = 45                  # last 60 days held out, walk-forward
ML_FEATURE_PREFIXES = ("lag", "roll", "day_of_week", "is_weekend", "month",
                        "day_of_month", "week_of_year", "is_holiday_proxy")


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def get_ml_feature_cols(df, target):
    cols = [c for c in df.columns if c.startswith(target) and "lag" in c or c.startswith(target) and "roll" in c]
    cols += [c for c in df.columns if any(c.startswith(p) for p in
             ["day_of_week", "is_weekend", "month", "day_of_month", "week_of_year", "is_holiday_proxy"])]
    cols += [c for c in df.columns if c.startswith("net_pressure") and ("lag" in c or "roll" in c)]
    return sorted(set(cols))


def walk_forward_ml(df, target, model_builder, horizon):
    """Expanding-window walk-forward validation for an ML/baseline model at a fixed horizon."""
    feature_cols = get_ml_feature_cols(df, target)
    y = df[target].shift(-horizon + 1) if horizon > 1 else df[target]
    # shift target forward so X at time t predicts y at time t+horizon-1 -> simpler: predict target 'horizon' steps ahead
    data = df.copy()
    data["y_target"] = data[target].shift(-horizon)
    data = data.dropna(subset=["y_target"])

    n = len(data)
    test_start = n - TEST_SIZE
    preds, actuals = [], []
    model = None
    REFIT_EVERY = 9  # expanding-window refit cadence (keeps walk-forward eval fast)
    for step, i in enumerate(range(test_start, n)):
        if model is None or step % REFIT_EVERY == 0:
            train = data.iloc[:i]
            model = model_builder()
            model.fit(train[feature_cols], train["y_target"])
        test_row = data.iloc[[i]]
        pred = model.predict(test_row[feature_cols])[0]
        preds.append(pred)
        actuals.append(test_row["y_target"].values[0])
    return np.array(actuals), np.array(preds), model  # last fitted model returned for saving


def walk_forward_naive(df, target, horizon):
    data = df.copy()
    data["y_target"] = data[target].shift(-horizon)
    data["y_pred_naive"] = data[target]  # persistence: predict last known value
    data = data.dropna(subset=["y_target"])
    test = data.iloc[-TEST_SIZE:]
    return test["y_target"].values, test["y_pred_naive"].values


def walk_forward_moving_avg(df, target, horizon, window=7):
    data = df.copy()
    data["y_target"] = data[target].shift(-horizon)
    data["y_pred_ma"] = data[target].rolling(window).mean()
    data = data.dropna(subset=["y_target", "y_pred_ma"])
    test = data.iloc[-TEST_SIZE:]
    return test["y_target"].values, test["y_pred_ma"].values


def walk_forward_statsmodel(df, target, horizon, kind="arima"):
    """Rolling-origin evaluation for ARIMA / Exponential Smoothing (refit every 7 steps to save time)."""
    series = df[target]
    n = len(series)
    test_start = n - TEST_SIZE
    actuals, preds = [], []
    model_fit = None
    for step, i in enumerate(range(test_start, n - horizon + 1)):
        train = series.iloc[:i]
        if step % 9 == 0 or model_fit is None:
            try:
                if kind == "arima":
                    model_fit = ARIMA(train, order=(2, 1, 2)).fit()
                else:
                    model_fit = ExponentialSmoothing(
                        train, trend="add", seasonal="add", seasonal_periods=7
                    ).fit()
            except Exception:
                continue
        try:
            fc = model_fit.forecast(horizon)
            pred = fc.iloc[horizon - 1] if hasattr(fc, "iloc") else fc[horizon - 1]
        except Exception:
            continue
        actual = series.iloc[i + horizon - 1]
        actuals.append(actual)
        preds.append(pred)
    return np.array(actuals), np.array(preds), model_fit


def score(actuals, preds):
    return {
        "MAE": mean_absolute_error(actuals, preds),
        "RMSE": np.sqrt(mean_squared_error(actuals, preds)),
        "MAPE": mape(actuals, preds),
    }


def main():
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"], index_col="date")
    results = []
    forecast_store = {}

    for target in TARGETS:
        forecast_store[target] = {}
        for horizon in HORIZONS:
            # Naive persistence
            a, p = walk_forward_naive(df, target, horizon)
            results.append({"target": target, "model": "Naive Persistence", "horizon": horizon, **score(a, p)})

            # Moving average
            a, p = walk_forward_moving_avg(df, target, horizon)
            results.append({"target": target, "model": "Moving Average (7d)", "horizon": horizon, **score(a, p)})

            # ARIMA
            a, p, arima_model = walk_forward_statsmodel(df, target, horizon, kind="arima")
            if len(a) > 0:
                results.append({"target": target, "model": "ARIMA(2,1,2)", "horizon": horizon, **score(a, p)})

            # Exponential Smoothing (Holt-Winters, weekly seasonality)
            a, p, ets_model = walk_forward_statsmodel(df, target, horizon, kind="ets")
            if len(a) > 0:
                results.append({"target": target, "model": "Exponential Smoothing", "horizon": horizon, **score(a, p)})

            # Random Forest
            a, p, rf_model = walk_forward_ml(
                df, target, lambda: RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42), horizon
            )
            results.append({"target": target, "model": "Random Forest", "horizon": horizon, **score(a, p)})
            if horizon == 1:
                forecast_store[target]["Random Forest"] = (a, p)
                with open(os.path.join(MODEL_DIR, f"rf_{target}_h{horizon}.pkl"), "wb") as f:
                    pickle.dump(rf_model, f)

            # Gradient Boosting
            a, p, gbr_model = walk_forward_ml(
                df, target, lambda: GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                                                                random_state=42), horizon
            )
            results.append({"target": target, "model": "Gradient Boosting", "horizon": horizon, **score(a, p)})
            if horizon == 1:
                forecast_store[target]["Gradient Boosting"] = (a, p)
                with open(os.path.join(MODEL_DIR, f"gbr_{target}_h{horizon}.pkl"), "wb") as f:
                    pickle.dump(gbr_model, f)

            print(f"Done: target={target}, horizon={horizon}")

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(os.path.join(OUT_DIR, "model_metrics.csv"), index=False)
    print("\nSaved metrics -> outputs/model_metrics.csv")
    print(metrics_df.sort_values(["target", "horizon", "MAE"]).to_string(index=False))

    # save 1-day-ahead actual vs predicted series for the dashboard
    dates_test = df.index[-TEST_SIZE:]
    for target, models in forecast_store.items():
        out = pd.DataFrame({"date": dates_test})
        for model_name, (a, p) in models.items():
            out[f"actual"] = a
            out[f"pred_{model_name.replace(' ', '_')}"] = p
        out.to_csv(os.path.join(OUT_DIR, f"forecast_{target}.csv"), index=False)


if __name__ == "__main__":
    main()
