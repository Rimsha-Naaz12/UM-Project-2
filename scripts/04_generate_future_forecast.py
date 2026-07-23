"""
STEP 4 - Generate genuine future forecasts (beyond the last known date)
Run: python scripts/04_generate_future_forecast.py
Input : data/features_uac_data.csv, models/*.pkl
Output: outputs/future_forecast_<target>.csv  (used by the Streamlit app)
"""
import os
import pickle
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

DATA_PATH = os.path.join("data", "features_uac_data.csv")
CLEAN_PATH = os.path.join("data", "clean_uac_data.csv")
MODEL_DIR = "models"
OUT_DIR = "outputs"
TARGETS = ["hhs_care", "hhs_discharged"]
MAX_HORIZON = 21


def get_ml_feature_cols(df, target):
    cols = [c for c in df.columns if c.startswith(target) and ("lag" in c or "roll" in c)]
    cols += [c for c in df.columns if any(c.startswith(p) for p in
             ["day_of_week", "is_weekend", "month", "day_of_month", "week_of_year", "is_holiday_proxy"])]
    cols += [c for c in df.columns if c.startswith("net_pressure") and ("lag" in c or "roll" in c)]
    return sorted(set(cols))


def recursive_ml_forecast(raw_df, target, model, horizon, feat_cols):
    """Iteratively roll the series forward, rebuilding features at each step."""
    work = raw_df.copy()
    fixed_holidays = {(1, 1), (7, 4), (11, 11), (12, 25), (12, 24), (12, 31), (1, 20)}
    preds = []
    future_dates = pd.date_range(work.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")

    for date in future_dates:
        row = {}
        for col in ["hhs_care", "hhs_discharged", "cbp_transferred_out", "net_pressure"]:
            for lag in [1, 7, 14]:
                idx = date - pd.Timedelta(days=lag)
                row[f"{col}_lag{lag}"] = work[col].get(idx, work[col].iloc[-1])
        for col in ["hhs_care", "hhs_discharged", "net_pressure"]:
            recent = work[col].iloc[-14:]
            row[f"{col}_roll_mean7"] = recent.iloc[-7:].mean()
            row[f"{col}_roll_var7"] = recent.iloc[-7:].var()
            row[f"{col}_roll_mean14"] = recent.mean()
            row[f"{col}_roll_var14"] = recent.var()
        row["day_of_week"] = date.dayofweek
        row["is_weekend"] = int(date.dayofweek in [5, 6])
        row["month"] = date.month
        row["day_of_month"] = date.day
        row["week_of_year"] = int(date.isocalendar().week)
        row["is_holiday_proxy"] = int((date.month, date.day) in fixed_holidays)

        x = pd.DataFrame([row])[feat_cols]
        pred = model.predict(x)[0]
        preds.append(pred)

        # append this prediction to `work` so subsequent lag/rolling features can use it
        new_row = work.iloc[[-1]].copy()
        new_row.index = [date]
        new_row[target] = pred
        work = pd.concat([work, new_row])

    return future_dates, np.array(preds)


def main():
    raw_df = pd.read_csv(CLEAN_PATH, parse_dates=["date"], index_col="date")
    feat_df = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date")

    for target in TARGETS:
        out = pd.DataFrame()
        for algo, prefix in [("Random Forest", "rf"), ("Gradient Boosting", "gbr")]:
            model_path = os.path.join(MODEL_DIR, f"{prefix}_{target}_h1.pkl")
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            feat_cols = get_ml_feature_cols(feat_df, target)
            dates, preds = recursive_ml_forecast(raw_df, target, model, MAX_HORIZON, feat_cols)
            out["date"] = dates
            out[f"forecast_{algo.replace(' ', '_')}"] = preds

        # Exponential Smoothing with an approximate 90% CI band for uncertainty quantification
        ets_model = ExponentialSmoothing(
            raw_df[target], trend="add", seasonal="add", seasonal_periods=7
        ).fit()
        ets_fc = ets_model.forecast(MAX_HORIZON)
        resid_std = np.std(ets_model.resid)
        out["forecast_ETS"] = ets_fc.values
        out["ci_lower_90"] = ets_fc.values - 1.645 * resid_std * np.sqrt(np.arange(1, MAX_HORIZON + 1))
        out["ci_upper_90"] = ets_fc.values + 1.645 * resid_std * np.sqrt(np.arange(1, MAX_HORIZON + 1))

        out.to_csv(os.path.join(OUT_DIR, f"future_forecast_{target}.csv"), index=False)
        print(f"Saved outputs/future_forecast_{target}.csv")
        print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
