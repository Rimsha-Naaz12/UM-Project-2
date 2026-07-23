"""
STEP 2 - Feature Engineering for Forecasting
Run: python scripts/02_feature_engineering.py
Input : data/clean_uac_data.csv
Output: data/features_uac_data.csv
"""
import pandas as pd
import os

CLEAN_PATH = os.path.join("data", "clean_uac_data.csv")
FEATURES_PATH = os.path.join("data", "features_uac_data.csv")

TARGETS = ["hhs_care", "hhs_discharged"]
LAGS = [1, 7, 14]
ROLL_WINDOWS = [7, 14]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- lag features ---
    for col in ["hhs_care", "hhs_discharged", "cbp_transferred_out", "net_pressure"]:
        for lag in LAGS:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # --- rolling mean / variance (computed on lag-1 to avoid leakage) ---
    for col in ["hhs_care", "hhs_discharged", "net_pressure"]:
        shifted = df[col].shift(1)
        for w in ROLL_WINDOWS:
            df[f"{col}_roll_mean{w}"] = shifted.rolling(w).mean()
            df[f"{col}_roll_var{w}"] = shifted.rolling(w).var()

    # --- calendar effects ---
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = df.index.month
    df["day_of_month"] = df.index.day
    df["week_of_year"] = df.index.isocalendar().week.astype(int)

    # US federal holiday proxy (no external dependency)
    fixed_holidays = {(1, 1), (7, 4), (11, 11), (12, 25), (12, 24), (12, 31), (1, 20)}
    df["is_holiday_proxy"] = df.index.map(lambda d: int((d.month, d.day) in fixed_holidays))

    df = df.dropna()
    return df


def main():
    df = pd.read_csv(CLEAN_PATH, parse_dates=["date"], index_col="date")
    feat_df = build_features(df)
    feat_df.to_csv(FEATURES_PATH)
    print(f"Saved feature set: {FEATURES_PATH}")
    print("Shape:", feat_df.shape)
    print("Columns:", list(feat_df.columns))


if __name__ == "__main__":
    main()
