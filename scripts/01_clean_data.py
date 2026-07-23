"""
STEP 1 - Data Cleaning & Time-Series Preparation
Run: python scripts/01_clean_data.py
Input : data/raw_uac_data.csv
Output: data/clean_uac_data.csv
"""
import pandas as pd
import numpy as np
import os

RAW_PATH = os.path.join("data", "raw_uac_data.csv")
CLEAN_PATH = os.path.join("data", "clean_uac_data.csv")

COLMAP = {
    "Date": "date",
    "Children apprehended and placed in CBP custody*": "cbp_intake",
    "Children in CBP custody": "cbp_active",
    "Children transferred out of CBP custody": "cbp_transferred_out",
    "Children in HHS Care": "hhs_care",
    "Children discharged from HHS Care": "hhs_discharged",
}


def load_raw(path=RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=COLMAP)
    df = df.dropna(subset=["date"])  # drop trailing blank rows
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # strip thousands-separator commas from numeric-looking string columns, coerce to numeric
    num_cols = ["cbp_intake", "cbp_active", "cbp_transferred_out", "hhs_care", "hhs_discharged"]
    for c in num_cols:
        df[c] = (
            df[c].astype(str).str.replace(",", "", regex=False).str.strip()
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # sort ascending by date, drop exact duplicate dates (keep last reported)
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last")
    df = df.set_index("date")

    # reindex to a continuous daily calendar so gaps are explicit
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_range)
    df.index.name = "date"

    # the source only reports on business days -> interpolate the numeric series
    # (time-weighted interpolation handles the uneven gaps correctly)
    for c in num_cols:
        df[c] = df[c].interpolate(method="time", limit_direction="both")

    # flow-based pressure indicator required by the project brief
    df["net_pressure"] = df["cbp_transferred_out"] - df["hhs_discharged"]

    # round intake/flow columns back to whole children after interpolation
    for c in num_cols + ["net_pressure"]:
        df[c] = df[c].round(0)

    return df


def main():
    df = load_raw()
    clean_df = clean(df)
    clean_df.to_csv(CLEAN_PATH)
    print(f"Saved cleaned data: {CLEAN_PATH}")
    print(clean_df.describe())
    print("\nDate range:", clean_df.index.min().date(), "->", clean_df.index.max().date())
    print("Rows:", len(clean_df))


if __name__ == "__main__":
    main()
