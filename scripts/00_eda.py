"""
STEP 0 - Exploratory Data Analysis
Run: python scripts/00_eda.py
Input : data/clean_uac_data.csv
Output: outputs/eda/*.png  (use these charts directly in the research paper)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

CLEAN_PATH = os.path.join("data", "clean_uac_data.csv")
EDA_DIR = os.path.join("outputs", "eda")
os.makedirs(EDA_DIR, exist_ok=True)


def main():
    df = pd.read_csv(CLEAN_PATH, parse_dates=["date"], index_col="date")

    # 1. Trend of core series
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["hhs_care"], label="Children in HHS Care")
    plt.plot(df.index, df["cbp_active"], label="Children in CBP Custody")
    plt.legend(); plt.title("Care Load Over Time"); plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "01_care_load_trend.png")); plt.close()

    # 2. Intake vs discharge flow
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["cbp_transferred_out"], label="Transferred into HHS", alpha=0.7)
    plt.plot(df.index, df["hhs_discharged"], label="Discharged from HHS", alpha=0.7)
    plt.legend(); plt.title("Daily Flow: Transfers In vs Discharges Out"); plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "02_flow_comparison.png")); plt.close()

    # 3. Net pressure indicator
    plt.figure(figsize=(12, 4))
    plt.bar(df.index, df["net_pressure"], color=(df["net_pressure"] > 0).map({True: "crimson", False: "seagreen"}))
    plt.title("Net System Pressure (Transfers In − Discharges Out)"); plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "03_net_pressure.png")); plt.close()

    # 4. Seasonal decomposition of HHS care load
    decomp = seasonal_decompose(df["hhs_care"], model="additive", period=7)
    fig = decomp.plot(); fig.set_size_inches(12, 8); plt.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "04_seasonal_decomposition.png")); plt.close()

    # 5. Day-of-week seasonality
    dow = df.copy(); dow["dow"] = dow.index.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    means = dow.groupby("dow")[["cbp_intake", "hhs_discharged"]].mean().reindex(order)
    means.plot(kind="bar", figsize=(10, 5)); plt.title("Average Intake / Discharge by Day of Week")
    plt.tight_layout(); plt.savefig(os.path.join(EDA_DIR, "05_day_of_week.png")); plt.close()

    # 6. Correlation heatmap
    import numpy as np
    corr = df[["cbp_intake", "cbp_active", "cbp_transferred_out", "hhs_care", "hhs_discharged", "net_pressure"]].corr()
    plt.figure(figsize=(7, 6))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(); plt.xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.title("Correlation Matrix"); plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "06_correlation_heatmap.png")); plt.close()

    # Summary stats table -> csv for the paper
    df.describe().to_csv(os.path.join(EDA_DIR, "summary_statistics.csv"))

    print("EDA charts saved to outputs/eda/")
    for f in sorted(os.listdir(EDA_DIR)):
        print(" -", f)


if __name__ == "__main__":
    main()
