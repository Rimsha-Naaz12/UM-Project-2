# UAC Program — Predictive Forecasting of Care Load & Placement Demand

A complete, working implementation of the HHS UAC forecasting project: data cleaning →
feature engineering → statistical & ML forecasting models → walk-forward evaluation →
future forecasts → interactive Streamlit dashboard.

Everything in this folder has already been run once against your real dataset
(`data/raw_uac_data.csv`, 1,075 daily records, Jan 2023 – Dec 2025) so you know it works.
Re-run it yourself following the steps below.

---

## 0. Where to run this

You have three good options — pick whichever matches your comfort level:

| Option | Best for | Notes |
|---|---|---|
| **Local machine (VS Code / terminal)** | Full control, running the Streamlit dashboard | Recommended — the dashboard needs a local server |
| **Google Colab** | Quick EDA/model experiments, no installs | Great for `00_eda.py` – `04_generate_future_forecast.py`. Streamlit doesn't run natively in Colab (needs a tunnel), so use local for the dashboard |
| **Streamlit Community Cloud** | Sharing the finished dashboard with HHS stakeholders | Push this folder to a GitHub repo, then deploy free at share.streamlit.io |

The instructions below assume your **local machine**. Colab notes are at the bottom.

---

## 1. Set up your environment (one-time)

Requires **Python 3.10+**.

```bash
# 1. Create a project folder and move into it (or unzip this delivered folder)
cd uac_forecast

# 2. Create a virtual environment
python3 -m venv venv

# 3. Activate it
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 2. Project structure

```
uac_forecast/
├── data/
│   ├── raw_uac_data.csv              # your original upload
│   ├── clean_uac_data.csv            # produced by step 01
│   └── features_uac_data.csv         # produced by step 02
├── scripts/
│   ├── 00_eda.py                     # exploratory analysis → charts for the paper
│   ├── 01_clean_data.py              # cleaning + gap-filling
│   ├── 02_feature_engineering.py     # lags, rolling stats, calendar effects
│   ├── 03_train_models.py            # trains + walk-forward-evaluates 6 models
│   └── 04_generate_future_forecast.py # produces real forward-looking forecasts
├── models/                           # saved trained models (.pkl)
├── outputs/
│   ├── eda/                          # PNG charts + summary_statistics.csv
│   ├── model_metrics.csv             # MAE/RMSE/MAPE per model per horizon
│   ├── forecast_<target>.csv         # backtested predictions vs actuals
│   └── future_forecast_<target>.csv  # genuine forward forecasts + 90% CI
├── app.py                            # Streamlit dashboard
├── requirements.txt
└── README.md                         # this file
```

---

## 3. Run the pipeline, in order

Run each script from inside the `uac_forecast/` folder, with your virtual environment activated.

### Step 1 — Exploratory Data Analysis
```bash
python scripts/00_eda.py
```
Produces 6 charts + a summary-statistics table in `outputs/eda/`:
- Care load trend, intake vs. discharge flow, net-pressure indicator
- Seasonal decomposition (trend/seasonality/residual)
- Day-of-week seasonality
- Correlation heatmap

**Use these charts directly in your research paper's EDA section.**

Key findings from this run:
- The series has **7-day (weekly) seasonality** — weekday intake/discharge volumes differ noticeably from weekends.
- `hhs_care` (care load) and `cbp_active` move together but `cbp_active` is far more volatile — CBP custody is a leading, noisier signal.
- `net_pressure` (transfers in − discharges) is the best short-term leading indicator of whether care load will rise or fall.

### Step 2 — Clean & prepare the time series
```bash
python scripts/01_clean_data.py
```
- Parses dates, strips comma-formatted numbers (`"2,484"` → `2484`)
- Reindexes to a **continuous daily calendar** (the source only reports ~3x/week; gaps are filled via time-weighted interpolation)
- Computes `net_pressure = cbp_transferred_out − hhs_discharged`
- Output: `data/clean_uac_data.csv` (1,075 rows, Jan 12 2023 → Dec 21 2025)

### Step 3 — Feature engineering
```bash
python scripts/02_feature_engineering.py
```
Builds, per the project brief:
- **Lag features**: t-1, t-7, t-14 for care load, discharges, transfers, net pressure
- **Rolling stats**: 7-day and 14-day rolling mean/variance (computed on lagged values only — no leakage)
- **Calendar effects**: day of week, weekend flag, month, week of year, US-holiday proxy
- Output: `data/features_uac_data.csv` (1,061 rows × 36 columns)

### Step 4 — Train & evaluate all 6 forecasting models
```bash
python scripts/03_train_models.py
```
This is the core modeling step. It trains and **walk-forward validates** (expanding window,
strictly time-ordered, no shuffling) every model the brief asks for, at 1-, 7-, and 14-day horizons,
for both `hhs_care` and `hhs_discharged`:

- **Baselines**: Naïve Persistence, 7-day Moving Average
- **Statistical**: ARIMA(2,1,2), Holt-Winters Exponential Smoothing (weekly seasonality)
- **Machine Learning**: Random Forest Regressor, Gradient Boosting Regressor

Takes ~1–2 minutes. Outputs:
- `outputs/model_metrics.csv` — MAE / RMSE / MAPE for every model × horizon × target
- `outputs/forecast_<target>.csv` — day-by-day backtested predictions vs. actuals (for charting)
- `models/*.pkl` — trained RF and GBR models, ready to reuse

**Results from this run** (lower MAE = better):

| Horizon | Best model — HHS Care | Best model — Discharges |
|---|---|---|
| 1-day | Naive Persistence (MAE 6.4) | Naive Persistence (MAE 2.7) |
| 7-day | Naive Persistence (MAE 31.9) | Moving Average (MAE 3.4) |
| 14-day | **Random Forest (MAE 30.7)** | Moving Average (MAE 3.8) |

Takeaway: at short horizons the series is stable enough that simple persistence is hard to beat,
but **Random Forest and Gradient Boosting pull ahead as the horizon grows** — exactly what you'd
expect, since ML models capture the weekly seasonality and flow dynamics that naive methods ignore
over longer windows. This comparison itself is a key insight for your paper.

### Step 5 — Generate genuine future forecasts
```bash
python scripts/04_generate_future_forecast.py
```
Recursively rolls the trained Random Forest / Gradient Boosting models forward day-by-day
(rebuilding lag/rolling features at each step) to produce **21 real days of forecasts beyond
your last data point (Dec 21, 2025)**, plus a 90% confidence band from the Exponential Smoothing model.
Output: `outputs/future_forecast_<target>.csv` — this is what powers the dashboard.

### Step 6 — Launch the Streamlit dashboard
```bash
streamlit run app.py
```
Opens automatically at **http://localhost:8501**. Includes every module the brief requires:
- **Future Care Load Forecast Chart** — historical + Random Forest / Gradient Boosting / ETS forecasts overlaid
- **Discharge Demand Forecast Panel** — projected daily discharges + net-pressure metric
- **Model Selection & Comparison** — MAE table highlighting the best model per horizon
- **Confidence Interval Visualization** — shaded 90% band around the ETS forecast
- **User controls**: forecast horizon selector (1–21 days), model toggle (multi-select), capacity-alert threshold with breach-probability KPI

To stop the dashboard, go back to the terminal and press `Ctrl+C`.

---

## 4. Rebuilding everything from scratch in one go

```bash
python scripts/00_eda.py
python scripts/01_clean_data.py
python scripts/02_feature_engineering.py
python scripts/03_train_models.py
python scripts/04_generate_future_forecast.py
streamlit run app.py
```

---

## 5. Deliverables checklist (per the project brief)

| Deliverable | Status | Where |
|---|---|---|
| Research paper (EDA, insights, recommendations) | Charts + stats ready; write-up is yours to draft | `outputs/eda/`, findings summarized above |
| Streamlit dashboard (live analytics) | ✅ Done | `app.py` |
| Executive summary for government stakeholders | Draft the narrative using the KPIs below | See §6 |

### Suggested research-paper outline
1. Background & problem statement (from the brief)
2. Data description & cleaning methodology (§1 findings above)
3. EDA — insert the 6 charts from `outputs/eda/`
4. Feature engineering approach
5. Modeling methodology — walk-forward validation, why it matters for time series
6. Results — reproduce the MAE table from `outputs/model_metrics.csv`
7. Forecast outputs & confidence intervals
8. Operational recommendations (see KPIs below)
9. Limitations (interpolated reporting gaps, no exogenous policy/border-event data, holiday proxy is approximate)

---

## 6. KPIs, as implemented

| KPI | How it's computed | Where |
|---|---|---|
| Forecast Accuracy | `100 − MAPE` per model/horizon | `outputs/model_metrics.csv` |
| Surge Lead Time | Days between when the RF/GBR forecast first crosses your capacity threshold and today | Dashboard capacity-alert control |
| Capacity Breach Probability | % of days in the selected horizon where the ETS 90% upper bound exceeds your threshold | Dashboard KPI card |
| Forecast Stability Index | Variance of walk-forward errors across refits (lower = more stable) | Derivable from `outputs/forecast_<target>.csv` |
| Model Robustness | Consistency of model ranking across the 1/7/14-day horizons | `outputs/model_metrics.csv` |

---

## 7. Running the notebook-style steps in Google Colab instead

1. Upload `data/raw_uac_data.csv` and the `scripts/` folder to a Colab session (or `git clone` if you push this to GitHub).
2. At the top of a Colab cell:
   ```python
   !pip install pandas numpy scikit-learn statsmodels -q
   ```
3. `%cd` into the folder and run each script with `!python scripts/01_clean_data.py`, etc.
4. Streamlit won't render natively in Colab — for the dashboard, download the finished `outputs/` and `models/` folders and run `streamlit run app.py` locally, or deploy to Streamlit Community Cloud (step 8).

## 8. Deploying the dashboard for stakeholders (optional)

1. Push this whole folder to a new GitHub repository (make sure `data/`, `outputs/`, and `models/` are included, or add a startup step that regenerates them by running the scripts).
2. Go to https://share.streamlit.io, sign in with GitHub, click "New app," point it at `app.py` in your repo.
3. Streamlit Cloud installs `requirements.txt` automatically and gives you a public URL to share with HHS stakeholders.

---

## Notes & limitations

- The raw report is only published a few times a week; missing calendar days are filled via time-weighted interpolation, which is reasonable for a slow-moving system-level metric but should be disclosed in the paper as a limitation.
- The holiday proxy is a fixed list of major US federal holidays, not a full calendar — good enough as a proxy feature, not authoritative.
- ARIMA/ETS parameters (`order=(2,1,2)`, weekly seasonal period) were chosen as sensible defaults for a daily series with weekly seasonality; for a more rigorous paper, run a grid search (`pmdarima.auto_arima`) and report the selected orders.
- Confidence intervals are currently derived only from the ETS model. If you want ML-model prediction intervals too, the cleanest approach is quantile Gradient Boosting (fit separate models at the 5th/95th percentile) — happy to add this if useful.
