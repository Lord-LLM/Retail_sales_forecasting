# Retail Sales Forecasting

A Streamlit dashboard that downloads the monthly U.S. retail-sales series (FRED `RSXFS`) and compares **SARIMA**, **Prophet**, and **LSTM** forecasts side by side — backtested against held-out data and projected forward.

**🔗 Live app:** [retailsalesforecasting.streamlit.app](https://retailsalesforecasting-7hhpf64vbawbngb9dpmvpv.streamlit.app/)

---

## Features

- **Data** — pulls any date range of `RSXFS` (Advance Retail Sales: Retail Trade) directly from FRED.
- **Preprocessing** — builds lag (1/3/12-month), rolling mean/std, and calendar features.
- **Models** — trains SARIMA, Prophet, and/or LSTM on demand, with tunable LSTM settings (units, epochs, lookback window).
- **Backtesting** — holds out the last *N* months and scores each model with MAE, RMSE, and MAPE.
- **Future forecasting** — retrains on the full series to project beyond the latest available month, with confidence intervals and a CSV download.
- **Diagnostics** — time series plot, STL decomposition, ADF stationarity test, ACF/PACF, and a month-by-year seasonality heatmap.

## Models Used

| Model | Type | Notes |
|---|---|---|
| **SARIMA** | Classical statistical | Captures trend + seasonality via `(p,d,q)(P,D,Q,s)` order |
| **Prophet** | Decomposable time series | Trend + yearly seasonality (weekly/daily seasonality disabled — the data is monthly) |
| **LSTM** | Deep learning | Captures nonlinear patterns; needs more data and tuning than the other two |

## Evaluation Metrics

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **MAPE** — Mean Absolute Percentage Error

---

## Getting Started

### Clone and set up an environment

**Windows (PowerShell)**

```powershell
git clone https://github.com/poip-boop/retail-sales-forecasting.git
cd retail-sales-forecasting
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux**

```bash
git clone https://github.com/poip-boop/retail-sales-forecasting.git
cd retail-sales-forecasting
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### FRED API key

The app needs a free FRED API key to download the retail-sales data.

1. Create one at <https://fred.stlouisfed.org/docs/api/api_key.html>.
2. Provide it one of two ways:

   **Environment variable**

   ```powershell
   $env:FRED_API_KEY = "your-key"        # PowerShell
   ```

   ```bash
   export FRED_API_KEY="your-key"        # macOS/Linux
   ```

   **Streamlit secrets file**

   Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and replace the placeholder:

   ```toml
   FRED_API_KEY = "your-key"
   ```

   `.streamlit/secrets.toml` is ignored by Git, so your key never gets committed.

### Run

```powershell
streamlit run Retail_sales_time_series_forecasting.py
```

The app opens at `http://localhost:8501`. Use the sidebar to set the date range, forecast horizon, models to run, and LSTM hyperparameters.

---

## Project Structure

```
retail-sales-forecasting/
├── Retail_sales_time_series_forecasting.py   # Streamlit app
├── requirements.txt
├── .streamlit/
│   └── secrets.toml.example
└── README.md
```

## Deployment

The live demo runs on [Streamlit Community Cloud](https://streamlit.io/cloud). To deploy your own copy:

1. Push the repo to GitHub.
2. Create a new app on Streamlit Cloud pointing at `Retail_sales_time_series_forecasting.py`.
3. Add `FRED_API_KEY` under **App settings → Secrets**.

## Troubleshooting

| Issue | Fix |
|---|---|
| `FRED_API_KEY environment variable not set` | Set the env var or secrets file as described above, then restart the app. |
| App is slow on first load | Prophet/TensorFlow imports and the first model fit take longest; subsequent reruns are cached. |
| LSTM error about insufficient data | Widen the date range or shrink the forecast horizon / lookback window in the sidebar. |

## License

Add your preferred license (e.g. MIT) here.
