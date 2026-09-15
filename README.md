# Retail Sales Forecasting

This Streamlit dashboard compares **SARIMA**, **Prophet**, and **LSTM** forecasts for the monthly FRED retail-sales series `RSXFS`.

## Overview

The dashboard:

- Downloads the selected date range from FRED.
- Preprocesses the monthly series and creates lag and rolling features.
- Trains the selected forecasting models.
- Compares backtest results using **MAE**, **RMSE**, and **MAPE**.
- Displays forecasts, confidence intervals, ACF/PACF, and STL decomposition.

## Models Used

### SARIMA

Classical statistical model for seasonality and trend.

### Prophet

Decomposable time-series model for trend and yearly seasonality.

### LSTM

Deep learning model for sequential data. It can capture nonlinear patterns but requires more data and tuning.

## Evaluation Metrics

- **MAE**: Mean absolute error.
- **RMSE**: Root mean squared error.
- **MAPE**: Mean absolute percentage error.

## Getting Started

```powershell
git clone https://github.com/poip-boop/retail-sales-forecasting.git
cd retail-sales-forecasting
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## FRED API Key

The app needs a free FRED API key to download the retail-sales data. Create one at <https://fred.stlouisfed.org/docs/api/api_key.html>, then either set an environment variable:

```powershell
$env:FRED_API_KEY = "your-key"
```

Or copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and replace the placeholder. The secrets file is ignored by Git.

## Run

```powershell
streamlit run Retail_sales_time_series_forecasting.py
```





