"""Streamlit dashboard for comparing retail sales forecasts."""

import logging
import warnings
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from fredapi import Fred
from plotly.subplots import make_subplots
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Sequential
import tensorflow as tf

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

DEFAULTS = {
    "SERIES_ID": "RSXFS",
    "SERIES_NAME": "Advance Retail Sales: Retail Trade (RSXFS)",
    "DEFAULT_START_DATE": date(1992, 1, 1),
    "DEFAULT_END_DATE": date(2025, 4, 1),
    "FORECAST_HORIZON": 12,
    "LSTM_N_STEPS": 12,
    "LSTM_EPOCHS": 20,
    "LSTM_BATCH_SIZE": 16,
    "LSTM_UNITS": 32,
    "SARIMA_ORDER": (1, 1, 1),
    "SARIMA_SEASONAL_ORDER": (1, 1, 1, 12),
}


def get_fred_client() -> Fred | None:
    """Build the FRED client at runtime; never crash the app on import."""
    api_key = st.secrets.get("FRED_API_KEY", None) if hasattr(st, "secrets") else None
    if not api_key:
        import os
        api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        return None
    return Fred(api_key=api_key)


@st.cache_data(show_spinner=False)
def fetch_fred_series(series_id: str, start_date: str, end_date: str, _fred: Fred) -> pd.DataFrame:
    """Fetch a series from FRED and return a tidy date/value DataFrame."""
    logger.info("Fetching FRED series %s", series_id)
    raw = _fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
    df = raw.rename("value").to_frame()
    df.index.name = "date"
    df = df.reset_index()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna()


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag/rolling/calendar features to the time series."""
    df = df.dropna(subset=["value"]).copy()
    df = df.set_index("date")
    for lag in (1, 3, 12):
        df[f"lag_{lag}"] = df["value"].shift(lag)
    df["roll_mean_3"] = df["value"].rolling(3).mean()
    df["roll_std_12"] = df["value"].rolling(12).std()
    df["month"] = df.index.month
    df["year"] = df.index.year
    return df.dropna()


def _series_cache_key(s: pd.Series):
    return (tuple(np.round(s.values, 4)), str(s.index[0]), str(s.index[-1]))


@st.cache_resource(show_spinner=False, hash_funcs={pd.Series: _series_cache_key})
def train_sarima(train: pd.Series, horizon: int, order, seasonal_order):
    model = SARIMAX(
        train, order=order, seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    pred = fit.get_forecast(steps=horizon)
    ci = pred.conf_int()
    return pred.predicted_mean, ci.iloc[:, 0], ci.iloc[:, 1]


@st.cache_resource(show_spinner=False, hash_funcs={pd.Series: _series_cache_key})
def train_prophet(train: pd.Series, horizon: int):
    df = train.reset_index().rename(columns={"date": "ds", "value": "y"})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(df)
    future = model.make_future_dataframe(periods=horizon, freq="MS")
    forecast = model.predict(future).set_index("ds")
    return (
        forecast["yhat"].iloc[-horizon:],
        forecast["yhat_lower"].iloc[-horizon:],
        forecast["yhat_upper"].iloc[-horizon:],
    )


@st.cache_resource(show_spinner=False, hash_funcs={pd.Series: _series_cache_key})
def train_lstm(train: pd.Series, horizon: int, n_steps: int, units: int, epochs: int, batch_size: int):
    if len(train) <= n_steps + horizon:
        raise ValueError(
            f"Not enough history for LSTM: need more than {n_steps + horizon} points, have {len(train)}."
        )

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(train.values.reshape(-1, 1))

    def create_sequences(data, steps):
        X, y = [], []
        for i in range(len(data) - steps):
            X.append(data[i:i + steps])
            y.append(data[i + steps])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled, n_steps)

    model = Sequential([
        Input(shape=(n_steps, 1)),
        LSTM(units, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    forecast = []
    current_seq = X[-1].copy()
    for _ in range(horizon):
        pred = model.predict(current_seq.reshape(1, n_steps, 1), verbose=0)
        forecast.append(pred[0, 0])
        current_seq = np.roll(current_seq, -1, axis=0)
        current_seq[-1] = pred[0, 0]

    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
    idx = pd.date_range(start=train.index[-1], periods=horizon + 1, freq="MS")[1:]
    forecast = pd.Series(forecast, index=idx)
    std = np.std(forecast)
    return forecast, forecast - 1.96 * std, forecast + 1.96 * std


MODEL_REGISTRY = {
    "SARIMA": lambda train, horizon, cfg: train_sarima(
        train, horizon, cfg["SARIMA_ORDER"], cfg["SARIMA_SEASONAL_ORDER"]
    ),
    "Prophet": lambda train, horizon, cfg: train_prophet(train, horizon),
    "LSTM": lambda train, horizon, cfg: train_lstm(
        train, horizon, cfg["LSTM_N_STEPS"], cfg["LSTM_UNITS"], cfg["LSTM_EPOCHS"], cfg["LSTM_BATCH_SIZE"]
    ),
}


def evaluate_forecast(true: pd.Series, pred: pd.Series) -> dict:
    y_true, y_pred = np.array(true), np.array(pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nz = y_true != 0
    mape = np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100 if np.any(nz) else np.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape}


def plot_acf_pacf(series: pd.Series, lags: int = 24) -> go.Figure:
    acf_vals = acf(series, nlags=lags)
    pacf_vals = pacf(series, nlags=lags)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(lags + 1)), y=acf_vals, name="ACF"))
    fig.add_trace(go.Bar(x=list(range(lags + 1)), y=pacf_vals, name="PACF"))
    fig.update_layout(title="ACF and PACF", barmode="group", template="plotly_white")
    return fig


def plot_stl(series: pd.Series, period: int = 12) -> go.Figure:
    decomp = STL(series, period=period).fit()
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=("Observed", "Trend", "Seasonal", "Residual"))
    fig.add_trace(go.Scatter(x=series.index, y=decomp.observed, mode="lines", name="Observed"), row=1, col=1)
    fig.add_trace(go.Scatter(x=series.index, y=decomp.trend, mode="lines", name="Trend"), row=2, col=1)
    fig.add_trace(go.Scatter(x=series.index, y=decomp.seasonal, mode="lines", name="Seasonal"), row=3, col=1)
    fig.add_trace(go.Scatter(x=series.index, y=decomp.resid, mode="markers", name="Residual", marker=dict(size=4)), row=4, col=1)
    fig.update_layout(height=650, showlegend=False, template="plotly_white", title="STL Decomposition")
    return fig


def forecast_chart(actual: pd.Series, forecast: pd.Series, lower: pd.Series, upper: pd.Series, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual.index, y=actual.values, mode="lines", name="Actual", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode="lines", name="Forecast", line=dict(color="#ff7f0e")))
    fig.add_trace(go.Scatter(
        x=list(forecast.index) + list(forecast.index[::-1]),
        y=list(upper.values) + list(lower.values[::-1]),
        fill="toself", fillcolor="rgba(255,127,14,0.15)", line=dict(width=0),
        name="Confidence Interval", showlegend=True,
    ))
    fig.update_layout(title=title, template="plotly_white", hovermode="x unified")
    return fig


def main():
    st.set_page_config(page_title="Retail Sales Forecasting", page_icon="📈", layout="wide")

    st.title("Retail Sales Forecasting Dashboard")
    st.caption(f"Source: FRED — {DEFAULTS['SERIES_NAME']}")

    fred = get_fred_client()
    if fred is None:
        st.error(
            "No FRED API key found. Set the `FRED_API_KEY` environment variable "
            "(or add it to `st.secrets`) and restart the app."
        )
        st.stop()

    with st.sidebar:
        st.header("Settings")
        start_date = st.date_input("Start date", DEFAULTS["DEFAULT_START_DATE"])
        end_date = st.date_input("End date", DEFAULTS["DEFAULT_END_DATE"])
        if start_date >= end_date:
            st.error("Start date must be before end date.")
            st.stop()

        horizon = st.slider("Backtest / forecast horizon (months)", 1, 24, DEFAULTS["FORECAST_HORIZON"])
        selected_models = st.multiselect(
            "Models", list(MODEL_REGISTRY.keys()), default=list(MODEL_REGISTRY.keys())
        )

        with st.expander("LSTM settings"):
            lstm_units = st.slider("Units", 8, 128, DEFAULTS["LSTM_UNITS"], step=8)
            lstm_epochs = st.slider("Epochs", 5, 100, DEFAULTS["LSTM_EPOCHS"], step=5)
            lstm_steps = st.slider("Lookback window (months)", 3, 24, DEFAULTS["LSTM_N_STEPS"])

        cfg = {
            **DEFAULTS,
            "LSTM_UNITS": lstm_units,
            "LSTM_EPOCHS": lstm_epochs,
            "LSTM_N_STEPS": lstm_steps,
        }

        if st.button("🔄 Clear cache & refetch"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    with st.spinner("Fetching data from FRED…"):
        raw_df = fetch_fred_series(cfg["SERIES_ID"], str(start_date), str(end_date), fred)

    if raw_df.empty:
        st.warning("No data returned for this date range. Try widening it.")
        st.stop()

    df = preprocess_data(raw_df)
    min_required = cfg["LSTM_N_STEPS"] + horizon + 5
    if len(df) < min_required:
        st.warning(f"Not enough data after preprocessing ({len(df)} rows). Widen the date range or shrink the horizon/lookback.")
        st.stop()

    full_series = df["value"]
    train = full_series.iloc[:-horizon]
    test = full_series.iloc[-horizon:]

    # KPI header
    latest = full_series.iloc[-1]
    mom = (full_series.iloc[-1] / full_series.iloc[-2] - 1) * 100 if len(full_series) > 1 else np.nan
    yoy = (full_series.iloc[-1] / full_series.iloc[-13] - 1) * 100 if len(full_series) > 13 else np.nan
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest value", f"{latest:,.1f}")
    c2.metric("MoM change", f"{mom:+.2f}%" if not np.isnan(mom) else "n/a")
    c3.metric("YoY change", f"{yoy:+.2f}%" if not np.isnan(yoy) else "n/a")
    c4.metric("Observations", f"{len(full_series):,}")

    tab_eda, tab_backtest, tab_future = st.tabs(["📊 Overview", "🧪 Model Backtest", "🔮 Future Forecast"])

    # --- Overview -----------------------------------------------------
    with tab_eda:
        st.plotly_chart(
            px.line(full_series.reset_index(), x="date", y="value", title="Retail Sales Over Time", template="plotly_white"),
            use_container_width=True,
        )
        st.plotly_chart(plot_stl(full_series), use_container_width=True)

        stat, pval, *_ = adfuller(full_series)
        stationary = "stationary" if pval < 0.05 else "non-stationary"
        st.info(f"ADF statistic = {stat:.4f}, p-value = {pval:.4f} → series looks **{stationary}** at the 5% level.")

        st.plotly_chart(plot_acf_pacf(full_series), use_container_width=True)

        heatmap_data = df.reset_index().pivot_table(index="year", columns="month", values="value")
        st.plotly_chart(
            px.imshow(heatmap_data, labels=dict(x="Month", y="Year", color="Sales"), title="Seasonality Heatmap", template="plotly_white"),
            use_container_width=True,
        )

    # --- Backtest (train on all-but-last-N, compare vs held-out actuals) --
    with tab_backtest:
        st.subheader("Backtest: train on history, compare against the most recent held-out months")
        if not selected_models:
            st.warning("Select at least one model in the sidebar.")
        else:
            rows = []
            for name in selected_models:
                with st.spinner(f"Training {name}…"):
                    try:
                        forecast, lower, upper = MODEL_REGISTRY[name](train, horizon, cfg)
                        metrics = evaluate_forecast(test, forecast)
                        metrics["Model"] = name
                        rows.append(metrics)
                        with st.expander(f"{name} — forecast vs actual", expanded=False):
                            st.plotly_chart(
                                forecast_chart(full_series, forecast, lower, upper, f"{name}: Backtest"),
                                use_container_width=True,
                            )
                    except Exception as e:
                        logger.error("Error running %s: %s", name, e, exc_info=True)
                        st.error(f"{name} failed: {e}")
            if rows:
                st.dataframe(
                    pd.DataFrame(rows).set_index("Model").style.format("{:.2f}"),
                    use_container_width=True,
                )

    # --- Future forecast (train on full series, forecast beyond last date) --
    with tab_future:
        st.subheader("Forecast beyond the latest available data point")
        model_choice = st.selectbox("Model", list(MODEL_REGISTRY.keys()), key="future_model_select")
        with st.spinner(f"Training {model_choice} on full history…"):
            try:
                forecast, lower, upper = MODEL_REGISTRY[model_choice](full_series, horizon, cfg)
                fig = forecast_chart(full_series, forecast, lower, upper, f"{model_choice}: {horizon}-month Forecast")
                st.plotly_chart(fig, use_container_width=True)

                out = pd.DataFrame({"date": forecast.index, "forecast": forecast.values, "lower": lower.values, "upper": upper.values})
                st.dataframe(out, use_container_width=True, hide_index=True)
                st.download_button("⬇️ Download forecast (CSV)", out.to_csv(index=False), "future_forecast.csv", "text/csv")
            except Exception as e:
                logger.error("Error generating future forecast for %s: %s", model_choice, e, exc_info=True)
                st.error(f"Failed to generate forecast for {model_choice}: {e}")


if __name__ == "__main__":
    main()
