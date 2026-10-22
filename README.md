# Nifty 50 trend forecaster

A Streamlit app that forecasts the direction of the Nifty 50 index with a machine-learning ensemble
built on 50+ technical and cross-market indicators. While NSE is in session (Monday to Friday,
09:15–15:30 IST) the forecast page re-scores the live bar every 60 seconds.

| Forecast | Horizon | Data |
|---|---|---|
| Next session | 1 trading day | Daily bars since 2007 |
| Week ahead | 5 trading days | Daily bars since 2007 |
| Next hour (experimental) | 4 × 15-minute bars | Last 60 days of intraday bars |

**Indicators:** EMA 9/21/50/200, Supertrend, ADX/DMI, Ichimoku, MACD, RSI, Stochastic, Williams %R,
CCI, ATR, Bollinger Bands, floor pivots (P, R1/R2, S1/S2), 52-week range, India VIX, Nifty Bank
relative strength, previous-day S&P 500 and USD/INR moves.

**Model:** soft-voting ensemble of gradient boosting, random forest and logistic regression, validated
with purged walk-forward splits. The *Model performance* page shows out-of-sample accuracy, AUC and an
equity curve. Expect accuracy in the low-to-mid 50s: index direction is close to a random walk.

Data comes from Yahoo Finance (delayed). This is for research and education, not investment advice.

## Run locally

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run streamlit_app.py
```

The first load trains the models (about 15 seconds on a laptop, longer on small cloud instances);
they are then cached and retrained once per completed session.

## Tests

```bash
.venv/bin/pip install pytest && .venv/bin/python -m pytest
```

## Deploy

**Streamlit Community Cloud (free):** at [share.streamlit.io](https://share.streamlit.io) choose
*Create app*, select this repository and branch, and set the main file path to
`NIFTY50_trend_forecaster/streamlit_app.py`. Dependencies install from the `requirements.txt` next to it.
Community Cloud reads `.streamlit/config.toml` only from the repository root, so the app falls back to
the viewer's light or dark theme there.

**Docker (Render, Railway, Fly.io, Cloud Run, any VM):**

```bash
docker build -t nifty-forecaster .
docker run -p 8501:8501 nifty-forecaster
```
