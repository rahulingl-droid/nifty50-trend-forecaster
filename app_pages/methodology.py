import streamlit as st

st.title("Methodology")

st.markdown(
    """
### Data
- **Nifty 50** (`^NSEI`) daily bars since 2007 plus the last 60 days of 15-minute bars, from Yahoo Finance.
  Quotes are delayed and the index's volume field is unreliable, so no volume-based indicator is used.
- **Cross-market drivers:** India VIX, Nifty Bank, S&P 500 and USD/INR. US and FX closes print after the
  Indian session ends, so the model only sees *the previous day's* S&P 500 and USD/INR move.

### Indicators (features)
| Group | Indicators |
|---|---|
| Trend | EMA 9/21/50/200 distance and crossovers, Supertrend (10, 3), ADX with +DI/−DI, Ichimoku tenkan, kijun and senkou B |
| Momentum | RSI (14), MACD (12, 26, 9) line, histogram and slope, Stochastic %K/%D, Williams %R, CCI (20), 1–20 bar returns |
| Volatility | ATR (14), Bollinger %B and bandwidth, realised volatility and its regime ratio, India VIX level and change |
| Levels | Classic floor pivot, R1, R2, S1, S2 from the previous session, 52-week high and low distance |
| Context | Day of week, gap at open, bar body and range position, Nifty Bank relative strength; intraday adds time of day and return from the day's open |

All features are scale-free (ratios, oscillators, or distances in ATR units), so a model trained on 2010
prices applies to today's levels.

### Model
A soft-voting ensemble of **histogram gradient boosting**, a **random forest**, and **L2-regularised
logistic regression**. Each predicts the probability that the close *h* bars ahead is higher than now:

- **Next session:** h = 1 trading day
- **Next 5 sessions:** h = 5 trading days (a Monday-to-Friday week)
- **Next hour:** h = 4 fifteen-minute bars, only within the same session

Models retrain once per completed session. While the market is open (Monday to Friday, 09:15–15:30 IST)
the page refreshes every 60 seconds and re-scores the live bar with the latest indicators.

### Validation
Five-fold expanding-window walk-forward validation with a purge gap equal to the horizon, so no
overlapping target leaks from test into train. The *Model performance* page reports only these
out-of-sample results.

### Read this before using the forecasts
Index direction is close to a random walk. Out-of-sample accuracy in the low-to-mid 50s and an AUC
slightly above 0.5 is a realistic result; anything far higher on daily index data usually means look-ahead
bias. Treat the output as one input among many, not a trading signal. Exchange holidays are not modelled.
"""
)
