"""Price data from Yahoo Finance (delayed public quotes)."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from forecaster.market import IST

NIFTY = "^NSEI"
EXOGENOUS = {
    "^INDIAVIX": "vix",
    "^NSEBANK": "banknifty",
    "^GSPC": "sp500",
    "USDINR=X": "usdinr",
}
OHLC = ["Open", "High", "Low", "Close"]


def _download(tickers: str | list[str], period: str, interval: str) -> pd.DataFrame:
    raw = yf.download(
        tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        label = tickers if isinstance(tickers, str) else ",".join(tickers)
        raise RuntimeError(f"No data returned from Yahoo Finance for {label} ({interval}).")
    return raw


def _ticker_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    frame = raw
    if isinstance(raw.columns, pd.MultiIndex):
        level = "Ticker" if "Ticker" in (raw.columns.names or []) else None
        if level is None:
            for candidate in range(raw.columns.nlevels):
                if ticker in set(raw.columns.get_level_values(candidate)):
                    level = candidate
                    break
        if level is None:
            raise RuntimeError(f"Yahoo response did not include {ticker}.")
        frame = raw.xs(ticker, axis=1, level=level)
    keep = [column for column in OHLC if column in frame.columns]
    frame = frame.loc[:, keep].apply(pd.to_numeric, errors="coerce").dropna(subset=["Close"])
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame


def _close_series(raw: pd.DataFrame, ticker: str) -> pd.Series:
    close = _ticker_frame(raw, ticker)["Close"]
    close = pd.to_numeric(close, errors="coerce").dropna()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close


def _as_naive_midnight(index: pd.Index) -> pd.DatetimeIndex:
    stamps = pd.DatetimeIndex(index)
    if stamps.tz is not None:
        stamps = stamps.tz_convert(IST).tz_localize(None)
    return stamps.normalize()


def load_daily(period: str = "max") -> pd.DataFrame:
    """Daily Nifty OHLC plus exogenous closes, aligned to Nifty trading days.

    Yahoo includes the in-progress session as the last row, so during market
    hours the final row is today's developing bar. S&P 500 is lagged one
    session because the US cash close is still in the future when India closes.
    """
    nifty = _ticker_frame(_download(NIFTY, period, "1d"), NIFTY)
    nifty.index = _as_naive_midnight(nifty.index)
    nifty = nifty[~nifty.index.duplicated(keep="last")].sort_index()

    extra = _download(list(EXOGENOUS), period, "1d")
    aligned: dict[str, pd.Series] = {}
    for ticker, name in EXOGENOUS.items():
        try:
            series = _close_series(extra, ticker)
        except RuntimeError:
            continue
        series.index = _as_naive_midnight(series.index)
        series = series[~series.index.duplicated(keep="last")].sort_index()
        calendar = nifty.index.union(series.index)
        filled = series.reindex(calendar).ffill()
        if name == "sp500":
            filled = filled.shift(1)
        aligned[name] = filled.reindex(nifty.index)

    if aligned:
        nifty = nifty.join(pd.DataFrame(aligned))
    return nifty


def sessions_from_intraday(bars: pd.DataFrame) -> pd.DataFrame:
    """Collapse intraday bars into one OHLC row per IST session."""
    if bars.empty:
        return pd.DataFrame(columns=OHLC)
    stamps = pd.DatetimeIndex(bars.index)
    if stamps.tz is not None:
        days = stamps.tz_convert(IST).date
    else:
        days = stamps.date
    grouped = bars.groupby(days)
    frame = pd.DataFrame(
        {
            "Open": grouped["Open"].first(),
            "High": grouped["High"].max(),
            "Low": grouped["Low"].min(),
            "Close": grouped["Close"].last(),
        }
    )
    frame.index = pd.to_datetime(frame.index)
    return frame


def stitch_daily(daily: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Append sessions that exist on the intraday tape but not yet on the daily feed.

    Yahoo's daily history can lag a finished session by a day. The 15-minute bars
    already contain it. Existing daily rows are left alone, because the official
    open comes from the auction and the first intraday bar does not. Macro
    columns are left blank on a stitched session so a carried print is not
    mistaken for a new return.
    """
    rolled = sessions_from_intraday(bars)
    missing = rolled.index.difference(pd.DatetimeIndex(daily.index))
    if len(missing) == 0:
        return daily
    extra = rolled.loc[missing].copy()
    for column in daily.columns:
        if column not in extra.columns:
            extra[column] = float("nan")
    extra = extra[daily.columns]
    combined = pd.concat([daily, extra]).sort_index()
    for column in combined.columns:
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    return combined


def load_intraday(interval: str = "15m", period: str = "60d") -> pd.DataFrame:
    """Intraday Nifty OHLC bars in IST. Yahoo keeps about 60 days of 15-minute history."""
    frame = _ticker_frame(_download(NIFTY, period, interval), NIFTY)
    stamps = pd.DatetimeIndex(frame.index)
    if stamps.tz is None:
        stamps = stamps.tz_localize("UTC")
    frame.index = stamps.tz_convert(IST)
    return frame.sort_index()
