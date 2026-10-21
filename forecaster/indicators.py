"""Technical indicators computed from OHLC bars.

Nifty 50 index volume from Yahoo is unreliable, often zero, so every indicator
here uses price only. Frames need Open, High, Low and Close columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def wilder(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, adjust=False).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["Close"].shift()
    spans = pd.concat(
        [df["High"] - df["Low"], (df["High"] - prev).abs(), (df["Low"] - prev).abs()],
        axis=1,
    )
    return spans.max(axis=1)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    return wilder(true_range(df), length)


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = wilder(delta.clip(lower=0), length)
    loss = wilder((-delta).clip(lower=0), length)
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    signal_line = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame(
        {"macd": line, "macd_signal": signal_line, "macd_hist": line - signal_line},
        index=close.index,
    )


def adx(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr_n = atr(df, length)
    plus_di = 100 * wilder(plus_dm, length) / tr_n.replace(0, np.nan)
    minus_di = 100 * wilder(minus_dm, length) / tr_n.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return pd.DataFrame(
        {"adx": wilder(dx, length), "di_plus": plus_di, "di_minus": minus_di},
        index=df.index,
    )


def stochastic(df: pd.DataFrame, length: int = 14, smooth: int = 3) -> pd.DataFrame:
    lowest = df["Low"].rolling(length).min()
    highest = df["High"].rolling(length).max()
    k = 100 * (df["Close"] - lowest) / (highest - lowest).replace(0, np.nan)
    return pd.DataFrame({"stoch_k": k, "stoch_d": k.rolling(smooth).mean()}, index=df.index)


def williams_r(df: pd.DataFrame, length: int = 14) -> pd.Series:
    highest = df["High"].rolling(length).max()
    lowest = df["Low"].rolling(length).min()
    return -100 * (highest - df["Close"]) / (highest - lowest).replace(0, np.nan)


def cci(df: pd.DataFrame, length: int = 20) -> pd.Series:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    mean = typical.rolling(length).mean()
    mad = (typical - mean).abs().rolling(length).mean()
    return (typical - mean) / (0.015 * mad.replace(0, np.nan))


def bollinger(close: pd.Series, length: int = 20, width: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(length).mean()
    sd = close.rolling(length).std(ddof=0)
    upper = mid + width * sd
    lower = mid - width * sd
    return pd.DataFrame(
        {
            "bb_mid": mid,
            "bb_up": upper,
            "bb_low": lower,
            "bb_width": (upper - lower) / mid.replace(0, np.nan),
            "bb_pctb": (close - lower) / (upper - lower).replace(0, np.nan),
        },
        index=close.index,
    )


def supertrend(df: pd.DataFrame, length: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """Supertrend line and direction (+1 up-trend, -1 down-trend)."""
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    atr_v = atr(df, length).to_numpy(dtype=float)
    basic_upper = (high + low) / 2 + multiplier * atr_v
    basic_lower = (high + low) / 2 - multiplier * atr_v
    n = len(df)
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    direction = np.zeros(n)
    line = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(atr_v[i]):
            continue
        if i == 0 or np.isnan(final_upper[i - 1]):
            final_upper[i] = basic_upper[i]
            final_lower[i] = basic_lower[i]
            direction[i] = 1
            line[i] = final_lower[i]
            continue
        if basic_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]
        if basic_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]
        if direction[i - 1] > 0:
            direction[i] = -1 if close[i] < final_lower[i] else 1
        else:
            direction[i] = 1 if close[i] > final_upper[i] else -1
        line[i] = final_lower[i] if direction[i] > 0 else final_upper[i]
    return pd.DataFrame({"supertrend": line, "supertrend_dir": direction}, index=df.index)


def ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou: int = 52) -> pd.DataFrame:
    high, low = df["High"], df["Low"]
    tenkan_s = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_s = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    span_a = ((tenkan_s + kijun_s) / 2).shift(kijun)
    span_b = ((high.rolling(senkou).max() + low.rolling(senkou).min()) / 2).shift(kijun)
    return pd.DataFrame(
        {
            "tenkan": tenkan_s,
            "kijun": kijun_s,
            "span_a": span_a,
            "span_b": span_b,
            "cloud_top": pd.concat([span_a, span_b], axis=1).max(axis=1),
            "cloud_base": pd.concat([span_a, span_b], axis=1).min(axis=1),
        },
        index=df.index,
    )


def pivot_points(df: pd.DataFrame) -> pd.DataFrame:
    """Classic floor pivots from the previous session's high, low and close."""
    prev_high = df["High"].shift(1)
    prev_low = df["Low"].shift(1)
    prev_close = df["Close"].shift(1)
    pivot = (prev_high + prev_low + prev_close) / 3
    span = prev_high - prev_low
    return pd.DataFrame(
        {
            "pivot": pivot,
            "r1": 2 * pivot - prev_low,
            "s1": 2 * pivot - prev_high,
            "r2": pivot + span,
            "s2": pivot - span,
        },
        index=df.index,
    )


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    """Adds raw indicator columns to an OHLC frame."""
    out = df.copy()
    close = out["Close"]
    for length in (9, 21, 50, 200):
        out[f"ema_{length}"] = ema(close, length)
    out["atr"] = atr(out)
    out["rsi"] = rsi(close)
    out = out.join(macd(close))
    out = out.join(adx(out))
    out = out.join(stochastic(out))
    out = out.join(bollinger(close))
    out = out.join(supertrend(out))
    out = out.join(ichimoku(out))
    out["williams_r"] = williams_r(out)
    out["cci"] = cci(out)
    return out
