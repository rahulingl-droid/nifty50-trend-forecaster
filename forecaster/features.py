"""Scale-free model features, targets, and the indicator scorecard."""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecaster import indicators as ind

PERCENT_FEATURES = {
    "ret_lag1",
    "ret_lag2",
    "ret_lag3",
    "ret_lag5",
    "ret_10",
    "ret_20",
    "atr_pct",
    "dist_ema_9",
    "dist_ema_21",
    "dist_ema_50",
    "dist_ema_200",
    "ema9_vs_21",
    "ema50_vs_200",
    "macd_hist_pct",
    "dist_supertrend",
    "cloud_pos",
    "tk_vs_kj",
    "dist_pivot",
    "dist_r1",
    "dist_s1",
    "dist_52w_high",
    "dist_52w_low",
    "vix_change",
    "vix_vs_20d",
    "banknifty_rel_1d",
    "banknifty_rel_5d",
    "sp500_prev_ret",
    "usdinr_ret_5d",
    "ret_from_open",
}

LABELS = {
    "ret_lag1": "1-session return",
    "ret_lag2": "2-session return",
    "ret_lag3": "3-session return",
    "ret_lag5": "5-session return",
    "ret_10": "10-session return",
    "ret_20": "20-session return",
    "atr_pct": "ATR %",
    "dist_ema_9": "Distance from EMA 9",
    "dist_ema_21": "Distance from EMA 21",
    "dist_ema_50": "Distance from EMA 50",
    "dist_ema_200": "Distance from EMA 200",
    "ema9_vs_21": "EMA 9 vs 21",
    "ema50_vs_200": "EMA 50 vs 200",
    "rsi": "RSI 14",
    "macd_hist_pct": "MACD histogram",
    "adx": "ADX 14",
    "di_spread": "DI+ minus DI-",
    "stoch_k": "Stochastic %K",
    "bb_pctb": "Bollinger %B",
    "bb_width": "Bollinger width",
    "supertrend_dir": "Supertrend direction",
    "dist_supertrend": "Distance from Supertrend",
    "williams_r": "Williams %R",
    "cci": "CCI 20",
    "cloud_pos": "Price vs Ichimoku cloud",
    "tk_vs_kj": "Tenkan vs Kijun",
    "dist_pivot": "Distance from pivot",
    "dist_r1": "Distance from R1",
    "dist_s1": "Distance from S1",
    "dist_52w_high": "Distance from 52-week high",
    "dist_52w_low": "Distance from 52-week low",
    "day_of_week": "Weekday",
    "vix": "India VIX",
    "vix_change": "India VIX change",
    "vix_vs_20d": "India VIX vs 20-session average",
    "banknifty_rel_1d": "Bank Nifty vs Nifty, 1 session",
    "banknifty_rel_5d": "Bank Nifty vs Nifty, 5 sessions",
    "sp500_prev_ret": "Prior S&P 500 return",
    "usdinr_ret_5d": "USD/INR 5-session return",
    "session_progress": "Session progress",
    "ret_from_open": "Return from the open",
    "day_range_position": "Position in the day's range",
}


def label(feature: str) -> str:
    return LABELS.get(feature, feature.replace("_", " "))


def _technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features shared by the daily and intraday models. `df` must come from add_all."""
    close = df["Close"]
    log_close = np.log(close.replace(0, np.nan))
    out = pd.DataFrame(index=df.index)
    for lag in (1, 2, 3, 5):
        out[f"ret_lag{lag}"] = log_close.diff(lag)
    out["ret_10"] = log_close.diff(10)
    out["ret_20"] = log_close.diff(20)
    out["atr_pct"] = df["atr"] / close.replace(0, np.nan)
    for length in (9, 21, 50, 200):
        out[f"dist_ema_{length}"] = close / df[f"ema_{length}"].replace(0, np.nan) - 1
    out["ema9_vs_21"] = df["ema_9"] / df["ema_21"].replace(0, np.nan) - 1
    out["ema50_vs_200"] = df["ema_50"] / df["ema_200"].replace(0, np.nan) - 1
    out["rsi"] = df["rsi"]
    out["macd_hist_pct"] = df["macd_hist"] / close.replace(0, np.nan)
    out["adx"] = df["adx"]
    out["di_spread"] = df["di_plus"] - df["di_minus"]
    out["stoch_k"] = df["stoch_k"]
    out["bb_pctb"] = df["bb_pctb"]
    out["bb_width"] = df["bb_width"]
    out["supertrend_dir"] = df["supertrend_dir"]
    out["dist_supertrend"] = close / df["supertrend"].replace(0, np.nan) - 1
    out["williams_r"] = df["williams_r"]
    out["cci"] = df["cci"]
    out["cloud_pos"] = (close - df["cloud_base"]) / close.replace(0, np.nan)
    out["tk_vs_kj"] = df["tenkan"] / df["kijun"].replace(0, np.nan) - 1
    if "pivot" in df.columns:
        out["dist_pivot"] = close / df["pivot"].replace(0, np.nan) - 1
        out["dist_r1"] = close / df["r1"].replace(0, np.nan) - 1
        out["dist_s1"] = close / df["s1"].replace(0, np.nan) - 1
    return out.replace([np.inf, -np.inf], np.nan)


def daily_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Daily bars with indicators and same-day pivot levels from the previous session."""
    out = ind.add_all(df)
    return out.join(ind.pivot_points(df))


def daily_features(df: pd.DataFrame) -> pd.DataFrame:
    out = _technical_features(df)
    close = df["Close"]
    out["dist_52w_high"] = close / close.rolling(252).max() - 1
    out["dist_52w_low"] = close / close.rolling(252).min() - 1
    out["day_of_week"] = pd.DatetimeIndex(df.index).dayofweek.astype(float)
    if "vix" in df.columns:
        vix = df["vix"].replace(0, np.nan)
        out["vix"] = vix
        out["vix_change"] = np.log(vix).diff()
        out["vix_vs_20d"] = vix / vix.rolling(20).mean() - 1
    if "banknifty" in df.columns:
        bank = np.log(df["banknifty"].replace(0, np.nan))
        nifty = np.log(close.replace(0, np.nan))
        out["banknifty_rel_1d"] = bank.diff(1) - nifty.diff(1)
        out["banknifty_rel_5d"] = bank.diff(5) - nifty.diff(5)
    if "sp500" in df.columns:
        out["sp500_prev_ret"] = np.log(df["sp500"].replace(0, np.nan)).diff()
    if "usdinr" in df.columns:
        out["usdinr_ret_5d"] = np.log(df["usdinr"].replace(0, np.nan)).diff(5)
    return out.replace([np.inf, -np.inf], np.nan)


def intraday_frame(bars: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Intraday bars with indicators and today's floor pivots from the prior daily session."""
    out = ind.add_all(bars)
    pivots = ind.pivot_points(daily)
    if pivots.empty:
        return out
    by_day = pivots.copy()
    by_day.index = pd.DatetimeIndex(by_day.index).tz_localize(None).normalize()
    day_key = pd.Series(pd.DatetimeIndex(out.index).tz_convert("Asia/Kolkata").tz_localize(None).normalize(), index=out.index)
    attached = by_day.reindex(day_key.to_numpy())
    attached.index = out.index
    return out.join(attached)


def intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    out = _technical_features(df)
    stamps = pd.DatetimeIndex(df.index)
    minutes = stamps.hour * 60 + stamps.minute - (9 * 60 + 15)
    out["session_progress"] = pd.Series(minutes / 375, index=df.index).clip(0, 1)
    out["day_of_week"] = stamps.dayofweek.astype(float)
    day = pd.Series(stamps.date, index=df.index)
    session_open = df.groupby(day)["Open"].transform("first")
    out["ret_from_open"] = df["Close"] / session_open.replace(0, np.nan) - 1
    running_high = df.groupby(day)["High"].cummax()
    running_low = df.groupby(day)["Low"].cummin()
    width = (running_high - running_low).replace(0, np.nan)
    out["day_range_position"] = (df["Close"] - running_low) / width
    return out.replace([np.inf, -np.inf], np.nan)


def direction_target(close: pd.Series, horizon: int = 1, same_session: bool = False) -> pd.Series:
    """1 when close is higher `horizon` bars ahead, 0 when lower, NaN when unknown."""
    future = close.shift(-horizon)
    target = (future > close).astype(float).where(future.notna())
    if same_session:
        days = pd.Series(pd.DatetimeIndex(close.index).date, index=close.index)
        target = target.where(days.eq(days.shift(-horizon)))
    return target


def session_close_target(close: pd.Series) -> pd.Series:
    """1 when the same session's final close is above this bar. The last bar is unknown."""
    day = pd.Series(pd.DatetimeIndex(close.index).date, index=close.index)
    session_close = close.groupby(day).transform("last")
    is_last = close.groupby(day).cumcount(ascending=False).eq(0)
    target = (session_close > close).astype(float).where(session_close.notna())
    return target.where(~is_last)


def _bias(value: float, up_median: float, down_median: float) -> str:
    if pd.isna(value) or pd.isna(up_median) or pd.isna(down_median):
        return "Neutral"
    if abs(up_median - down_median) <= 1e-8:
        return "Neutral"
    if abs(value - up_median) < abs(value - down_median):
        return "Bullish"
    return "Bearish"


def indicator_pivot(
    features: pd.DataFrame,
    target: pd.Series,
    importance: pd.Series,
    week: list[pd.Timestamp],
    top: int = 8,
) -> pd.DataFrame:
    """Pivot the strongest features across this week's sessions.

    Up-day and down-day medians use only rows before this week, so the bias
    column is a historical comparison rather than a peek at the week's outcome.
    """
    ranked = importance.dropna().sort_values(ascending=False).head(top)
    known = target.dropna()
    shared = features.index.intersection(known.index)
    history = features.loc[shared]
    labels = known.loc[shared]
    if week:
        week_start = min(pd.Timestamp(day).normalize() for day in week)
        history = history.loc[pd.DatetimeIndex(history.index).normalize() < week_start]
        labels = labels.loc[history.index]
    rows = []
    feature_dates = pd.DatetimeIndex(features.index).normalize()
    for name, score in ranked.items():
        column = history[name] if name in history.columns else pd.Series(dtype=float)
        up = column.loc[labels.eq(1)].median() if not column.empty else np.nan
        down = column.loc[labels.eq(0)].median() if not column.empty else np.nan
        latest = features[name].iloc[-1] if name in features.columns else np.nan
        row: dict[str, object] = {
            "Indicator": label(str(name)),
            "Importance": float(score),
            "Up-day median": up,
            "Down-day median": down,
            "Latest": latest,
            "Bias": _bias(float(latest) if pd.notna(latest) else np.nan, float(up) if pd.notna(up) else np.nan, float(down) if pd.notna(down) else np.nan),
            "_percent": name in PERCENT_FEATURES,
        }
        for day in week:
            stamp = pd.Timestamp(day).normalize()
            matches = features.loc[feature_dates == stamp, name] if name in features.columns else pd.Series(dtype=float)
            row[stamp.strftime("%A")] = float(matches.iloc[-1]) if not matches.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _side(condition: bool | None, bullish: str, bearish: str) -> str:
    if condition is None:
        return "Neutral"
    return bullish if condition else bearish


def indicator_scorecard(frame: pd.DataFrame) -> pd.DataFrame:
    """Rule-of-thumb bullish or bearish reading for each indicator on the latest bar."""
    row = frame.iloc[-1]
    close = float(row["Close"])

    def reading(name: str, group: str, value: str, signal: str) -> dict[str, str]:
        return {"Indicator": name, "Group": group, "Reading": value, "Signal": signal}

    rows = [
        reading(
            "Price vs EMA 21",
            "Trend",
            f"{close / float(row['ema_21']) - 1:+.2%}",
            _side(close > float(row["ema_21"]), "Bullish", "Bearish"),
        ),
        reading(
            "Price vs EMA 50",
            "Trend",
            f"{close / float(row['ema_50']) - 1:+.2%}",
            _side(close > float(row["ema_50"]), "Bullish", "Bearish"),
        ),
        reading(
            "Price vs EMA 200",
            "Trend",
            f"{close / float(row['ema_200']) - 1:+.2%}",
            _side(close > float(row["ema_200"]), "Bullish", "Bearish"),
        ),
        reading(
            "EMA 9 / 21 cross",
            "Trend",
            "EMA 9 above EMA 21" if float(row["ema_9"]) > float(row["ema_21"]) else "EMA 9 below EMA 21",
            _side(float(row["ema_9"]) > float(row["ema_21"]), "Bullish", "Bearish"),
        ),
        reading(
            "Supertrend (10, 3)",
            "Trend",
            f"{float(row['supertrend']):,.0f}",
            "Bullish" if float(row["supertrend_dir"]) > 0 else "Bearish",
        ),
        reading(
            "Ichimoku cloud",
            "Trend",
            "Above the cloud" if close > float(row["cloud_top"]) else "Below the cloud" if close < float(row["cloud_base"]) else "Inside the cloud",
            "Bullish" if close > float(row["cloud_top"]) else "Bearish" if close < float(row["cloud_base"]) else "Neutral",
        ),
        reading("RSI 14", "Momentum", f"{float(row['rsi']):.1f}", "Bullish" if float(row["rsi"]) >= 60 else "Bearish" if float(row["rsi"]) <= 40 else "Neutral"),
        reading("MACD histogram", "Momentum", f"{float(row['macd_hist']):+.1f}", _side(float(row["macd_hist"]) > 0, "Bullish", "Bearish")),
        reading(
            "ADX / DI",
            "Momentum",
            f"ADX {float(row['adx']):.1f}",
            "Bullish" if float(row["adx"]) >= 20 and float(row["di_plus"]) > float(row["di_minus"]) else "Bearish" if float(row["adx"]) >= 20 else "Neutral",
        ),
        reading("Stochastic %K", "Momentum", f"{float(row['stoch_k']):.1f}", "Bullish" if float(row["stoch_k"]) >= 60 else "Bearish" if float(row["stoch_k"]) <= 40 else "Neutral"),
        reading("Williams %R", "Momentum", f"{float(row['williams_r']):.1f}", "Bullish" if float(row["williams_r"]) >= -40 else "Bearish" if float(row["williams_r"]) <= -60 else "Neutral"),
        reading("CCI 20", "Momentum", f"{float(row['cci']):+.0f}", "Bullish" if float(row["cci"]) >= 100 else "Bearish" if float(row["cci"]) <= -100 else "Neutral"),
        reading("Bollinger %B", "Volatility", f"{float(row['bb_pctb']):.2f}", "Bullish" if float(row["bb_pctb"]) >= 0.8 else "Bearish" if float(row["bb_pctb"]) <= 0.2 else "Neutral"),
    ]
    if "pivot" in frame.columns and pd.notna(row["pivot"]):
        rows.append(
            reading(
                "Floor pivot",
                "Pivot",
                f"{float(row['pivot']):,.0f}",
                _side(close > float(row["pivot"]), "Bullish", "Bearish"),
            )
        )
    return pd.DataFrame(rows)
