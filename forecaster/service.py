"""Cached data and model access shared by the Streamlit app."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
import streamlit as st

from forecaster import data, features as fe, model as ml
from forecaster.market import SessionStatus, session_status
from forecaster.model import ForecastModel
from forecaster.week import as_date, build_week

HORIZONS = (1, 2, 3, 4, 5)


@st.cache_data(ttl="1h", show_spinner=False)
def daily_history() -> pd.DataFrame:
    return data.load_daily("max")


@st.cache_data(ttl="60s", show_spinner=False)
def daily_recent() -> pd.DataFrame:
    return data.load_daily("10d")


@st.cache_data(ttl="60s", show_spinner=False)
def intraday_bars() -> pd.DataFrame:
    return data.load_intraday("15m", "60d")


def merged_daily() -> pd.DataFrame:
    history = daily_history()
    recent = daily_recent()
    frame = pd.concat([history, recent])
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    try:
        frame = data.stitch_daily(frame, intraday_bars())
    except RuntimeError:
        return frame
    return frame


def _without_open_session(frame: pd.DataFrame, today: date, market_open: bool) -> pd.DataFrame:
    if frame.empty or not market_open:
        return frame
    if as_date(frame.index[-1]) == today:
        return frame.iloc[:-1]
    return frame


def _forward_return(close: pd.Series, horizon: int) -> pd.Series:
    return close.shift(-horizon) / close - 1


@st.cache_resource(ttl="6h", show_spinner="Training the walk-forward ensemble on Nifty history…")
def daily_models(cutoff: str) -> dict[int, ForecastModel]:
    del cutoff  # cache key: a new completed session trains again
    status = session_status()
    raw = _without_open_session(merged_daily(), status.now.date(), status.is_open)
    framed = fe.daily_frame(raw)
    model_features = fe.daily_features(framed)
    trained: dict[int, ForecastModel] = {}
    for horizon in HORIZONS:
        trained[horizon] = ml.train(
            model_features,
            fe.direction_target(framed["Close"], horizon),
            _forward_return(framed["Close"], horizon),
            horizon,
        )
    return trained


@st.cache_resource(ttl="12h", show_spinner="Training the live session model…")
def intraday_model(cutoff: str) -> ForecastModel | None:
    del cutoff
    status = session_status()
    try:
        bars = intraday_bars()
    except RuntimeError:
        return None
    dates = pd.Series(pd.DatetimeIndex(bars.index).date, index=bars.index)
    history = bars.loc[dates != status.now.date()]
    if history.empty:
        return None
    daily = _without_open_session(merged_daily(), status.now.date(), status.is_open)
    framed = fe.intraday_frame(history, fe.daily_frame(daily))
    model_features = fe.intraday_features(framed)
    close = framed["Close"]
    day = pd.Series(pd.DatetimeIndex(close.index).date, index=close.index)
    session_close = close.groupby(day).transform("last")
    is_last = close.groupby(day).cumcount(ascending=False).eq(0)
    forward = (session_close / close - 1).where(~is_last)
    try:
        return ml.train(model_features, fe.session_close_target(close), forward, horizon=0)
    except ValueError:
        return None


@dataclass
class Snapshot:
    status: SessionStatus
    daily: pd.DataFrame
    framed: pd.DataFrame
    features: pd.DataFrame
    models: dict[int, ForecastModel]
    intraday: pd.DataFrame
    session_probability: float | None
    components: dict[str, float]
    week: pd.DataFrame


def snapshot() -> Snapshot:
    status = session_status()
    daily = merged_daily()
    framed = fe.daily_frame(daily)
    model_features = fe.daily_features(framed)
    cutoff = str(as_date(_without_open_session(daily, status.now.date(), status.is_open).index.max()))
    models = daily_models(cutoff)
    session_probability = None
    components: dict[str, float] = {}
    intraday = pd.DataFrame()
    try:
        intraday = intraday_bars()
    except RuntimeError:
        intraday = pd.DataFrame()
    live_model = intraday_model(status.now.date().isoformat())
    if live_model is not None and not intraday.empty and status.is_open:
        live_frame = fe.intraday_frame(intraday, framed)
        live_features = fe.intraday_features(live_frame)
        latest = live_features.iloc[[-1]]
        session_probability = live_model.predict_proba(latest)
        components = live_model.component_probas(latest)
    else:
        decision = _without_open_session(model_features, status.now.date(), status.is_open)
        if not decision.empty and 1 in models:
            latest = decision.iloc[[-1]]
            components = models[1].component_probas(latest)
    week = build_week(
        daily,
        model_features,
        models,
        status.now.date(),
        session_probability,
        status.is_open,
    )
    return Snapshot(
        status=status,
        daily=daily,
        framed=framed,
        features=model_features,
        models=models,
        intraday=intraday,
        session_probability=session_probability,
        components=components,
        week=week,
    )


def outlook(probability: float | None) -> tuple[str, str, str]:
    """Label, Streamlit color and Material icon for a probability of an up move."""
    if probability is None or pd.isna(probability):
        return "No call", "gray", ":material/remove:"
    if probability >= 0.56:
        return "Up", "green", ":material/trending_up:"
    if probability <= 0.44:
        return "Down", "red", ":material/trending_down:"
    return "Flat", "gray", ":material/trending_flat:"
