"""Map horizon forecasts onto the Monday-Friday board."""

from __future__ import annotations

from datetime import date

import pandas as pd

from forecaster.market import next_weekday, week_dates
from forecaster.model import ForecastModel

HORIZONS = (1, 2, 3, 4, 5)


def as_date(value) -> date:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("Asia/Kolkata").tz_localize(None)
    return stamp.date()


def trading_dates(index: pd.Index) -> list[date]:
    seen: list[date] = []
    for value in index:
        day = as_date(value)
        if not seen or seen[-1] != day:
            seen.append(day)
    return seen


def project_sessions(known: list[date], through: date) -> list[date]:
    if not known:
        return []
    days = list(known)
    cursor = days[-1]
    while cursor < through:
        cursor = next_weekday(cursor)
        if cursor not in days:
            days.append(cursor)
    return days


def sessions_between(days: list[date], start: date, end: date) -> int | None:
    """How many sessions after `start` land on `end`."""
    if start not in days or end not in days:
        return None
    horizon = days.index(end) - days.index(start)
    if horizon < 1:
        return None
    return horizon


def _row_on(features: pd.DataFrame, day: date) -> pd.DataFrame | None:
    matches = [stamp for stamp in features.index if as_date(stamp) == day]
    if not matches:
        return None
    return features.loc[[matches[-1]]]


def probability(model: ForecastModel, features: pd.DataFrame, day: date) -> tuple[float | None, str]:
    oos_days = [stamp for stamp in model.oos.index if as_date(stamp) == day]
    if oos_days:
        return float(model.oos.loc[oos_days[-1], "proba"]), "walk-forward"
    row = _row_on(features, day)
    if row is None:
        return None, "missing"
    return model.predict_proba(row), "fitted model"


def build_week(
    daily: pd.DataFrame,
    features: pd.DataFrame,
    models: dict[int, ForecastModel],
    today: date,
    session_probability: float | None,
    market_open: bool,
) -> pd.DataFrame:
    """One row per weekday.

    The Monday call is the ensemble's view from the last session before this
    week. The live call updates from the latest completed session, and while
    the market is open today's row uses the intraday session model.
    """
    known = trading_dates(daily.index)
    week = week_dates(today)
    calendar = project_sessions(known, week[-1])
    prior = [day for day in known if day < week[0]]
    monday_decision = prior[-1] if prior else None
    completed = [day for day in known if day < today or (day == today and not market_open)]
    live_decision = completed[-1] if completed else None

    rows = []
    for day in week:
        phase = _phase(day, today, known, market_open)
        horizon_monday = sessions_between(calendar, monday_decision, day) if monday_decision else None
        monday_prob, monday_source = _lookup(models, features, monday_decision, horizon_monday)
        if phase == "settled":
            previous = _previous(known, day)
            live_prob, live_source = _lookup(models, features, previous, 1)
            actual = _return(daily, day)
        elif phase == "live" and market_open and session_probability is not None:
            live_prob, live_source = session_probability, "session model"
            actual = _developing_return(daily, day)
        else:
            horizon_live = sessions_between(calendar, live_decision, day) if live_decision else None
            live_prob, live_source = _lookup(models, features, live_decision, horizon_live)
            actual = _developing_return(daily, day) if phase == "live" else None
        rows.append(
            {
                "Weekday": day.strftime("%A"),
                "Date": day.isoformat(),
                "Phase": phase,
                "Monday call": monday_prob,
                "Monday source": monday_source,
                "Live call": live_prob,
                "Live source": live_source,
                "Return": actual,
            }
        )
    return pd.DataFrame(rows)


def _lookup(models, features, decision: date | None, horizon: int | None) -> tuple[float | None, str]:
    if decision is None or horizon is None or horizon not in models:
        return None, "outside horizon"
    return probability(models[horizon], features, decision)


def _previous(days: list[date], day: date) -> date | None:
    if day not in days:
        return None
    position = days.index(day)
    if position == 0:
        return None
    return days[position - 1]


def _phase(day: date, today: date, known: list[date], market_open: bool) -> str:
    if day > today:
        return "ahead"
    if day == today:
        if market_open:
            return "live"
        return "settled" if day in known else "ahead"
    if day in known:
        return "settled"
    return "no session"


def _close_on(daily: pd.DataFrame, day: date) -> float | None:
    matches = daily.loc[[stamp for stamp in daily.index if as_date(stamp) == day], "Close"]
    if matches.empty:
        return None
    return float(matches.iloc[-1])


def _return(daily: pd.DataFrame, day: date) -> float | None:
    dates = trading_dates(daily.index)
    if day not in dates:
        return None
    position = dates.index(day)
    if position == 0:
        return None
    previous = _close_on(daily, dates[position - 1])
    current = _close_on(daily, day)
    if previous in (None, 0) or current is None:
        return None
    return current / previous - 1


def _developing_return(daily: pd.DataFrame, day: date) -> float | None:
    row = daily.loc[[stamp for stamp in daily.index if as_date(stamp) == day]]
    if row.empty:
        return None
    open_ = float(row["Open"].iloc[-1])
    close = float(row["Close"].iloc[-1])
    if open_ == 0:
        return None
    return close / open_ - 1
