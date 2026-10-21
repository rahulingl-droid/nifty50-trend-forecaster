"""NSE session clock (Monday to Friday, 09:15 to 15:30 IST).

Exchange holidays are not modelled. On a holiday the clock can still say the
market is open, and the quote feed simply has no new bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
PRE_OPEN = time(9, 0)
OPEN = time(9, 15)
CLOSE = time(15, 30)


@dataclass(frozen=True)
class SessionStatus:
    state: str
    label: str
    now: datetime
    next_session: datetime

    @property
    def is_open(self) -> bool:
        return self.state == "open"


def next_weekday(day: date) -> date:
    nxt = day + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt


def _at_open(day: date) -> datetime:
    return datetime.combine(day, OPEN, tzinfo=IST)


def session_status(now: datetime | None = None) -> SessionStatus:
    current = (now or datetime.now(IST)).astimezone(IST)
    today = current.date()
    clock = current.timetz().replace(tzinfo=None)
    if today.weekday() < 5 and OPEN <= clock < CLOSE:
        return SessionStatus("open", "Market open", current, _at_open(today))
    if today.weekday() < 5 and PRE_OPEN <= clock < OPEN:
        return SessionStatus("pre-open", "Pre-open session", current, _at_open(today))
    if today.weekday() < 5 and clock < OPEN:
        nxt = _at_open(today)
        label = "Market opens at 09:15 IST"
    else:
        nxt = _at_open(next_weekday(today))
        label = "Market closed"
    return SessionStatus("closed", label, current, nxt)


def trading_days_ahead(start: date, count: int) -> list[date]:
    days: list[date] = []
    cursor = start
    while len(days) < count:
        cursor = next_weekday(cursor)
        days.append(cursor)
    return days


def week_dates(day: date) -> list[date]:
    monday = day - timedelta(days=day.weekday())
    return [monday + timedelta(days=offset) for offset in range(5)]
