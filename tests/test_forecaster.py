"""Offline tests on synthetic prices (no network)."""

from datetime import datetime

import numpy as np
import pandas as pd

from forecaster import features as fe
from forecaster import indicators as ind
from forecaster import model as ml
from forecaster.market import IST, session_status, trading_days_ahead
from forecaster.data import stitch_daily
from forecaster.week import project_sessions, sessions_between, trading_dates


def synthetic_daily(n: int = 900, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2018-01-01", periods=n)
    returns = rng.normal(0.0003, 0.01, n)
    close = 15000 * np.exp(np.cumsum(returns))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(np.maximum(open_, close) * (1 + rng.uniform(0.0, 0.008, n)), np.maximum(open_, close))
    low = np.minimum(np.minimum(open_, close) * (1 - rng.uniform(0.0, 0.008, n)), np.minimum(open_, close))
    frame = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=index)
    frame["vix"] = np.clip(15 + np.cumsum(rng.normal(0, 0.2, n)), 9, 40)
    frame["banknifty"] = close * (2.1 + np.cumsum(rng.normal(0, 0.001, n)))
    frame["sp500"] = 4000 * np.exp(np.cumsum(rng.normal(0.0002, 0.008, n)))
    frame["usdinr"] = np.clip(75 + np.cumsum(rng.normal(0, 0.05, n)), 60, 100)
    return frame


def test_rsi_bounds_and_supertrend_direction():
    frame = synthetic_daily()
    rsi = ind.rsi(frame["Close"]).dropna()
    assert rsi.between(0, 100).all()
    direction = ind.supertrend(frame)["supertrend_dir"].iloc[20:]
    assert set(np.unique(direction)).issubset({-1.0, 1.0})
    assert set(np.unique(direction)) == {-1.0, 1.0}


def test_features_are_causal():
    raw = synthetic_daily()
    base = fe.daily_features(fe.daily_frame(raw))
    tweaked = raw.copy()
    tweaked.iloc[-1, tweaked.columns.get_loc("Close")] *= 1.08
    tweaked.iloc[-1, tweaked.columns.get_loc("High")] *= 1.08
    alt = fe.daily_features(fe.daily_frame(tweaked))
    pd.testing.assert_frame_equal(base.iloc[:-1], alt.iloc[:-1])


def test_direction_target_masks_future_and_sessions():
    morning = pd.date_range("2026-01-05 15:00", periods=3, freq="15min")
    nxt = pd.date_range("2026-01-06 09:15", periods=5, freq="15min")
    index = morning.append(nxt)
    close = pd.Series(np.arange(len(index), dtype=float) + 1, index=index)
    target = fe.direction_target(close, horizon=2, same_session=True)
    assert target.iloc[-2:].isna().all()
    # 15:30 looks two bars ahead, into the next session.
    assert np.isnan(target.iloc[2])
    assert target.iloc[0] == 1


def test_training_produces_probabilities():
    framed = fe.daily_frame(synthetic_daily())
    model_features = fe.daily_features(framed)
    close = framed["Close"]
    trained = ml.train(
        model_features,
        fe.direction_target(close, 1),
        close.shift(-1) / close - 1,
        horizon=1,
    )
    probability = trained.predict_proba(model_features.dropna().tail(1))
    assert 0.0 <= probability <= 1.0
    assert "accuracy" in trained.metrics
    assert set(trained.component_probas(model_features.dropna().tail(1))) == {
        "gradient_boosting",
        "random_forest",
        "logistic",
    }


def test_session_clock():
    opened = session_status(datetime(2026, 1, 7, 10, 0, tzinfo=IST))
    assert opened.is_open
    assert opened.state == "open"

    preopen = session_status(datetime(2026, 1, 7, 9, 5, tzinfo=IST))
    assert preopen.state == "pre-open"
    assert not preopen.is_open

    friday_close = session_status(datetime(2026, 1, 9, 15, 30, tzinfo=IST))
    assert not friday_close.is_open
    assert friday_close.state == "closed"
    assert friday_close.next_session.weekday() == 0
    assert friday_close.next_session.date() == trading_days_ahead(friday_close.now.date(), 1)[0]


def test_stitch_appends_only_missing_sessions():
    daily = pd.DataFrame(
        {"Open": [100.0], "High": [110.0], "Low": [90.0], "Close": [105.0], "vix": [14.0]},
        index=pd.to_datetime(["2026-01-05"]),
    )
    bars = pd.DataFrame(
        {
            "Open": [100.0, 106.0, 120.0],
            "High": [108.0, 112.0, 130.0],
            "Low": [99.0, 104.0, 118.0],
            "Close": [107.0, 105.0, 125.0],
        },
        index=pd.to_datetime(["2026-01-05 09:15", "2026-01-05 15:15", "2026-01-06 15:15"]),
    )
    stitched = stitch_daily(daily, bars)
    assert list(stitched.index.strftime("%Y-%m-%d")) == ["2026-01-05", "2026-01-06"]
    assert stitched.loc["2026-01-05", "Open"] == 100.0
    assert stitched.loc["2026-01-06", "Close"] == 125.0
    assert pd.isna(stitched.loc["2026-01-06", "vix"])


def test_week_horizon_counts_sessions():
    index = pd.bdate_range("2026-01-02", periods=6)
    days = trading_dates(index)
    assert sessions_between(days, days[0], days[1]) == 1
    assert sessions_between(days, days[0], days[5]) == 5
    projected = project_sessions(days[:3], days[5])
    assert projected[-1] == days[5]
