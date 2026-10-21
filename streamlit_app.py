"""Nifty 50 Monday-to-Friday trend forecast."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from forecaster import charts, features as fe
from forecaster.market import session_status
from forecaster.service import outlook, snapshot
from forecaster.week import as_date

st.set_page_config(
    page_title="Nifty 50 week trend",
    page_icon=":material/monitoring:",
    layout="wide",
)

st.title("Nifty 50 week trend")
st.caption(
    "A walk-forward ensemble reads trend, momentum, volatility, floor pivots and "
    "macro context, then updates a Monday-to-Friday direction call from delayed "
    "Yahoo Finance quotes. Research only: this is not a recommendation to buy or sell."
)

status = session_status()
badge_color = {"open": "green", "pre-open": "orange", "closed": "gray"}[status.state]
st.badge(status.label, icon=":material/schedule:", color=badge_color)


def _fmt_feature(value: float, percent: bool) -> str:
    if pd.isna(value):
        return "—"
    if percent:
        return f"{value:+.2%}"
    return f"{value:,.2f}"


def _call(probability: float | None) -> str:
    if probability is None or pd.isna(probability):
        return "No call"
    word, _, _ = outlook(float(probability))
    return f"{word} {float(probability):.0%}"


with st.sidebar:
    st.header("Model card")
    st.caption(
        "Five daily models each answer a different question: is the close higher "
        "1, 2, 3, 4 or 5 sessions ahead? While the market is open, today's card "
        "switches to an intraday model aimed at the session close."
    )
    if st.button("Refresh quotes", icon=":material/refresh:"):
        st.cache_data.clear()


@st.fragment(run_every="60s" if status.is_open else "15m")
def dashboard() -> None:
    try:
        with st.skeleton(height=240):
            snap = snapshot()
    except Exception as exc:  # noqa: BLE001 - show the feed failure, keep the page usable
        st.warning(f"The forecast could not be built. {exc}")
        return

    model = snap.models[1]
    metrics = model.metrics
    last = snap.framed.iloc[-1]
    previous = snap.framed["Close"].iloc[-2]
    known_vix = snap.framed["vix"].dropna() if "vix" in snap.framed.columns else pd.Series(dtype=float)
    change = float(last["Close"]) / float(previous) - 1
    spark = [float(value) for value in snap.framed["Close"].tail(30)]
    friday = snap.week.iloc[-1]
    week_probability = friday["Live call"] if pd.notna(friday["Live call"]) else friday["Monday call"]
    week_word, _, week_icon = outlook(None if pd.isna(week_probability) else float(week_probability))
    skill = metrics.get("roc_auc", float("nan"))
    beats_base = (
        skill == skill
        and skill > 0.53
        and metrics["brier"] < metrics["base_brier"] - 0.005
        and metrics["accuracy"] > metrics["base_accuracy"]
    )

    with st.container(horizontal=True):
        st.metric(
            "Nifty 50",
            f"{float(last['Close']):,.0f}",
            f"{change:+.2%}",
            border=True,
            chart_data=spark,
            chart_type="line",
            icon=":material/show_chart:",
        )
        st.metric(
            "Friday call",
            week_word,
            None if pd.isna(week_probability) else f"{float(week_probability):.0%}",
            delta_description="probability the week finishes higher than the last completed close",
            border=True,
            icon=week_icon,
        )
        st.metric(
            "India VIX",
            "—" if known_vix.empty else f"{float(known_vix.iloc[-1]):.2f}",
            border=True,
            icon=":material/speed:",
        )
        st.metric(
            "Walk-forward AUC",
            "—" if skill != skill else f"{skill:.2f}",
            f"{metrics['accuracy'] - metrics['base_accuracy']:+.1%}",
            delta_description="accuracy versus always picking the more common direction",
            border=True,
            icon=":material/query_stats:",
        )

    if not beats_base:
        st.badge(
            "Holdout skill is weak",
            icon=":material/warning:",
            color="orange",
            help="On unseen sessions the ensemble did not clearly beat a naive baseline. Read the arrows as a model output, not an edge.",
        )

    st.subheader("Monday to Friday")
    st.caption(
        f"Quotes through {as_date(snap.daily.index[-1]).strftime('%d %b %Y')}. "
        "The Monday call is frozen at the last session before this week. "
        "The live call is refreshed from the latest completed bar."
    )
    with st.container(horizontal=True):
        for _, day in snap.week.iterrows():
            probability = day["Live call"] if day["Phase"] != "settled" else day["Monday call"]
            word, _, icon = outlook(None if pd.isna(probability) else float(probability))
            if day["Phase"] == "no session":
                value = "No session"
                icon = ":material/event_busy:"
            elif day["Phase"] == "settled" and pd.notna(day["Return"]):
                value = f"{float(day['Return']):+.2%}"
            elif day["Phase"] == "live" and pd.notna(day["Return"]):
                value = f"{float(day['Return']):+.2%} so far"
            else:
                value = word
            delta = None if pd.isna(probability) else f"{float(probability):.0%} up"
            st.metric(
                f"{day['Weekday'][:3]} {day['Date'][8:]}",
                value,
                delta,
                delta_description=day["Phase"],
                border=True,
                icon=icon,
                help=(
                    f"Monday call {_call(None if pd.isna(day['Monday call']) else float(day['Monday call']))} "
                    f"({day['Monday source']}). Live call {_call(None if pd.isna(day['Live call']) else float(day['Live call']))} "
                    f"({day['Live source']})."
                ),
            )

    price_col, importance_col = st.columns(2)
    with price_col:
        with st.container(border=True):
            st.subheader("Price and trend")
            st.altair_chart(
                charts.price_chart(snap.framed),
                alt="Nifty 50 close with the 21- and 50-session averages",
            )
    with importance_col:
        with st.container(border=True):
            st.subheader("Best indicators")
            ranking = st.segmented_control(
                "Rank indicators by",
                options=[1, 5],
                default=1,
                required=True,
                format_func=lambda horizon: "Next session" if horizon == 1 else "Five sessions",
            )
            ranked = snap.models[int(ranking)]
            st.altair_chart(
                charts.importance_chart(ranked.importance, fe.LABELS),
                alt="Which inputs the random forest relied on most",
            )

    st.subheader("Indicator pivot")
    st.caption(
        "Each row is one of the strongest inputs. Weekday cells are that session's "
        "reading. Bias says whether the latest reading sits closer to the historical "
        "up-day median or the down-day median, using only sessions before this week."
    )
    pivot = fe.indicator_pivot(
        snap.features,
        fe.direction_target(snap.framed["Close"], int(ranking)),
        ranked.importance,
        list(pd.to_datetime(snap.week["Date"])),
    )
    display = pivot.drop(columns=["_percent"]).copy()
    for column in display.columns:
        if column in {"Indicator", "Bias"}:
            continue
        if column == "Importance":
            display[column] = pivot[column].map(lambda value: "—" if pd.isna(value) else f"{float(value):.1%}")
            continue
        display[column] = [
            _fmt_feature(value, bool(percent))
            for value, percent in zip(pivot[column], pivot["_percent"], strict=True)
        ]
    st.dataframe(display, hide_index=True, alt="Best indicators pivoted across this week")

    levels = {
        "S2": float(last["s2"]) if pd.notna(last.get("s2")) else None,
        "S1": float(last["s1"]) if pd.notna(last.get("s1")) else None,
        "Pivot": float(last["pivot"]) if pd.notna(last.get("pivot")) else None,
        "R1": float(last["r1"]) if pd.notna(last.get("r1")) else None,
        "R2": float(last["r2"]) if pd.notna(last.get("r2")) else None,
    }
    st.subheader("Floor pivots")
    st.caption("Classic pivots from the previous session, drawn on the latest 15-minute session.")
    with st.container(horizontal=True):
        for name, level in levels.items():
            if level is None:
                continue
            gap = float(last["Close"]) / level - 1
            st.metric(name, f"{level:,.0f}", f"{gap:+.2%}", delta_description="price versus the level", border=True)
    if not snap.intraday.empty:
        with st.container(border=True):
            st.altair_chart(
                charts.intraday_chart(snap.intraday, levels),
                alt="Latest Nifty 50 session against the floor pivot levels",
            )

    score_col, track_col = st.columns(2)
    with score_col:
        with st.container(border=True):
            st.subheader("Indicator scorecard")
            st.caption("Fixed rules on the latest bar. The ensemble can disagree with any single rule.")
            st.dataframe(
                fe.indicator_scorecard(snap.framed),
                hide_index=True,
                alt="Rule-of-thumb reading for each indicator",
            )
    with track_col:
        with st.container(border=True):
            st.subheader("Walk-forward record")
            st.caption(
                f"Next-session accuracy {metrics['accuracy']:.1%} against a "
                f"{metrics['base_accuracy']:.1%} naive baseline, on {metrics['rows']} unseen sessions. "
                f"Brier {metrics['brier']:.3f} versus {metrics['base_brier']:.3f} for guessing the base rate."
            )
            st.altair_chart(
                charts.probability_chart(model.oos),
                alt="Walk-forward probability of an up move over recent sessions",
            )
            st.altair_chart(
                charts.equity_chart(ml_equity(model)),
                alt="Growth of one rupee for long-or-flat versus buy and hold, before costs",
            )

    if snap.components:
        spread = max(snap.components.values()) - min(snap.components.values())
        names = ", ".join(f"{name.replace('_', ' ')} {prob:.0%}" for name, prob in snap.components.items())
        if spread >= 0.15:
            st.badge("The three models disagree", icon=":material/warning:", color="orange")
        st.caption(f"Latest component probabilities: {names}.")

    with st.expander("How the forecast is built"):
        st.markdown(
            """
            The daily models are a soft vote of a gradient-boosted tree, a shallow
            random forest and a logistic regression. They train only on finished
            Nifty sessions. Accuracy, AUC and the equity curve use later sessions
            the model had not seen yet, with a gap so a multi-day label does not
            leak into the next test window.

            S&P 500 is lagged by one US session because that market is still open
            when the NSE closes. India VIX, Bank Nifty and USD/INR are aligned to
            the same NSE date. Index volume is ignored because the public feed is
            not a reliable share count.

            Nothing here includes brokerage, slippage, or the chance the next
            print revises the last bar. A probability near 50% is the model
            declining to lean.
            """
        )


def ml_equity(model):
    from forecaster.model import equity_curves

    return equity_curves(model.oos)


dashboard()
