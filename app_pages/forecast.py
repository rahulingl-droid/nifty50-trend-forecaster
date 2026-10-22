import math

import streamlit as st

from forecaster import charts, features as fe, service
from forecaster.market import session_status, trading_days_ahead

SIGNAL_LABELS = {"Bullish": "▲ Bullish", "Bearish": "▼ Bearish", "Neutral": "– Neutral"}

st.title("Nifty 50 trend forecaster")
st.caption(
    "Machine-learning forecast of Nifty 50 direction from 50+ technical and cross-market indicators. "
    "Delayed public data from Yahoo Finance. For research and education, not investment advice."
)

status = session_status()
with st.container(horizontal=True, vertical_alignment="center"):
    if status.is_open:
        st.badge("Market open · refreshing every minute", icon=":material/sensors:", color="green")
    elif status.state == "pre-open":
        st.badge(status.label, icon=":material/schedule:", color="orange")
    else:
        st.badge(status.label, icon=":material/bedtime:", color="gray")
    st.caption(f"{status.now:%A, %d %b %Y · %H:%M} IST")


def forecast_card(title: str, target: str, probability: float, model, detail: str) -> None:
    label, color, icon = service.outlook(probability)
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(target)
        st.badge(label, icon=icon, color=color)
        st.metric("Probability of an up move", f"{probability:.0%}")
        st.progress(probability)
        st.caption(detail)
        st.caption(
            f"Walk-forward accuracy {model.metrics['accuracy']:.1%} · AUC {model.metrics['auc']:.3f} "
            f"· base rate up {model.metrics['base_rate_up']:.1%}"
        )


def render_live() -> None:
    try:
        snap = service.snapshot()
    except Exception as error:  # Yahoo outages or rate limits
        st.error(f"Could not load market data: {error}", icon=":material/cloud_off:")
        return
    daily, latest, previous = snap.daily, snap.daily.iloc[-1], snap.daily.iloc[-2]
    last_date = daily.index[-1].date()
    bar_state = "live" if snap.status.is_open and last_date == snap.status.now.date() else "last"

    with st.container(horizontal=True):
        st.metric(
            f"Nifty 50 ({bar_state})",
            f"{latest['Close']:,.2f}",
            f"{latest['Close'] / previous['Close'] - 1:+.2%}",
            chart_data=daily["Close"].tail(30).round(2).tolist(),
            border=True,
        )
        st.metric(
            "India VIX",
            f"{latest['vix']:.2f}",
            f"{latest['vix'] / previous['vix'] - 1:+.2%}",
            delta_color="inverse",
            chart_data=daily["vix"].tail(30).round(2).tolist(),
            border=True,
        )
        st.metric("Day range", f"{latest['Low']:,.0f} – {latest['High']:,.0f}", border=True)
        st.metric("ATR (14 sessions)", f"{latest['atr']:,.0f}", f"{latest['atr'] / latest['Close']:.2%} of price",
                  delta_color="off", delta_arrow="off", border=True)

    with st.spinner("Training models on completed sessions (first run takes about a minute)…"):
        models = {h: service.daily_model(h, snap.cutoff) for h in service.HORIZONS}
        hourly = service.intraday_model(snap.intraday_cutoff)

    st.subheader("Forecast")
    close, atr = latest["Close"], latest["atr"]
    cols = st.columns(3)
    for col, (horizon, name) in zip(cols[:2], service.HORIZONS.items()):
        model = models[horizon]
        probability = float(model.predict_proba(snap.daily_X.tail(1))[0])
        target_day = snap.status.next_session if horizon == 1 else trading_days_ahead(last_date, horizon)
        band = atr * math.sqrt(horizon)
        votes = " · ".join(f"{k.replace('_', ' ')} {v:.0%}" for k, v in model.component_probas(snap.daily_X.tail(1)).items())
        with col:
            forecast_card(
                name,
                f"Close on {target_day:%a %d %b} vs {close:,.0f} ({bar_state} close)",
                probability,
                model,
                f"Expected range ±1 ATR: {close - band:,.0f} – {close + band:,.0f}. Ensemble votes: {votes}.",
            )
    with cols[2]:
        intraday = snap.intraday
        bar_time = intraday.index[-1]
        probability = float(hourly.predict_proba(snap.intraday_X.tail(1))[0])
        note = (
            f"Based on the 15-minute bar at {bar_time:%H:%M} IST."
            if snap.status.is_open
            else f"Market closed. Last reading from {bar_time:%d %b %H:%M} IST; resumes {snap.status.next_session:%a %d %b} 09:15."
        )
        forecast_card("Next hour (intraday, experimental)", "Price 4 bars (60 minutes) ahead vs now", probability, hourly, note)

    st.subheader("Price and indicators")
    view = st.segmented_control(
        "Chart", ["Daily", "Today (15-minute)"], default="Daily", required=True, key="chart_view",
        label_visibility="collapsed",
    )
    with st.container(border=True):
        if view == "Daily":
            sessions = st.select_slider("Sessions shown", options=[60, 120, 250, 500], value=120, key="sessions")
            st.altair_chart(charts.price_chart(daily, sessions), alt="Nifty 50 close with 21 and 50 session EMAs")
        else:
            st.altair_chart(charts.intraday_chart(snap.intraday),
                            alt="Today's Nifty 50 15-minute closes with floor pivot, R1 and S1 levels")

    scorecard = fe.indicator_scorecard(daily)
    counts = scorecard["Signal"].value_counts()
    bullish, bearish = int(counts.get("Bullish", 0)), int(counts.get("Bearish", 0))
    with st.container(border=True):
        st.markdown("**Indicator consensus (daily)**")
        with st.container(horizontal=True):
            st.metric("Bullish", bullish, border=True)
            st.metric("Bearish", bearish, border=True)
            st.metric("Neutral", int(counts.get("Neutral", 0)), border=True)
        scorecard["Signal"] = scorecard["Signal"].map(SIGNAL_LABELS)
        st.dataframe(scorecard, hide_index=True, alt="Daily indicator readings with bullish, bearish or neutral signal")
        st.caption("Rule-of-thumb readings shown for context. The ML forecast learns its own weighting of these inputs.")


# Auto-refresh only while the exchange is in session.
live = st.fragment(run_every="60s" if status.is_open else None)(render_live)
live()
