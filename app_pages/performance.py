import streamlit as st

from forecaster import charts, model as ml, service

st.title("Model performance")
st.caption(
    "Every number here is out-of-sample: the series is split into five consecutive folds and each fold "
    "is predicted by a model trained only on earlier data, with a purge gap equal to the horizon."
)

choice = st.segmented_control(
    "Model",
    ["Next session", "Next 5 sessions", "Next hour (intraday)"],
    default="Next session",
    required=True,
    key="perf_model",
)

try:
    snap = service.snapshot()
except Exception as error:
    st.error(f"Could not load market data: {error}", icon=":material/cloud_off:")
    st.stop()

with st.spinner("Training models (first run takes about a minute)…"):
    if choice == "Next hour (intraday)":
        model = service.intraday_model(snap.intraday_cutoff)
        unit, history = "15-minute bars", 400
    else:
        model = service.daily_model(1 if choice == "Next session" else 5, snap.cutoff)
        unit, history = "sessions", 250

m = model.metrics
with st.container(horizontal=True):
    st.metric("Accuracy", f"{m['accuracy']:.1%}", f"{m['accuracy'] - max(m['base_rate_up'], 1 - m['base_rate_up']):+.1%}",
              delta_description="vs always guessing the majority side", border=True)
    st.metric("ROC AUC", f"{m['auc']:.3f}", f"{m['auc'] - 0.5:+.3f}", delta_description="vs coin flip", border=True)
    st.metric("Accuracy when confident", f"{m['confident_accuracy']:.1%}",
              delta_description=f"probability beyond 45–55%, {m['confident_share']:.0%} of calls", border=True)
    st.metric("Long-or-flat return", f"{m['strategy_return']:+.0%}",
              f"{m['strategy_return'] - m['buy_hold_return']:+.0%}", delta_description="vs buy and hold", border=True)

st.caption(f"{m['samples']:,} out-of-sample {unit}. Returns ignore costs, slippage and taxes.")

left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.markdown("**Equity curve, out-of-sample**")
        st.altair_chart(charts.equity_chart(ml.equity_curves(model.oos, model.horizon)),
                        alt="Growth of one rupee for the model's long-or-flat strategy versus buy and hold")
with right:
    with st.container(border=True):
        st.markdown(f"**Predicted probability, last {history} {unit}**")
        st.altair_chart(charts.probability_chart(model.oos, history),
                        alt="Out-of-sample probability of an up move over time")

left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.markdown("**Most influential indicators (random forest)**")
        st.altair_chart(charts.importance_chart(model.importance), alt="Top 15 features by importance")
with right:
    with st.container(border=True):
        st.markdown("**Recent out-of-sample calls**")
        recent = model.oos.tail(30).iloc[::-1].copy()
        recent["Outcome"] = recent["actual"].map({1: "▲ Up", 0: "▼ Down"})
        recent["Call"] = recent["predicted"].map({1: "▲ Up", 0: "▼ Down"})
        recent["Correct"] = recent["actual"] == recent["predicted"]
        st.dataframe(
            recent[["proba", "Call", "Outcome", "forward_return", "Correct"]],
            column_config={
                "proba": st.column_config.ProgressColumn("P(up)", format="percent", min_value=0, max_value=1),
                "forward_return": st.column_config.NumberColumn("Realised move", format="percent"),
            },
            alt="Recent out-of-sample predictions versus outcomes",
        )
