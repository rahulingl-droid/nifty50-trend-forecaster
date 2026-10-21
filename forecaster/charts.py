"""Altair charts. Colors follow the financial dashboard theme."""

from __future__ import annotations

import altair as alt
import pandas as pd

SERIES = ["#60A5FA", "#34D399", "#A78BFA", "#F87171", "#FBBF24", "#38BDF8"]
MUTED = "#94A3B8"


def _lines(frame: pd.DataFrame, y_format: str, height: int = 320) -> alt.Chart:
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["Date"], empty=False)
    lines = (
        alt.Chart(frame)
        .mark_line()
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Value:Q", title=None, axis=alt.Axis(format=y_format), scale=alt.Scale(zero=False)),
            color=alt.Color("Series:N", scale=alt.Scale(range=SERIES), title=None, legend=alt.Legend(orient="top")),
            tooltip=["Date:T", "Series:N", alt.Tooltip("Value:Q", format=y_format)],
        )
    )
    points = lines.mark_point(size=40).encode(opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
    rule = (
        alt.Chart(frame)
        .mark_rule(color=MUTED)
        .encode(x="Date:T", opacity=alt.condition(nearest, alt.value(0.4), alt.value(0)))
        .add_params(nearest)
    )
    return (lines + points + rule).properties(height=height)


def price_chart(frame: pd.DataFrame) -> alt.Chart:
    plot = frame.tail(180).rename(columns={"Close": "Nifty 50", "ema_21": "EMA 21", "ema_50": "EMA 50"})
    long = plot[["Nifty 50", "EMA 21", "EMA 50"]].rename_axis("Date").reset_index().melt("Date", var_name="Series", value_name="Value")
    return _lines(long, ",.0f")


def intraday_chart(bars: pd.DataFrame, levels: dict[str, float]) -> alt.Chart:
    day = bars.loc[pd.DatetimeIndex(bars.index).date == pd.DatetimeIndex(bars.index).date.max()].copy()
    long = day[["Close"]].rename(columns={"Close": "Nifty 50"}).rename_axis("Time").reset_index().melt("Time", var_name="Series", value_name="Value")
    price = (
        alt.Chart(long)
        .mark_line(color=SERIES[0])
        .encode(
            x=alt.X("Time:T", title="Time (IST)"),
            y=alt.Y("Value:Q", title=None, scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("Time:T", title="Time"), alt.Tooltip("Value:Q", format=",.0f")],
        )
    )
    rules = []
    for name, value in levels.items():
        if value is None or pd.isna(value):
            continue
        band = pd.DataFrame({"Value": [value]})
        rules.append(alt.Chart(band).mark_rule(strokeDash=[4, 4], color=MUTED).encode(y="Value:Q"))
        rules.append(
            alt.Chart(pd.DataFrame({"Value": [value], "Level": [name]})).mark_text(align="left", dx=4, color=MUTED).encode(y="Value:Q", text="Level:N")
        )
    chart = price
    for layer in rules:
        chart = chart + layer
    return chart.properties(height=320)


def equity_chart(curves: pd.DataFrame) -> alt.Chart:
    long = curves.rename_axis("Date").reset_index().melt("Date", var_name="Series", value_name="Value")
    return _lines(long, ",.2f")


def importance_chart(importance: pd.Series, labels: dict[str, str]) -> alt.Chart:
    top = importance.head(10)
    frame = pd.DataFrame({"Feature": [labels.get(name, name) for name in top.index], "Importance": top.to_numpy()})
    return (
        alt.Chart(frame)
        .mark_bar(color=SERIES[0])
        .encode(
            x=alt.X("Importance:Q", title=None, axis=alt.Axis(format=".0%")),
            y=alt.Y("Feature:N", sort="-x", title=None),
            tooltip=[alt.Tooltip("Feature:N"), alt.Tooltip("Importance:Q", format=".1%")],
        )
        .properties(height=320)
    )


def probability_chart(oos: pd.DataFrame) -> alt.Chart:
    plot = oos.tail(180)
    frame = pd.DataFrame(
        {"Date": plot.index, "Probability of an up move": plot["proba"].to_numpy(), "Coin flip": 0.5}
    )
    long = frame.melt("Date", var_name="Series", value_name="Value")
    return _lines(long, ".0%")
