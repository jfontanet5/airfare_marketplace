"""Reusable presentation pieces. Pure rendering — no service calls here."""

from __future__ import annotations

from collections.abc import Sequence

import altair as alt
import pandas as pd
import streamlit as st

from airfare.domain.models import Itinerary, ScoredOffer
from airfare.ui import format as fmt

CSS = """
<style>
  .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px; }
  h1 { letter-spacing: -0.02em; }
  .afm-eyebrow { font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; opacity: .65; }
  .afm-price { font-size: 1.75rem; font-weight: 700; line-height: 1.1; }
  .afm-muted { opacity: .7; font-size: .85rem; }
  .afm-chip { display: inline-block; padding: .12rem .55rem; border-radius: 999px; font-size: .74rem;
              font-weight: 600; margin-right: .3rem; border: 1px solid transparent; }
  .afm-chip.rec  { background: rgba(15,98,254,.12); color: #0F62FE; }
  .afm-chip.low  { background: rgba(16,185,129,.14); color: #047857; }
  .afm-chip.typical { background: rgba(107,114,128,.14); color: #374151; }
  .afm-chip.high { background: rgba(245,158,11,.16); color: #92400E; }
  .afm-chip.warn { background: rgba(245,158,11,.16); color: #92400E; }
  .afm-chip.demo { background: rgba(139,92,246,.14); color: #5B21B6; }
  .afm-chip.live { background: rgba(16,185,129,.14); color: #047857; }
  .afm-seg { font-variant-numeric: tabular-nums; }
  div[data-testid="stMetric"] { padding: .4rem .6rem; border-radius: 10px; }
  @media (prefers-color-scheme: dark) {
    .afm-chip.typical { color: #D1D5DB; }
    .afm-chip.low { color: #6EE7B7; }
    .afm-chip.high, .afm-chip.warn { color: #FCD34D; }
    .afm-chip.demo { color: #C4B5FD; }
    .afm-chip.rec { color: #93C5FD; }
  }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def chip(text: str, kind: str = "typical") -> str:
    return f"<span class='afm-chip {kind}'>{text}</span>"


def eyebrow(text: str) -> None:
    st.markdown(f"<div class='afm-eyebrow'>{text}</div>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str, chips: Sequence[str] = ()) -> None:
    st.title(title)
    line = f"<span class='afm-muted'>{subtitle}</span>"
    if chips:
        line += " &nbsp; " + " ".join(chips)
    st.markdown(line, unsafe_allow_html=True)


# --------------------------------------------------------------------------- itineraries


def itinerary_summary(it: Itinerary) -> str:
    """'7:00 AM → 10:45 AM · SJU → JFK · Nonstop · 3h 45m'"""
    return " · ".join(
        [
            f"{fmt.clock(it.dep_at)} → {fmt.clock(it.arr_at)}",
            fmt.route_line(it),
            fmt.stops_label(it.stops),
            fmt.hours(it.duration_minutes),
        ]
    )


def itinerary_detail(it: Itinerary) -> None:
    layovers = it.layovers()
    for i, s in enumerate(it.segments):
        carrier = s.carrier_name or s.carrier_code or ""
        flight = f"{s.carrier_code or ''} {s.flight_number or ''}".strip()
        aircraft = f" · {s.aircraft_code}" if s.aircraft_code else ""
        st.markdown(
            f"<div class='afm-seg'><b>{s.origin}</b> {fmt.clock(s.dep_at)} → "
            f"<b>{s.destination}</b> {fmt.clock(s.arr_at)}"
            f"<span class='afm-muted'> · {carrier} {flight}{aircraft}</span></div>",
            unsafe_allow_html=True,
        )
        if i < len(layovers):
            st.markdown(
                f"<div class='afm-muted'>&nbsp;&nbsp;⟳ {fmt.delta(layovers[i])} layover in {s.destination}</div>",
                unsafe_allow_html=True,
            )


def offer_card(
    s: ScoredOffer,
    *,
    recommended: bool,
    drop_prob: float | None,
    position: str | None,
    model_is_demo: bool,
) -> None:
    o = s.offer
    with st.container(border=True):
        left, right = st.columns([3.2, 1], vertical_alignment="top")
        with left:
            chips = []
            if recommended:
                chips.append(chip("Recommended", "rec"))
            if position:
                chips.append(chip(f"{position} for this route", position))
            if o.price.usd is None:
                chips.append(chip("native currency", "warn"))
            st.markdown(f"**{o.airline_display}** &nbsp; {' '.join(chips)}", unsafe_allow_html=True)
            if o.outbound:
                st.markdown(f"**Out** &nbsp; {itinerary_summary(o.outbound)}")
            if o.inbound:
                st.markdown(f"**Back** &nbsp; {itinerary_summary(o.inbound)}")
            elif o.return_date:
                st.markdown(
                    f"<span class='afm-muted'>Back {o.return_date:%a %b %-d} · leg chosen at booking · "
                    "price is the round-trip total</span>",
                    unsafe_allow_html=True,
                )
            with st.expander("Flight details and scoring"):
                if o.outbound:
                    st.caption("Outbound")
                    itinerary_detail(o.outbound)
                if o.inbound:
                    st.caption("Return")
                    itinerary_detail(o.inbound)
                st.caption(f"Score {s.score:,.0f} — fare plus penalties, lower is better")
                st.markdown("  \n".join(f"· {r}" for r in s.reasons).replace("$", "\\$"))
        with right:
            st.markdown(f"<div class='afm-price'>{fmt.money(o)}</div>", unsafe_allow_html=True)
            if o.price.currency != "USD" and o.price.usd is not None:
                st.markdown(
                    f"<span class='afm-muted'>{o.price.amount:,.2f} {o.price.currency} @ {o.price.fx_rate:.4f}</span>",
                    unsafe_allow_html=True,
                )
            if drop_prob is not None:
                label = "demo" if model_is_demo else "model"
                st.markdown(
                    f"<span class='afm-muted'>≥5% drop in 7d: <b>{drop_prob:.0%}</b> "
                    f"<span class='afm-chip {label}'>{label}</span></span>",
                    unsafe_allow_html=True,
                )
            st.caption(
                f"{o.departure_date:%a %b %-d}"
                + (f" – {o.return_date:%b %-d}" if o.return_date else "")
            )


# --------------------------------------------------------------------------- charts


def trend_chart(trend: pd.DataFrame, height: int = 260) -> alt.Chart | None:
    """Daily min / median / best-so-far over search days."""
    if trend.empty:
        return None
    df = trend.copy()
    df["best_so_far"] = df["min_price_usd"].cummin()
    long = df.melt(
        "search_day", ["min_price_usd", "median_price_usd", "best_so_far"], "series", "usd"
    )
    names = {
        "min_price_usd": "Daily low",
        "median_price_usd": "Daily median",
        "best_so_far": "Best so far",
    }
    long["series"] = long["series"].map(names)
    return (
        alt.Chart(long)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("search_day:T", title="Search day", axis=alt.Axis(format="%b %d")),
            y=alt.Y("usd:Q", title="USD", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "series:N",
                title=None,
                scale=alt.Scale(range=["#0F62FE", "#9CA3AF", "#10B981"]),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=[
                alt.Tooltip("search_day:T", title="Day"),
                "series:N",
                alt.Tooltip("usd:Q", format="$,.0f"),
            ],
        )
        .properties(height=height)
    )


def fare_calendar_chart(by_departure: pd.DataFrame, height: int = 220) -> alt.Chart | None:
    """Cheapest observed fare per departure date."""
    if by_departure.empty:
        return None
    df = by_departure.copy()
    df["departure_date"] = pd.to_datetime(df["departure_date"])
    return (
        alt.Chart(df)
        .mark_bar(color="#0F62FE", size=14, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("departure_date:T", title="Departure date", axis=alt.Axis(format="%b %d")),
            y=alt.Y("min_price_usd:Q", title="Cheapest observed (USD)"),
            tooltip=[
                alt.Tooltip("departure_date:T", title="Departs"),
                alt.Tooltip("min_price_usd:Q", title="Cheapest", format="$,.0f"),
                alt.Tooltip("last_seen:N", title="Last seen"),
            ],
        )
        .properties(height=height)
    )


def price_histogram(
    prices: pd.Series, current: float | None = None, height: int = 180
) -> alt.Chart:
    df = pd.DataFrame({"usd": prices})
    base = (
        alt.Chart(df)
        .mark_bar(color="#9CA3AF", opacity=0.8)
        .encode(
            x=alt.X("usd:Q", bin=alt.Bin(maxbins=25), title="Observed fare (USD)"),
            y=alt.Y("count()", title="Observations"),
        )
        .properties(height=height)
    )
    if current is None:
        return base
    rule = (
        alt.Chart(pd.DataFrame({"usd": [current]}))
        .mark_rule(color="#0F62FE", strokeWidth=2)
        .encode(x="usd:Q")
    )
    return base + rule
