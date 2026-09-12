"""Reusable presentation pieces. Pure rendering — no service calls here."""

from __future__ import annotations

from collections.abc import Sequence

import altair as alt
import pandas as pd
import streamlit as st

from airfare.domain.models import Itinerary, ScoredOffer
from airfare.ui import format as fmt

TEAL = "#1A7F8E"
TEAL_DARK = "#0F5561"
TEAL_MID = "#3FA3B1"
AQUA = "#CFE9EC"
AQUA_SOFT = "#E9F5F6"
AMBER = "#C97A00"  # reserved for warnings and "high" only
AMBER_SOFT = "#FBF0D9"
INK = "#23272B"
SLATE = "#6B7280"
LINE = "#E4E8EC"

CSS = f"""
<style>
  .block-container {{ padding-top: 1.2rem; padding-bottom: 4rem; max-width: 1180px; }}
  h1, h2, h3 {{ letter-spacing: -0.02em; color: {INK}; }}
  h1 {{ font-weight: 900; }}
  .afm-hero {{ display: flex; align-items: center; gap: 2rem; padding: 1.2rem 0 1.6rem 0; }}
  .afm-hero h1 {{ font-size: 2.6rem; line-height: 1.05; margin: 0 0 .6rem 0; }}
  .afm-hero p {{ font-size: 1.05rem; color: {SLATE}; max-width: 34rem; margin: 0; }}
  .afm-hero .afm-hex {{ margin-left: auto; flex: 0 0 auto; }}
  .afm-eyebrow {{ font-size: .76rem; text-transform: uppercase; letter-spacing: .1em; color: {TEAL}; font-weight: 700; }}
  .afm-price {{ font-size: 1.8rem; font-weight: 900; line-height: 1.1; color: {INK}; }}
  .afm-muted {{ color: {SLATE}; font-size: .86rem; }}
  .afm-chip {{ display: inline-block; padding: .14rem .6rem; border-radius: 999px; font-size: .72rem;
              font-weight: 700; margin-right: .3rem; letter-spacing: .01em; }}
  .afm-chip.rec     {{ background: {TEAL}; color: #fff; }}
  .afm-chip.live    {{ background: rgba(26,127,142,.12); color: {TEAL_DARK}; }}
  .afm-chip.low     {{ background: rgba(26,127,142,.12); color: {TEAL_DARK}; }}
  .afm-chip.typical {{ background: #EEF1F4; color: #4B5563; }}
  .afm-chip.high    {{ background: {AMBER_SOFT}; color: {AMBER}; }}
  .afm-chip.warn    {{ background: {AMBER_SOFT}; color: {AMBER}; }}
  .afm-chip.demo    {{ background: {AQUA}; color: {TEAL_DARK}; }}
  .afm-chip.model   {{ background: rgba(26,127,142,.12); color: {TEAL_DARK}; }}
  .afm-seg {{ font-variant-numeric: tabular-nums; }}
  div[data-testid="stMetric"] {{ background: #FFFFFF; border: 1px solid {LINE}; border-radius: .8rem; padding: .7rem .9rem; }}
  div[data-testid="stMetricLabel"] {{ color: {SLATE}; }}
  div[data-testid="stMetricValue"] {{ font-weight: 900; }}
  section[data-testid="stSidebar"] .afm-brand {{ font-weight: 900; font-size: 1.15rem; letter-spacing: -0.02em; }}
  .afm-hexrow {{ display:flex; gap:.35rem; margin:.4rem 0 .8rem 0; }}
</style>
"""

HEX_MARK = f"""
<svg width="190" height="170" viewBox="0 0 190 170" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <polygon points="60,4 108,4 132,46 108,88 60,88 36,46" fill="{TEAL_MID}"/>
  <polygon points="112,50 160,50 184,92 160,134 112,134 88,92" fill="{TEAL_DARK}"/>
  <polygon points="36,90 84,90 108,132 84,174 36,174 12,132" fill="{AQUA}"/>
  <polygon points="8,20 34,20 47,42 34,64 8,64 -5,42" fill="{AQUA_SOFT}"/>
</svg>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, lead: str) -> None:
    st.markdown(
        f"<div class='afm-hero'><div><h1>{title}</h1><p>{lead}</p></div>"
        f"<div class='afm-hex'>{HEX_MARK}</div></div>",
        unsafe_allow_html=True,
    )


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
                scale=alt.Scale(range=[TEAL_DARK, "#B8C0C8", TEAL_MID]),
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
        .mark_bar(color=TEAL, size=14, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
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
        .mark_bar(color="#B8C0C8", opacity=0.9)
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
        .mark_rule(color=TEAL, strokeWidth=2)
        .encode(x="usd:Q")
    )
    return base + rule
