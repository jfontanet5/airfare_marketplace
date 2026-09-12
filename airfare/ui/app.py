"""Airfare Marketplace — Streamlit UI.

Thin presentation layer over :class:`airfare.services.search.SearchService`.
"""

from __future__ import annotations

from datetime import date, timedelta

import altair as alt
import pandas as pd
import streamlit as st

from airfare.domain.models import Itinerary, Offer, ScoredOffer, SearchQuery
from airfare.domain.scoring import ScoringWeights
from airfare.ml.predict import price_position
from airfare.providers import ProviderError, ProviderName, available_providers, build_provider
from airfare.services.search import SearchResult, SearchService
from airfare.ui import bootstrap
from airfare.ui import format as fmt

st.set_page_config(page_title="Airfare Marketplace", page_icon="✈️", layout="wide")
bootstrap.configure_logging()

STOP_OPTIONS = {"Nonstop only": 0, "Up to 1 stop": 1, "Up to 2 stops": 2}
PROVIDER_LABELS = {
    ProviderName.MOCK: "Demo data (offline)",
    ProviderName.REPLAY: "Replay last stored search",
    ProviderName.AMADEUS: "Amadeus (live)",
}

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; max-width: 1200px; }
      .afm-card { border: 1px solid #E3E8EF; border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: .75rem; background: #fff; }
      .afm-card.rec { border-color: #0F62FE; box-shadow: 0 0 0 2px rgba(15,98,254,.08); }
      .afm-price { font-size: 1.6rem; font-weight: 700; }
      .afm-muted { color: #6B7280; font-size: .85rem; }
      .afm-badge { display:inline-block; padding: .1rem .5rem; border-radius: 999px; font-size:.75rem; background:#EEF2FF; color:#3730A3; margin-right:.25rem; }
      .afm-badge.warn { background:#FEF3C7; color:#92400E; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- header
settings = bootstrap.settings()
st.title("✈️ Airfare Marketplace")
st.caption("Provider-agnostic search · USD-normalized history · price-drop signal")

# --------------------------------------------------------------------------- search form
providers = available_providers(settings)
with st.form("search", border=True):
    c1, c2, c3 = st.columns([1.4, 1.4, 1])
    airport_list = bootstrap.airports(include_full=ProviderName.AMADEUS in providers)
    labels = {a.label: a.iata for a in airport_list}
    with c1:
        origin_label = st.selectbox(
            "From", options=list(labels), index=None, placeholder="City or airport code"
        )
    with c2:
        dest_label = st.selectbox(
            "To", options=list(labels), index=None, placeholder="City or airport code"
        )
    with c3:
        trip = st.radio("Trip", ["Roundtrip", "One-way"], horizontal=True)

    d1, d2, d3, d4 = st.columns(4)
    today = date.today()
    with d1:
        dep = st.date_input("Depart", value=today + timedelta(days=21), min_value=today)
    with d2:
        ret = st.date_input(
            "Return", value=dep + timedelta(days=5), min_value=dep, disabled=trip == "One-way"
        )
    with d3:
        pax = st.number_input("Passengers", min_value=1, max_value=9, value=1)
    with d4:
        stops_label = st.selectbox("Stops", list(STOP_OPTIONS), index=1)

    e1, e2, e3 = st.columns([1, 1, 1.2])
    with e1:
        flexible = st.checkbox("Flexible dates (±3 days)")
    with e2:
        airlines_raw = st.text_input("Airlines (IATA codes, optional)", placeholder="B6, AA, DL")
    with e3:
        provider_name = st.selectbox(
            "Data source", providers, format_func=lambda p: PROVIDER_LABELS[p], index=0
        )
    submitted = st.form_submit_button("Search fares", type="primary", width="stretch")


# --------------------------------------------------------------------------- render helpers
def _itinerary_block(title: str, it: Itinerary) -> None:
    st.markdown(
        f"**{title}** · {fmt.route_line(it)} · {fmt.stops_label(it.stops)} · {fmt.hours(it.duration_minutes)}"
    )
    for i, s in enumerate(it.segments):
        carrier = s.carrier_name or s.carrier_code or ""
        st.markdown(
            f"<span class='afm-muted'>{carrier} {s.flight_number or ''} · {s.origin} {fmt.clock(s.dep_at)} → "
            f"{s.destination} {fmt.clock(s.arr_at)}</span>",
            unsafe_allow_html=True,
        )
        layovers = it.layovers()
        if i < len(layovers):
            st.markdown(
                f"<span class='afm-muted'>&nbsp;&nbsp;↳ layover {fmt.delta(layovers[i])} in {s.destination}</span>",
                unsafe_allow_html=True,
            )


def _offer_card(
    s: ScoredOffer, recommended: bool, drop_prob: float | None, position: str | None
) -> None:
    o = s.offer
    css = "afm-card rec" if recommended else "afm-card"
    with st.container():
        st.markdown(f"<div class='{css}'>", unsafe_allow_html=True)
        left, right = st.columns([3, 1])
        with left:
            badges = []
            if recommended:
                badges.append("<span class='afm-badge'>Recommended</span>")
            if position:
                badges.append(f"<span class='afm-badge'>{position} for this route</span>")
            if o.price.usd is None:
                badges.append("<span class='afm-badge warn'>native currency</span>")
            st.markdown(" ".join(badges) + f" **{o.airline_display}**", unsafe_allow_html=True)
            if o.outbound:
                _itinerary_block("Outbound", o.outbound)
            if o.inbound:
                _itinerary_block("Return", o.inbound)
        with right:
            st.markdown(f"<div class='afm-price'>{fmt.money(o)}</div>", unsafe_allow_html=True)
            if o.price.currency != "USD" and o.price.usd is not None:
                st.markdown(
                    f"<span class='afm-muted'>{o.price.amount:,.2f} {o.price.currency} @ {o.price.fx_rate:.4f}</span>",
                    unsafe_allow_html=True,
                )
            if drop_prob is not None:
                st.markdown(
                    f"<span class='afm-muted'>Chance of ≥5% drop in 7 days: <b>{drop_prob:.0%}</b></span>",
                    unsafe_allow_html=True,
                )
            with st.expander("Why this score"):
                st.write(f"Score {s.score:,.0f} (fare + penalties; lower is better)")
                for r in s.reasons:
                    st.write(f"· {r}")
        st.markdown("</div>", unsafe_allow_html=True)


def _trend_chart(origin: str, destination: str, dep_date: date) -> None:
    trend = bootstrap.history().daily_min_trend(origin, destination, dep_date)
    if trend.empty:
        st.info("No price history for this route and date yet — every search adds to it.")
        return
    trend["best_so_far"] = trend["min_price_usd"].cummin()
    long = trend.melt(
        "search_day", ["min_price_usd", "median_price_usd", "best_so_far"], "series", "usd"
    )
    names = {
        "min_price_usd": "Daily min",
        "median_price_usd": "Daily median",
        "best_so_far": "Best so far",
    }
    long["series"] = long["series"].map(names)
    chart = (
        alt.Chart(long)
        .mark_line(point=True)
        .encode(
            x=alt.X("search_day:T", title="Search day"),
            y=alt.Y("usd:Q", title="USD", scale=alt.Scale(zero=False)),
            color=alt.Color("series:N", title=None),
            tooltip=["search_day:T", "series:N", alt.Tooltip("usd:Q", format="$,.0f")],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, width="stretch")


def _render(result: SearchResult) -> None:
    for w in result.warnings:
        st.warning(w)
    if not result.offers:
        st.info("No fares matched. Try widening stops, dates, or airlines.")
        return

    history = bootstrap.history()
    stats = history.route_price_stats(result.query.origin, result.query.destination)
    predictor = bootstrap.predictor()
    rec = result.recommended
    cheapest = result.cheapest
    priced = [o.price.usd for o in result.offers if o.price.usd is not None]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Options", len(result.offers))
    k2.metric("Cheapest", fmt.usd(cheapest.price.usd) if cheapest else "—")
    k3.metric("Median", fmt.usd(float(pd.Series(priced).median())) if priced else "—")
    if rec and predictor:
        k4.metric("Drop chance (recommended)", f"{predictor.probability(rec.offer):.0%}")
        if predictor.card.is_synthetic:
            k4.caption("synthetic-data demo model")
    else:
        k4.metric("Drop chance", "—")
        k4.caption("no model loaded — run `make train`")

    tab_results, tab_trend, tab_table = st.tabs(["Results", "Route trend", "Table"])
    with tab_results:
        sort = st.radio(
            "Sort by",
            ["Recommended", "Price", "Duration"],
            horizontal=True,
            label_visibility="collapsed",
        )
        ranked = list(result.ranked)
        if sort == "Price":
            ranked.sort(
                key=lambda s: s.offer.price.usd if s.offer.price.usd is not None else float("inf")
            )
        elif sort == "Duration":
            ranked.sort(key=lambda s: s.offer.total_duration_minutes or 10**9)
        for s in ranked[:25]:
            prob = (
                predictor.probability(s.offer)
                if predictor and s.offer.price.usd is not None
                else None
            )
            pos = (
                price_position(s.offer.price.usd, stats) if s.offer.price.usd is not None else None
            )
            _offer_card(
                s,
                recommended=rec is not None and s.offer.signature == rec.offer.signature,
                drop_prob=prob,
                position=pos,
            )
    with tab_trend:
        _trend_chart(result.query.origin, result.query.destination, result.query.departure_date)
        if stats:
            st.caption(
                f"{int(stats['count'])} observations on this route · min {fmt.usd(stats['min'])} · "
                f"median {fmt.usd(stats['median'])} · max {fmt.usd(stats['max'])}"
            )
    with tab_table:
        st.dataframe(_table(result.offers), width="stretch", hide_index=True)
    st.caption(
        f"Source: {PROVIDER_LABELS[ProviderName(result.provider)]} · searched {result.searched_at:%Y-%m-%d %H:%M} UTC · "
        f"{result.observations_persisted} observations saved"
    )


def _table(offers: list[Offer]) -> pd.DataFrame:
    rows = []
    for o in offers:
        out = o.outbound
        rows.append(
            {
                "Airline": o.airline_display,
                "Depart": o.departure_date,
                "Return": o.return_date,
                "Route": fmt.route_line(out) if out else "",
                "Dep": fmt.clock(out.dep_at) if out else "",
                "Arr": fmt.clock(out.arr_at) if out else "",
                "Stops": o.total_stops,
                "Duration": fmt.hours(o.total_duration_minutes),
                "Price (USD)": o.price.usd,
                "Quoted": f"{o.price.amount:,.2f} {o.price.currency}",
            }
        )
    return pd.DataFrame(rows).sort_values("Price (USD)", na_position="last")


# --------------------------------------------------------------------------- run
if submitted:
    if not origin_label or not dest_label:
        st.error("Choose both an origin and a destination.")
        st.stop()
    try:
        query = SearchQuery(
            origin=labels[origin_label],
            destination=labels[dest_label],
            departure_date=dep,
            return_date=ret if trip == "Roundtrip" else None,
            passengers=int(pax),
            max_stops=STOP_OPTIONS[stops_label],
            flexible_dates=flexible,
            airlines=frozenset(a.strip().upper() for a in airlines_raw.split(",") if a.strip()),
        )
    except ValueError as e:
        st.error(str(e))
        st.stop()

    weights = ScoringWeights(
        settings.score_stop_penalty_usd,
        settings.score_date_offset_penalty_usd,
        settings.score_duration_penalty_usd_per_hour,
    )
    try:
        provider = build_provider(provider_name, settings, bootstrap.history())
        service = SearchService(
            provider, bootstrap.fx(), bootstrap.history(), weights, settings.history_top_n
        )
        with st.spinner("Searching fares…"):
            result = service.search(query)
    except ProviderError as e:
        st.error(f"{e}")
        st.stop()
    st.session_state["result"] = result

if "result" in st.session_state:
    _render(st.session_state["result"])
else:
    st.markdown(
        "<div class='afm-muted'>Pick a route to begin. Demo data works fully offline; "
        "add Amadeus keys to <code>.env</code> for live fares.</div>",
        unsafe_allow_html=True,
    )
