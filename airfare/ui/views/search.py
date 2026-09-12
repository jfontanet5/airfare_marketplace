"""Search page: form → SearchService → ranked results."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from airfare.domain.models import Offer, ScoredOffer, SearchQuery
from airfare.domain.scoring import ScoringWeights
from airfare.ml.predict import price_position
from airfare.providers import ProviderError, ProviderName, available_providers, build_provider
from airfare.services.search import SearchResult, SearchService
from airfare.ui import bootstrap, nav
from airfare.ui import components as ui
from airfare.ui import format as fmt

STOP_OPTIONS = {"Nonstop only": 0, "Up to 1 stop": 1, "Up to 2 stops": 2}
PROVIDER_LABELS = {
    ProviderName.MOCK: "Demo data (offline)",
    ProviderName.REPLAY: "Replay last stored search",
    ProviderName.SERPAPI: "Google Flights via SerpApi (live)",
}
SORTS = ["Best", "Price", "Duration", "Departure"]


def _search_form(providers: list[ProviderName]) -> tuple[bool, dict[str, object]]:
    airport_list = bootstrap.airports(include_full=ProviderName.SERPAPI in providers)
    labels = {a.label: a.iata for a in airport_list}
    today = date.today()
    with st.form("search", border=True):
        c1, c2, c3 = st.columns([1.5, 1.5, 1])
        origin = c1.selectbox("From", list(labels), index=None, placeholder="City, airport or code")
        dest = c2.selectbox("To", list(labels), index=None, placeholder="City, airport or code")
        trip = c3.radio("Trip", ["Round trip", "One way"], horizontal=True)

        d1, d2, d3, d4 = st.columns(4)
        dep = d1.date_input("Depart", value=today + timedelta(days=21), min_value=today)
        ret = d2.date_input("Return", value=dep + timedelta(days=5), min_value=dep)
        pax = d3.number_input("Passengers", min_value=1, max_value=9, value=1)
        stops = d4.selectbox("Stops", list(STOP_OPTIONS), index=1)

        e1, e2, e3 = st.columns([1, 1.2, 1.4])
        flexible = e1.checkbox(
            "Flexible dates (±3 days)", help="Live mode: 7 requests instead of 1"
        )
        airlines = e2.text_input("Airlines (IATA codes)", placeholder="B6, AA, DL — optional")
        provider = e3.selectbox("Data source", providers, format_func=lambda p: PROVIDER_LABELS[p])
        submitted = st.form_submit_button("Search fares", type="primary", width="stretch")
    return submitted, {
        "origin": labels.get(origin or ""),
        "destination": labels.get(dest or ""),
        "trip": trip,
        "dep": dep,
        "ret": ret,
        "pax": int(pax),
        "stops": STOP_OPTIONS[stops],
        "flexible": flexible,
        "airlines": frozenset(a.strip().upper() for a in airlines.split(",") if a.strip()),
        "provider": provider,
    }


def _run_search(form: dict[str, object]) -> SearchResult | None:
    if not form["origin"] or not form["destination"]:
        st.error("Choose both an origin and a destination.")
        return None
    try:
        query = SearchQuery(
            origin=str(form["origin"]),
            destination=str(form["destination"]),
            departure_date=form["dep"],  # type: ignore[arg-type]
            return_date=form["ret"] if form["trip"] == "Round trip" else None,  # type: ignore[arg-type]
            passengers=int(form["pax"]),  # type: ignore[call-overload]
            max_stops=int(form["stops"]),  # type: ignore[call-overload]
            flexible_dates=bool(form["flexible"]),
            airlines=form["airlines"],  # type: ignore[arg-type]
        )
    except ValueError as e:
        st.error(str(e))
        return None

    settings = bootstrap.settings()
    weights = ScoringWeights(
        settings.score_stop_penalty_usd,
        settings.score_date_offset_penalty_usd,
        settings.score_duration_penalty_usd_per_hour,
    )
    try:
        provider = build_provider(str(form["provider"]), settings, bootstrap.history())
        service = SearchService(
            provider, bootstrap.fx(), bootstrap.history(), weights, settings.history_top_n
        )
        with st.spinner("Searching fares…"):
            return service.search(query)
    except ProviderError as e:
        st.error(str(e))
        return None


def _apply_filters(
    ranked: list[ScoredOffer], sort: str, airlines: list[str], nonstop: bool
) -> list[ScoredOffer]:
    out = [s for s in ranked if (not airlines or s.offer.airline_display in airlines)]
    if nonstop:
        out = [s for s in out if s.offer.total_stops == 0]
    if sort == "Price":
        out.sort(key=lambda s: s.offer.price.usd if s.offer.price.usd is not None else float("inf"))
    elif sort == "Duration":
        out.sort(key=lambda s: s.offer.total_duration_minutes or 10**9)
    elif sort == "Departure":
        out.sort(
            key=lambda s: (
                (
                    s.offer.outbound.dep_at.time()
                    if s.offer.outbound and s.offer.outbound.dep_at
                    else None
                )
                or pd.Timestamp.max.time()
            )
        )
    return out


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
                "Signature": o.signature,
            }
        )
    return pd.DataFrame(rows).sort_values("Price (USD)", na_position="last")


def _render_results(result: SearchResult) -> None:
    for w in result.warnings:
        st.warning(w)
    if not result.offers:
        st.info("No fares matched. Try more stops, other dates, or fewer airline restrictions.")
        return

    history = bootstrap.history()
    predictor = bootstrap.predictor()
    q = result.query
    stats = history.route_price_stats(q.origin, q.destination)
    rec = result.recommended
    cheapest = result.cheapest
    priced = [o.price.usd for o in result.offers if o.price.usd is not None]

    ui.eyebrow(
        f"{q.origin} → {q.destination} · {q.departure_date:%b %-d}"
        + (f" – {q.return_date:%b %-d}" if q.return_date else " · one way")
    )
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Options", len(result.offers))
    k2.metric("Cheapest", fmt.usd(cheapest.price.usd) if cheapest else "—")
    k3.metric("Median", fmt.usd(float(pd.Series(priced).median())) if priced else "—")
    if rec and predictor and rec.offer.price.usd is not None:
        k4.metric("Drop chance · recommended", f"{predictor.probability(rec.offer):.0%}")
        k4.caption(
            "synthetic-data demo model"
            if predictor.card.is_synthetic
            else f"model trained on {predictor.card.n_train:,} observations"
        )
    else:
        k4.metric("Drop chance", "—")
        k4.caption("no model loaded — run `make train`")

    f1, f2, f3 = st.columns([2.2, 1.8, 1], vertical_alignment="bottom")
    sort = (
        f1.segmented_control("Sort", SORTS, default="Best", label_visibility="collapsed") or "Best"
    )
    airline_opts = sorted({o.airline_display for o in result.offers})
    airlines = f2.multiselect(
        "Airlines", airline_opts, placeholder="All airlines", label_visibility="collapsed"
    )
    nonstop = f3.toggle("Nonstop only")

    shown = _apply_filters(result.ranked, sort, airlines, nonstop)
    tab_cards, tab_table, tab_history = st.tabs(
        [f"Results ({len(shown)})", "Table", "Price history"]
    )
    with tab_cards:
        for s in shown[:30]:
            usd = s.offer.price.usd
            prob = predictor.probability(s.offer) if predictor and usd is not None else None
            ui.offer_card(
                s,
                recommended=rec is not None and s.offer.signature == rec.offer.signature,
                drop_prob=prob,
                position=price_position(usd, stats) if usd is not None else None,
                model_is_demo=bool(predictor and predictor.card.is_synthetic),
            )
        if len(shown) > 30:
            st.caption(f"Showing 30 of {len(shown)}. Use the table for the full list.")
    with tab_table:
        st.dataframe(_table([s.offer for s in shown]), width="stretch", hide_index=True)
    with tab_history:
        chart = ui.trend_chart(history.daily_min_trend(q.origin, q.destination, q.departure_date))
        if chart is None:
            st.info("No history for this route and departure date yet — every search adds to it.")
        else:
            st.altair_chart(chart, width="stretch")
        if stats:
            st.caption(
                f"{int(stats['count'])} observations on {q.origin} → {q.destination} across all dates · "
                f"low {fmt.usd(stats['min'])} · median {fmt.usd(stats['median'])} · high {fmt.usd(stats['max'])}"
            )
        st.page_link(nav.TRENDS, label="Open the full route trends page →")

    st.caption(
        f"{PROVIDER_LABELS[ProviderName(result.provider)]} · searched {result.searched_at:%Y-%m-%d %H:%M} UTC · "
        f"{result.observations_persisted} observations saved to history"
    )


def render() -> None:
    settings = bootstrap.settings()
    providers = available_providers(settings)
    status = (
        ui.chip("live search on", "live")
        if settings.live_configured
        else ui.chip("offline mode", "typical")
    )
    ui.page_header(
        "Airfare Marketplace",
        "Transparent fares: one price basis, visible history, an honest signal.",
        [status],
    )

    submitted, form = _search_form(providers)
    if submitted:
        result = _run_search(form)
        if result is not None:
            st.session_state["result"] = result

    if "result" in st.session_state:
        _render_results(st.session_state["result"])
    else:
        st.markdown(
            "<div class='afm-muted'>Pick a route to begin. Demo data works fully offline; "
            "add a SerpApi key to <code>.env</code> for live Google Flights fares.</div>",
            unsafe_allow_html=True,
        )
