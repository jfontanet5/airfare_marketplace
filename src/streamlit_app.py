from airports_service import get_airports_df, iata_from_label
from providers.amadeus_provider import AmadeusProvider
from providers.csv_provider import CSVProvider
from providers.mock_provider import MockProvider
from core.scoring import (
    score_offers,
    pick_recommended,
    pick_best_by_price,
    format_offer_label,
)
from core.models import SearchParams
from ml_price_model import predict_price_drop_probability
from sqlite_history_store import SqlitePriceHistoryStore
from services.fx_rate_services import FxRateService

from datetime import date, timedelta, datetime, timezone
import joblib
import pandas as pd
import streamlit as st

from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Airfare Marketplace",
    layout="wide",
)


def fmt_time_ampm(dt):
    if not dt:
        return "N/A"
    try:
        return dt.strftime("%I:%M %p").lstrip("0")
    except Exception:
        return "N/A"


def fmt_usd(x) -> str:
    try:
        return f"${float(x):,.0f}"
    except Exception:
        return ""


@st.cache_resource
def load_price_drop_model():
    try:
        return joblib.load("models/price_drop_model.pkl")
    except Exception:
        return None


@st.cache_resource
def get_history_store():
    return SqlitePriceHistoryStore()


@st.cache_resource
def get_fx_service():
    return FxRateService()


def offer_total_usd(offer, fx_service: FxRateService, ts: datetime) -> float:
    """
    offer.total_price_usd is currently "amount in offer.currency" (misnamed).
    We convert it to USD for all UI + analytics.
    """
    amt = float(getattr(offer, "total_price_usd", 0.0) or 0.0)
    cur = str(getattr(offer, "currency", "USD") or "USD")
    return float(fx_service.amount_to_usd(amt, cur, ts))


history_store = get_history_store()
fx_service = get_fx_service()
price_model = load_price_drop_model()

st.title("✈️ Airfare Marketplace")
st.caption(
    "Historical intelligence + price-drop predictions.")
results_slot = st.empty()
with st.sidebar:
    st.header("Search flights")

    trip_structure = st.radio(
        "Trip structure",
        options=["Roundtrip", "One-way"],
        index=0,
    )

    airports_df = get_airports_df(force_refresh=False).copy()
    airports_df["option"] = airports_df["label"] + \
        "  |  " + airports_df["label"].str.lower()
    options = airports_df["option"].tolist()

    origin_option = st.selectbox(
        "Departing from",
        options=options,
        index=None,
        format_func=lambda x: x.split("  |  ")[0],
        placeholder="Type airport/city (e.g., sju, San Juan, JFK, New York)...",
    )

    destination_option = st.selectbox(
        "Departing to",
        options=options,
        index=None,
        format_func=lambda x: x.split("  |  ")[0],
        placeholder="Type airport/city (e.g., mad, Madrid, LHR, London)...",
    )

    origin_label = origin_option.split("  |  ")[0] if origin_option else None
    destination_label = destination_option.split(
        "  |  ")[0] if destination_option else None
    origin = iata_from_label(origin_label)
    destination = iata_from_label(destination_label)

    today = date.today()
    departure_date = st.date_input(
        "Departure date", value=today, min_value=today)

    return_date = None
    if trip_structure == "Roundtrip":
        return_date = st.date_input(
            "Return date",
            value=departure_date + timedelta(days=3),
            min_value=departure_date,
        )

    optimize_cheapest = st.checkbox(
        "Optimal mode (cheapest combinations)",
        value=True,
        help="Heuristic optimizer (Phase 4 will replace parts with learned models).",
    )
    optimization_mode = "Optimal" if optimize_cheapest else "Traditional"

    passengers = st.number_input("Passengers", min_value=1, value=1, step=1)

    st.markdown("---")

    max_stops = st.selectbox(
        "Max stops",
        options=["Nonstop only", "Up to 1 stop", "Up to 2+ stops"],
        index=1,
    )

    airlines = st.multiselect(
        "Preferred airlines (optional)",
        options=["Any", "American", "Delta", "United", "JetBlue", "Spirit"],
        default=["Any"],
    )

    multicity = st.checkbox("Allow multicity / open-jaw")
    flexible_dates = st.checkbox("Flexible dates ± 3 days")

    data_source = st.selectbox(
        "Data source",
        options=["Mock (offline)", "CSV (local)", "Amadeus (live, limited)"],
        index=0,
    )

    use_real_data = (data_source == "CSV (local)")
    use_live_amadeus = (data_source == "Amadeus (live, limited)")

    ready_to_search = bool(origin) and bool(
        destination) and origin != destination
    search_clicked = st.button("Search", disabled=not ready_to_search)
    if not ready_to_search:
        st.info("Select different origin and destination airports to enable Search.")

# Main
if search_clicked:
    results_slot.empty()
    with results_slot.container():

        search_time_utc = datetime.now(timezone.utc)

        params_obj = SearchParams(
            origin=origin.upper(),
            destination=destination.upper(),
            trip_structure=trip_structure,
            departure_date=departure_date,
            return_date=return_date,
            optimization_mode=optimization_mode,
            passengers=int(passengers),
            max_stops=max_stops,
            airlines=airlines,
            multicity=multicity,
            flexible_dates=flexible_dates,
            use_real_data=use_real_data,
        )

        if use_live_amadeus:
            provider = AmadeusProvider()
        elif params_obj.use_real_data:
            provider = CSVProvider()
        else:
            provider = MockProvider()

        st.subheader("Search parameters")
        st.json(
            {
                "origin": params_obj.origin,
                "destination": params_obj.destination,
                "trip_structure": params_obj.trip_structure,
                "departure_date": str(params_obj.departure_date),
                "return_date": str(params_obj.return_date) if params_obj.return_date else None,
                "optimization_mode": params_obj.optimization_mode,
                "passengers": int(params_obj.passengers),
                "max_stops": params_obj.max_stops,
                "airlines": params_obj.airlines,
                "multicity": bool(params_obj.multicity),
                "flexible_dates": bool(params_obj.flexible_dates),
                "source": data_source,
            }
        )

        try:
            with st.spinner("Fetching flight offers..."):
                offers_objects = provider.search(params_obj)
        except Exception as e:
            st.error(f"Failed to fetch offers: {e}")
            st.stop()

        if not offers_objects:
            st.warning("No offers found for this search.")
            st.stop()

        # Log history (do not block UI if logging fails)
        try:
            history_store.append_offers(
                offers_objects,
                origin=params_obj.origin,
                destination=params_obj.destination,
                trip_structure=params_obj.trip_structure,
                departure_date=params_obj.departure_date,
                return_date=params_obj.return_date,
                passengers=int(params_obj.passengers),
                max_stops_label=params_obj.max_stops,
                flexible_dates=bool(params_obj.flexible_dates),
                top_n=30,
            )
        except Exception:
            pass

        scored = score_offers(offers_objects, params_obj)
        recommended_scored = pick_recommended(scored)
        recommended_offer = recommended_scored.offer if recommended_scored else None
        best_global = pick_best_by_price(offers_objects)

        if params_obj.flexible_dates:
            st.info(
                "Flexible dates enabled: results cover ±3 days around your selected dates.")

        # ===== Recommended card =====
        st.markdown("### ⭐ Recommended option")
        if not recommended_offer:
            st.warning("No recommended offer could be selected.")
        else:
            first_seg = None
            if recommended_offer.itineraries and recommended_offer.itineraries[0].segments:
                first_seg = recommended_offer.itineraries[0].segments[0]

            col_r1, col_r2, col_r3 = st.columns(3)

            with col_r1:
                st.write(format_offer_label(recommended_offer))
                st.caption(
                    f"Departure: {recommended_offer.departure_date}"
                    + (
                        f" · Return: {recommended_offer.return_date}"
                        if recommended_offer.trip_structure == "Roundtrip" and recommended_offer.return_date
                        else ""
                    )
                )
                if first_seg:
                    carrier_display = first_seg.carrier_name or first_seg.carrier_code or ""
                    flight_display = f"{carrier_display} {first_seg.flight_number or ''}".strip(
                    )
                    st.caption(
                        f"Flight: {flight_display} · Dep: {fmt_time_ampm(first_seg.dep_at)} · Arr: {fmt_time_ampm(first_seg.arr_at)}"
                    )

            with col_r2:
                try:
                    rec_usd = offer_total_usd(
                        recommended_offer, fx_service, search_time_utc)
                    st.metric("Estimated total price (USD)", fmt_usd(rec_usd))
                except Exception as e:
                    st.metric("Estimated total price (USD)", "USD unavailable")
                    st.caption(f"FX error: {e}")

            with col_r3:
                stops_out = recommended_offer.stops_out or 0
                stops_ret = recommended_offer.stops_return or 0
                stops_text = (
                    f"{stops_out} stop(s)"
                    if recommended_offer.trip_structure == "One-way"
                    else f"{stops_out} out · {stops_ret} return"
                )
                st.metric(
                    "Stops",
                    stops_text,
                    help=(
                        f"Internal score: {recommended_scored.score:.1f} (lower is better)" if recommended_scored else None),
                )

            # ===== ML inference =====
            if price_model is not None:
                try:
                    prob_drop = predict_price_drop_probability(
                        price_model,
                        recommended_offer,
                        search_date=date.today(),
                    )
                    st.metric("Chance of price drop (next 7 days)",
                              f"{prob_drop:.0%}")
                except Exception as e:
                    st.caption(f"ML model available but failed: {e}")
            else:
                st.caption(
                    "Price-drop model not loaded (models/price_drop_model.pkl missing).")

            # ===== Historical trend (USD) =====
            try:
                mode, hist_df = history_store.get_market_trend_usd_dual(
                    params_obj.origin,
                    params_obj.destination,
                    params_obj.departure_date,
                    fx_service=fx_service,
                )
            except Exception as e:
                mode, hist_df = "raw", pd.DataFrame()
                st.info(f"History trend unavailable (FX or DB): {e}")

            if not hist_df.empty:
                if mode == "raw":
                    st.subheader("📉 Market trend (raw observations, USD)")
                    plot_df = hist_df.copy()
                    plot_df["search_datetime"] = pd.to_datetime(
                        plot_df["search_datetime"], utc=True, errors="coerce")
                    plot_df = plot_df.dropna(
                        subset=["search_datetime"]).sort_values("search_datetime")
                    plot_df["best_so_far"] = plot_df["price_usd"].cummin()
                    plot_df["rolling_avg"] = plot_df["price_usd"].rolling(
                        window=7, min_periods=1).mean()

                    chart_df = (
                        plot_df[["search_datetime", "price_usd",
                                "rolling_avg", "best_so_far"]]
                        .set_index("search_datetime")
                        .rename(
                            columns={
                                "price_usd": "Observed (USD)",
                                "rolling_avg": "Rolling avg (USD)",
                                "best_so_far": "Best so far (USD)",
                            }
                        )
                    )
                    st.line_chart(chart_df)
                else:
                    st.subheader("📉 Market trend (daily minimum, USD)")
                    plot_df = hist_df.copy()
                    plot_df["search_day"] = pd.to_datetime(
                        plot_df["search_day"], errors="coerce")
                    plot_df = plot_df.dropna(
                        subset=["search_day"]).sort_values("search_day")
                    plot_df["rolling_7d_avg"] = plot_df["min_price_usd"].rolling(
                        window=7, min_periods=1).mean()
                    plot_df["best_so_far"] = plot_df["min_price_usd"].cummin()

                    chart_df = (
                        plot_df[["search_day", "min_price_usd",
                                "rolling_7d_avg", "best_so_far"]]
                        .set_index("search_day")
                        .rename(
                            columns={
                                "min_price_usd": "Daily min (USD)",
                                "rolling_7d_avg": "7-day avg (USD)",
                                "best_so_far": "Best so far (USD)",
                            }
                        )
                    )
                    st.line_chart(chart_df)
            else:
                st.info("No historical price data available for this route yet.")

        st.markdown("---")

        # ===== Best global fare (USD only) =====
        st.markdown("### 🌍 Best global fare (lowest USD)")
        if best_global:
            try:
                best_usd = offer_total_usd(
                    best_global, fx_service, search_time_utc)
                st.metric(label=format_offer_label(
                    best_global), value=fmt_usd(best_usd))
            except Exception as e:
                st.metric(label=format_offer_label(
                    best_global), value="USD unavailable")
                st.caption(f"FX error: {e}")
        else:
            st.info("No offers found.")

        # ===== Candidate itineraries table (USD only) =====
        st.subheader("Candidate itineraries")

        # 1) Final UI-layer dedup (use canonical itinerary signature if present)
        display_map = {}
        for s in scored:
            o = s.offer
            sig = getattr(o, "offer_signature", None)

            # Fallback if offer_signature isn't present yet:
            # still removes most visible duplicates, but signature is strongly preferred.
            if not sig:
                first_seg = None
                if o.itineraries and o.itineraries[0].segments:
                    first_seg = o.itineraries[0].segments[0]
                dep_iso = first_seg.dep_at.isoformat() if first_seg and first_seg.dep_at else ""
                sig = "|".join(
                    [
                        o.provider or "",
                        o.origin or "",
                        o.destination or "",
                        str(o.departure_date or ""),
                        str(o.return_date or ""),
                        (o.airline or ""),
                        str(o.stops_out or 0),
                        str(o.stops_return or 0),
                        (first_seg.flight_number or "") if first_seg else "",
                        dep_iso,
                    ]
                )

            # Keep the *best* scored row per signature (lower score is better in your system)
            if sig not in display_map or s.score < display_map[sig].score:
                display_map[sig] = s

        dedup_scored = list(display_map.values())

        # 2) Cache FX rate per currency once per render (usually just EUR)
        fx_rate_by_currency = {}

        # 3) Build table rows from deduped list
        rows = []
        for s in dedup_scored:
            o = s.offer

            first_seg = None
            if o.itineraries and o.itineraries[0].segments:
                first_seg = o.itineraries[0].segments[0]

            # Convert to USD (single-currency UI)
            usd_price = None
            try:
                cur = str(getattr(o, "currency", "USD") or "USD").upper()
                if cur not in fx_rate_by_currency:
                    fx_rate_by_currency[cur] = fx_service.get_rate_to_usd(
                        cur, search_time_utc)
                rate = fx_rate_by_currency[cur]
                usd_price = float(getattr(o, "total_price_usd", 0.0)
                                  or 0.0) * float(rate)
            except Exception:
                usd_price = None

            rows.append(
                {
                    "Provider": o.provider,
                    "Origin": o.origin,
                    "Destination": o.destination,
                    "Type": o.trip_structure,
                    "Departure Date": str(o.departure_date),
                    "Return Date": str(o.return_date) if o.return_date else "",
                    "Airline": o.airline_name or o.airline or "",
                    "Flight Number": first_seg.flight_number if first_seg else "",
                    "Depart Time": fmt_time_ampm(first_seg.dep_at) if first_seg else "",
                    "Arrive Time": fmt_time_ampm(first_seg.arr_at) if first_seg else "",
                    "Stops Depart": int(o.stops_out or 0),
                    "Stops Return": int(o.stops_return or 0),
                    "Price (USD)": usd_price,
                    "Score": float(s.score),
                }
            )
        results_df = pd.DataFrame(rows)

        if not results_df.empty:
            results_df["Price (USD)"] = pd.to_numeric(
                results_df["Price (USD)"], errors="coerce")
            results_df = results_df.sort_values(
                by=["Price (USD)", "Score"], ascending=True, na_position="last")
            results_df["Price (USD)"] = results_df["Price (USD)"].map(
                lambda x: fmt_usd(x) if pd.notnull(x) else "")

        st.dataframe(results_df, width="stretch")

        st.caption(
            "Phase 4 note: UI is now USD-only to support consistent historical analytics and ML training. "
            "Next: dataset builder + time-split training on real SQLite observations."
        )
