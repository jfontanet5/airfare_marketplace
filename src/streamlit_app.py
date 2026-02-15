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
from data_access import get_route_history
from datetime import date, timedelta
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
    """
    Format a datetime as 'h:mm AM/PM'. Returns 'N/A' if dt is None.
    Works cross-platform (Windows/macOS/Linux).
    """
    if not dt:
        return "N/A"
    try:
        # Windows doesn't support %-I
        return dt.strftime("%#I:%M %p")
    except Exception:
        return dt.strftime("%I:%M %p").lstrip("0")


@st.cache_resource
def load_price_drop_model():
    """
    Load the trained price-drop model from disk.
    Returns None if the model file is missing or fails to load.
    """
    try:
        return joblib.load("models/price_drop_model.pkl")
    except Exception:
        return None


price_model = load_price_drop_model()

st.title("✈️ Airfare Marketplace (MVP)")
st.write("Price-drop predictions and smart recommendations — coming soon.")

with st.sidebar:
    st.header("Search flights")

    # Trip structure: controls whether we even show a return date
    trip_structure = st.radio(
        "Trip structure",
        options=["Roundtrip", "One-way"],
        index=0,
    )

    # --- Airports: dropdown with type-to-search, make search case-insensitive ---
    airports_df = get_airports_df(force_refresh=False).copy()

    # Create an internal option string that includes a lowercase key so "sju" matches "SJU"
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
        placeholder="Type airport/city (e.g., sju, San Juan, JFK, New York)...",
    )

    origin_label = origin_option.split("  |  ")[0] if origin_option else None
    destination_label = destination_option.split(
        "  |  ")[0] if destination_option else None

    origin = iata_from_label(origin_label)
    destination = iata_from_label(destination_label)

    # Dates
    today = date.today()
    departure_date = st.date_input(
        "Departure date",
        value=today,
        min_value=today,
    )

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
        help=(
            "When enabled, the engine can consider combinations of one-ways, mixed carriers, "
            "and alternative structures to find the cheapest valid option. "
            "When disabled, results stay closer to traditional itineraries "
            "(e.g., classic roundtrip, same carrier)."
        ),
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

# Main layout
if search_clicked:
    engine_params = {
        "origin": origin.upper(),
        "destination": destination.upper(),
        "trip_structure": trip_structure,
        "departure_date": departure_date,
        "return_date": return_date,
        "optimization_mode": optimization_mode,
        "passengers": int(passengers),
        "max_stops": max_stops,
        "airlines": airlines,
        "multicity": multicity,
        "flexible_dates": flexible_dates,
        "use_real_data": use_real_data,
    }

    display_summary = {
        "origin": origin.upper(),
        "destination": destination.upper(),
        "trip_structure": trip_structure,
        "departure_date": str(departure_date),
        "return_date": str(return_date) if return_date else None,
        "optimization_mode": optimization_mode,
        "passengers": int(passengers),
        "max_stops": max_stops,
        "airlines": airlines,
        "multicity": multicity,
        "flexible_dates": flexible_dates,
        "use_real_data": use_real_data,
        "use_live_amadeus": bool(use_live_amadeus),
    }

    st.subheader("Search parameters")
    st.json(display_summary)

    # ---- Fetch offers (provider layer) ----
    params_obj = SearchParams(**engine_params)

    if use_live_amadeus:
        provider = AmadeusProvider()
    elif params_obj.use_real_data:
        provider = CSVProvider()
    else:
        provider = MockProvider()

    try:
        with st.spinner("Fetching flight offers..."):
            offers_objects = provider.search(params_obj)
    except Exception as e:
        st.error(f"Failed to fetch offers: {e}")
        st.stop()

    if not offers_objects:
        st.warning("No offers found for this search.")
        st.stop()

    # ---- Object scoring / recommendation ----
    scored = score_offers(offers_objects, params_obj)
    recommended_scored = pick_recommended(scored)
    recommended_offer = recommended_scored.offer if recommended_scored else None

    best_global = pick_best_by_price(offers_objects)

    if params_obj.flexible_dates:
        st.markdown(
            "🗓️ **Flexible dates enabled:** showing results for a ±3 day window "
            "around your selected departure (and return, if roundtrip)."
        )

    # ⭐ Recommended option card
    st.markdown("### ⭐ Recommended option (balanced score)")
    if recommended_offer:
        col_r1, col_r2, col_r3 = st.columns(3)

        first_seg = None
        if recommended_offer.itineraries and recommended_offer.itineraries[0].segments:
            first_seg = recommended_offer.itineraries[0].segments[0]

        with col_r1:
            st.write(format_offer_label(recommended_offer))

            st.caption(
                f"Departure date: {recommended_offer.departure_date}"
                + (
                    f" · Return date: {recommended_offer.return_date}"
                    if recommended_offer.trip_structure == "Roundtrip" and recommended_offer.return_date
                    else ""
                )
            )

            if first_seg:
                carrier_display = first_seg.carrier_name or first_seg.carrier_code or ""
                flight_display = f"{carrier_display} {first_seg.flight_number or ''}".strip(
                )
                dep_display = fmt_time_ampm(first_seg.dep_at)
                arr_display = fmt_time_ampm(first_seg.arr_at)
                st.caption(
                    f"Flight: {flight_display} · Dep: {dep_display} · Arr: {arr_display}")

        with col_r2:
            st.metric(
                label="Estimated total price",
                value=f"${recommended_offer.total_price_usd:.0f} USD",
            )

        with col_r3:
            stops_out = recommended_offer.stops_out or 0
            stops_ret = recommended_offer.stops_return or 0

            if recommended_offer.trip_structure == "One-way":
                stops_text = f"{stops_out} stop(s) outbound"
            else:
                stops_text = f"{stops_out} stop(s) out · {stops_ret} stop(s) return"

            st.metric(
                label="Route quality",
                value=stops_text,
                help=(
                    f"Internal score: {recommended_scored.score:.1f} (lower is better)"
                    if recommended_scored is not None
                    else "Internal score: N/A"
                ),
            )

        # --- ML price-drop probability (TEMP shim: ML expects dict) ---
        if price_model is not None:
            try:
                legacy_for_ml = {
                    "airline": recommended_offer.airline,
                    "total_price_usd": recommended_offer.total_price_usd,
                    "stops_out": recommended_offer.stops_out,
                    "stops_return": recommended_offer.stops_return,
                    "trip_structure": recommended_offer.trip_structure,
                    "departure_date": str(recommended_offer.departure_date),
                    "return_date": str(recommended_offer.return_date) if recommended_offer.return_date else None,
                    "region": "US",  # TEMP legacy expectation; will remove in ML migration
                }

                prob_drop = predict_price_drop_probability(
                    price_model,
                    legacy_for_ml,
                    search_date=date.today(),
                )

                st.metric(
                    label="Chance of price drop (next 7 days)",
                    value=f"{prob_drop:.0%}",
                    help="Experimental ML estimate from a Random Forest model trained on synthetic data.",
                )

                if prob_drop >= 0.65:
                    suggestion = "High chance of drop → it may be worth waiting."
                elif prob_drop <= 0.35:
                    suggestion = "Low chance of drop → leaning toward buying now."
                else:
                    suggestion = "Uncertain zone → consider monitoring or setting alerts."

                st.caption(f"ML suggestion: {suggestion}")
            except Exception:
                st.caption(
                    "ML price-drop model is available but failed for this offer.")
        else:
            st.caption(
                "Price-drop model not loaded. (models/price_drop_model.pkl missing?)")

        # --- Price history chart (based on stored observations) ---
        hist_df = get_route_history(
            origin.upper(), destination.upper(), departure_date)

        if not hist_df.empty:
            st.subheader("📉 Historical prices (last recorded observations)")
            hist_df_chart = (
                hist_df[["search_datetime", "price_usd"]]
                .sort_values("search_datetime")
                .set_index("search_datetime")
            )
            st.line_chart(hist_df_chart)
        else:
            st.info("No historical price data available for this route yet.")

        # Debug
        with st.expander("🔍 Debug: recommended offer (optional)"):
            st.json(recommended_offer.raw if recommended_offer.raw else {})

    st.markdown("---")

    # 🌍 summary card (price-only)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌍 Best global fare (by price)")
        if best_global:
            st.metric(
                label=format_offer_label(best_global),
                value=f"${best_global.total_price_usd:.0f} USD",
                help=f"Provider: {best_global.provider}",
            )
        else:
            st.info("No offers found.")

    with col2:
        st.markdown("### 🇺🇸 Best US-region fare (legacy concept)")
        st.info(
            "Region-based selection has been removed in Phase 3 (it was synthetic/injected).")

    # Full offer table (object-based)
    st.subheader("Candidate itineraries")

    rows = []
    for s in scored:
        o = s.offer
        first_seg = None
        if o.itineraries and o.itineraries[0].segments:
            first_seg = o.itineraries[0].segments[0]

        rows.append(
            {
                "Provider": o.provider,
                "Origin": o.origin,
                "Destination": o.destination,
                "Type": o.trip_structure,
                "Departure Date": str(o.departure_date),
                "Return Date": str(o.return_date) if o.return_date else None,
                "Airline": o.airline_name or o.airline,
                "Flight Number": first_seg.flight_number if first_seg else None,
                "Depart Time": fmt_time_ampm(first_seg.dep_at) if first_seg else None,
                "Arrive Time": fmt_time_ampm(first_seg.arr_at) if first_seg else None,
                "Stops Depart": o.stops_out,
                "Stops Return": o.stops_return,
                "Price ($)": o.total_price_usd,
                "Score": s.score,
            }
        )

    results_df = pd.DataFrame(rows)
    if not results_df.empty and "Price ($)" in results_df.columns:
        results_df = results_df.sort_values(by=["Price ($)"], ascending=True)

    if "Price ($)" in results_df.columns:
        results_df["Price ($)"] = results_df["Price ($)"].map(
            lambda x: f"${float(x):,.0f}")

    st.dataframe(results_df, width="stretch")

    st.caption(
        "The recommended option uses a heuristic score that balances price, number of stops, "
        "and date offset. In the future, this scoring can be learned from data via an ML model."
    )
