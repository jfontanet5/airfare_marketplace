from datetime import date, timedelta
import joblib
import pandas as pd
import streamlit as st
from data_access import load_flight_offers, get_route_history
from ml_price_model import predict_price_drop_probability
from core.models import SearchParams
from providers.mock_provider import MockProvider
from providers.csv_provider import CSVProvider
from services.offer_bridge import offers_to_legacy_dicts
from providers.amadeus_provider import AmadeusProvider
from airports_service import get_airports_df, iata_from_label
from engine import (
    generate_dummy_offers,
    pick_best_offers,
    pick_recommended_offer,
    format_offer_label,
)
from dotenv import load_dotenv
load_dotenv()


st.set_page_config(
    page_title="Airfare Marketplace",
    layout="wide",
)


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
    # We keep the displayed label clean via format_func.
    airports_df["option"] = airports_df["label"] + \
        "  |  " + airports_df["label"].str.lower()

    options = airports_df["option"].tolist()

    origin_option = st.selectbox(
        "Departing from",
        options=options,
        index=None,
        format_func=lambda x: x.split("  |  ")[0],  # show only the clean label
        placeholder="Type airport/city (e.g., sju, San Juan, JFK, New York)...",
    )

    destination_option = st.selectbox(
        "Departing to",
        options=options,
        index=None,
        format_func=lambda x: x.split("  |  ")[0],
        placeholder="Type airport/city (e.g., sju, San Juan, JFK, New York)...",
    )

    # Extract clean label (left side) and then IATA
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
        min_value=today,  # don't allow past dates
    )

    return_date = None
    if trip_structure == "Roundtrip":
        return_date = st.date_input(
            "Return date",
            value=departure_date + timedelta(days=3),
            min_value=departure_date,  # can't be before departure
        )

    # How the engine should search
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

    # Map checkbox into a simple mode string
    optimization_mode = "Optimal" if optimize_cheapest else "Traditional"

    passengers = st.number_input("Passengers", min_value=1, value=1, step=1)

    st.markdown("---")

    # Filters / options
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

    ready_to_search = (
        bool(origin)
        and bool(destination)
        and origin != destination
    )

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

    # Bridge Offer objects -> legacy dicts (for existing engine/scoring code)
    offers = offers_to_legacy_dicts(offers_objects)

    if not offers:
        st.warning("No offers found for this search.")
        st.stop()

    # ---- Existing logic continues below ----
    best_global, best_us = pick_best_offers(offers)
    recommended = pick_recommended_offer(offers, engine_params)

    if engine_params["flexible_dates"]:
        st.markdown(
            "🗓️ **Flexible dates enabled:** showing results for a ±3 day window "
            "around your selected departure (and return, if roundtrip)."
        )

    # ⭐ Recommended option card
    st.markdown("### ⭐ Recommended option (balanced score)")
    if recommended:
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.write(format_offer_label(recommended))
            st.caption(
                f"Departure: {recommended.get('departure_date', 'N/A')}"
                + (
                    f" · Return: {recommended.get('return_date')}"
                    if engine_params["trip_structure"] == "Roundtrip"
                    else ""
                )
            )
        with col_r2:
            st.metric(
                label="Estimated total price",
                value=f"${recommended['total_price_usd']:.0f} USD",
            )
        with col_r3:
            stops_out = recommended.get("stops_out") or 0
            stops_ret = recommended.get("stops_return") or 0
            if engine_params["trip_structure"] == "One-way":
                stops_text = f"{stops_out} stop(s) outbound"
            else:
                stops_text = f"{stops_out} stop(s) out · {stops_ret} stop(s) return"
            st.metric(
                label="Route quality",
                value=stops_text,
                help=f"Internal score: {recommended.get('score', 0):.1f} (lower is better)",
            )

        # --- ML price-drop probability for the recommended option ---
        if price_model is not None:
            try:
                prob_drop = predict_price_drop_probability(
                    price_model,
                    recommended,
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

        # Optional: show raw dict for the recommended offer only
        with st.expander("🔍 Debug: recommended offer (optional)"):
            st.json(recommended)

    st.markdown("---")

    # 🌍 / 🇺🇸 summary cards (price-only)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌍 Best global fare (by price)")
        if best_global:
            st.metric(
                label=format_offer_label(best_global),
                value=f"${best_global['total_price_usd']:.0f} USD",
                help=f"Provider: {best_global.get('provider', 'N/A')}",
            )
        else:
            st.info("No global offers found.")

    with col2:
        st.markdown("### 🇺🇸 Best US-region fare (by price)")
        if best_us:
            st.metric(
                label=format_offer_label(best_us),
                value=f"${best_us['total_price_usd']:.0f} USD",
                help=f"Provider: {best_us.get('provider', 'N/A')}",
            )
        else:
            st.info("No US-region offers in this dataset.")

    # Full offer table (only relevant columns)
    st.subheader("Candidate itineraries")
    results_df = pd.DataFrame(offers)

    preferred_cols = [
        "provider",
        "region",
        "origin",
        "destination",
        "trip_structure",
        "departure_date",
        "return_date",
        "airline",
        "stops_out",
        "stops_return",
        "total_price_usd",
        "currency",
        "score",
    ]
    display_cols = [c for c in preferred_cols if c in results_df.columns]

    if display_cols:
        if "total_price_usd" in results_df.columns:
            results_df = results_df.sort_values(
                by=["total_price_usd"], ascending=True)
        st.dataframe(results_df[display_cols], width="stretch")
    else:
        st.dataframe(results_df, width="stretch")

    st.caption(
        "The recommended option uses a heuristic score that balances price, number of stops, "
        "date offset, and region. In the future, this scoring can be learned from data via an ML model."
    )
