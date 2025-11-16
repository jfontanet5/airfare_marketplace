import streamlit as st
import pandas as pd
from datetime import date, timedelta
from data_access import load_flight_offers
from datetime import date, timedelta
from engine import (
    generate_dummy_offers,
    pick_best_offers,
    pick_recommended_offer,
    format_offer_label,
)

st.set_page_config(
    page_title="Airfare Marketplace",
    layout="wide",
)

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

    # Core routing
    origin = st.text_input("Departing from", "SJU")
    destination = st.text_input("Departing to", "JFK")

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

    use_real_data = st.checkbox(
        "Use local real data (CSV)",
        value=False,
        help="When enabled, the engine will try to load offers from data/sample_flights.csv.",
    )

    search_clicked = st.button("Search")

# Main layout: later we can split into columns for results + insights
if search_clicked:
    # Params for the engine (raw objects: dates, bools, etc.)
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

    # Params for display (stringified dates)
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
    }

    st.subheader("Search parameters")
    st.json(display_summary)

    # ---- Fetch offers from either real data or dummy engine ----
    if engine_params["use_real_data"]:
        df_offers = load_flight_offers(engine_params)
        offers = df_offers.to_dict(orient="records")
    else:
        offers = generate_dummy_offers(engine_params)

    if not offers:
        st.warning("No offers found for this search.")
    else:
        # Compute best-by-price and recommended-by-score
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

        # Full offer table
        st.subheader("Candidate itineraries")
        results_df = pd.DataFrame(offers)
        st.dataframe(results_df, width="stretch")

        st.caption(
            "The recommended option uses a heuristic score that balances price, number of stops, "
            "date offset, and region. In the future, this scoring can be learned from data via an ML model."
        )
else:
    st.info("Use the sidebar to configure a search, then click **Search** to see dummy options. 🚀")
