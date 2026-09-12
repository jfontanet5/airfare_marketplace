"""Route trends: everything the history store knows about a route."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from airfare.ui import bootstrap
from airfare.ui import components as ui
from airfare.ui import format as fmt


def render() -> None:
    ui.page_header(
        "Route trends", "Observed fares over time, built from every search and collector run."
    )
    history = bootstrap.history()
    routes = history.routes()
    if routes.empty:
        st.info("No history yet. Run a search or `make collect` to start recording fares.")
        return

    labels: dict[str, tuple[str, str]] = {
        f"{o} → {d}  ({n} obs)": (str(o), str(d))
        for o, d, n in routes[["origin", "destination", "observations"]].itertuples(index=False)
    }
    c1, c2 = st.columns([2, 1])
    picked = c1.selectbox("Route", list(labels))
    origin, destination = labels[picked]
    obs = history.route_observations(origin, destination).dropna(subset=["price_usd"])
    dep_dates = sorted(obs["departure_date"].unique(), reverse=True)
    dep_choice = c2.selectbox("Departure date", ["All departure dates", *dep_dates])

    stats = history.route_price_stats(origin, destination)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Observations", f"{int(stats['count']):,}")
    m2.metric("Lowest seen", fmt.usd(stats["min"]))
    m3.metric("Median", fmt.usd(stats["median"]))
    m4.metric("Search days", int(obs["search_ts"].dt.floor("D").nunique()))

    if dep_choice == "All departure dates":
        st.subheader("Cheapest observed fare by departure date")
        chart = ui.fare_calendar_chart(history.cheapest_by_departure(origin, destination))
        if chart is not None:
            st.altair_chart(chart, width="stretch")
        st.subheader("Fare distribution")
        st.altair_chart(ui.price_histogram(obs["price_usd"]), width="stretch")
    else:
        dep = date.fromisoformat(str(dep_choice))
        st.subheader(f"Price over time for departure {dep:%a %b %-d, %Y}")
        chart = ui.trend_chart(history.daily_min_trend(origin, destination, dep))
        if chart is None:
            st.info("Not enough priced observations for this date.")
        else:
            st.altair_chart(chart, width="stretch")
        per_itin = (
            obs[obs["departure_date"] == str(dep)]
            .groupby(["signature", "airline_name", "flight_number"], dropna=False)
            .agg(
                first=("price_usd", "first"),
                last=("price_usd", "last"),
                low=("price_usd", "min"),
                seen=("price_usd", "size"),
            )
            .reset_index()
        )
        per_itin["change"] = per_itin["last"] - per_itin["first"]
        st.subheader("Per-itinerary movement")
        st.dataframe(
            per_itin.sort_values("last")[
                ["airline_name", "flight_number", "first", "last", "change", "low", "seen"]
            ],
            width="stretch",
            hide_index=True,
            column_config={
                "airline_name": "Airline",
                "flight_number": "Flight",
                "first": st.column_config.NumberColumn("First seen", format="$%d"),
                "last": st.column_config.NumberColumn("Latest", format="$%d"),
                "change": st.column_config.NumberColumn("Change", format="$%d"),
                "low": st.column_config.NumberColumn("Lowest", format="$%d"),
                "seen": st.column_config.NumberColumn("Observations"),
            },
        )

    with st.expander("Raw observations"):
        show = obs[
            [
                "search_ts",
                "provider",
                "departure_date",
                "return_date",
                "airline_name",
                "flight_number",
                "stops_out",
                "stops_return",
                "price_usd",
                "currency",
                "price_amount",
            ]
        ].copy()
        show["search_ts"] = pd.to_datetime(show["search_ts"]).dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(
            show.sort_values("search_ts", ascending=False), width="stretch", hide_index=True
        )
        st.download_button(
            "Download CSV",
            obs.to_csv(index=False).encode(),
            f"{origin}-{destination}-observations.csv",
            "text/csv",
        )
