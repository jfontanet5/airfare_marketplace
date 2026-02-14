import os
import pandas as pd
from typing import Dict, Any, List
from datetime import date, timedelta
from pathlib import Path

DATA_PATH = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), "data", "sample_flights.csv")


def load_flight_offers(search_params: Dict[str, Any]) -> pd.DataFrame:
    """
    Load flight offers from a local CSV and filter them according to the
    current search parameters.

    Expected columns in sample_flights.csv:
      origin, destination, departure_date, return_date, trip_structure,
      region, provider, airline, total_price_usd, stops_out, stops_return
    """

    if not os.path.exists(DATA_PATH):
        # If file is missing, return empty DataFrame and let the caller decide
        return pd.DataFrame()

    df = pd.read_csv(DATA_PATH)

    # Basic normalization
    df["origin"] = df["origin"].str.upper()
    df["destination"] = df["destination"].str.upper()

    # Filter by origin/destination
    origin = search_params["origin"].upper()
    destination = search_params["destination"].upper()
    trip_structure = search_params["trip_structure"]
    departure_date = search_params["departure_date"]       # datetime.date
    # datetime.date or None
    return_date = search_params["return_date"]
    max_stops_label = search_params["max_stops"]
    flexible_dates = search_params["flexible_dates"]

    # Trip structure filter
    df = df[df["trip_structure"] == trip_structure]

    # Origin / destination filter
    df = df[(df["origin"] == origin) & (df["destination"] == destination)]

    # Date window
    df["departure_date"] = pd.to_datetime(df["departure_date"]).dt.date

    if flexible_dates:
        # ±3 days around the selected departure date
        dep_min = departure_date - timedelta(days=3)
        dep_max = departure_date + timedelta(days=3)
    else:
        dep_min = departure_date
        dep_max = departure_date

    df = df[(df["departure_date"] >= dep_min) &
            (df["departure_date"] <= dep_max)]

    # Stops filter
    if "Nonstop" in max_stops_label:
        max_allowed_stops = 0
    elif "1 stop" in max_stops_label:
        max_allowed_stops = 1
    else:
        max_allowed_stops = 2

    if trip_structure == "One-way":
        df = df[df["stops_out"] <= max_allowed_stops]
    else:
        df = df[
            (df["stops_out"] <= max_allowed_stops)
            & (df["stops_return"] <= max_allowed_stops)
        ]

    return df


BASE_DIR = Path(__file__).resolve().parent.parent  # src/.. -> project root
HISTORY_PATH = BASE_DIR / "data" / "price_history.csv"


def load_price_history() -> pd.DataFrame:
    df = pd.read_csv(HISTORY_PATH)

    # adjust column names to match your CSV
    df["departure_date"] = pd.to_datetime(
        df["departure_date"],
        format="%Y-%m-%d",
        errors="coerce",
    )
    df["search_datetime"] = pd.to_datetime(
        df["search_datetime"],
        errors="coerce",
    )
    return df


def get_route_history(origin: str, destination: str, departure_date) -> pd.DataFrame:
    df = load_price_history()

    query_date = pd.to_datetime(departure_date)

    mask = (
        (df["origin"] == origin)
        & (df["destination"] == destination)
        & (df["departure_date"] == query_date)
    )
    return df[mask].copy()
