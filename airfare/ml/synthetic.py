"""Synthetic training data for the *demo* model.

This exists so the app has a working "chance of price drop" signal before the
collector has gathered enough real observations. The generating process is
deliberately simple and documented; the resulting model is labelled as a
synthetic demo in the UI and must not be presented as validated.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from airfare.ml.features import DROP_THRESHOLD_PCT, FEATURES, LABEL

ROUTE_BASE_FARE: dict[tuple[str, str], float] = {
    ("SJU", "JFK"): 320, ("SJU", "MCO"): 200, ("SJU", "MIA"): 190, ("SJU", "LAX"): 480,
    ("JFK", "SJU"): 320, ("MCO", "SJU"): 210, ("MIA", "SJU"): 195, ("MIA", "JFK"): 240,
    ("JFK", "MAD"): 610, ("MIA", "LHR"): 720,
}  # fmt: skip
CARRIER_MARKUP = {"B6": 1.00, "AA": 1.03, "DL": 1.04, "UA": 1.02, "NK": 0.94, "IB": 1.00}


def make_synthetic_training_data(
    n_rows: int = 5000, seed: int = 42, today: date | None = None
) -> pd.DataFrame:
    """Rows carry FEATURES + LABEL + a synthetic ``search_date`` for time-based splitting.

    Mechanism: an itinerary has a latent "floor" fare; the observed fare carries a
    markup that grows with days-to-departure and with carrier. The label is 1 when
    the floor is at least DROP_THRESHOLD_PCT below the observed fare, i.e. the
    price has room to fall.
    """
    rng = np.random.default_rng(seed)
    today = today or date.today()
    routes = list(ROUTE_BASE_FARE)
    carriers = list(CARRIER_MARKUP)
    rows = []
    for _ in range(n_rows):
        origin, destination = routes[rng.integers(len(routes))]
        carrier = carriers[rng.integers(len(carriers))]
        days_out = int(rng.integers(1, 120))
        search_date = today - timedelta(days=int(rng.integers(0, 90)))
        departure = search_date + timedelta(days=days_out)
        stops = int(rng.choice([0, 1, 2], p=[0.55, 0.35, 0.10]))

        floor = (
            ROUTE_BASE_FARE[(origin, destination)] * (1 + rng.normal(0, 0.12)) * (1 - 0.06 * stops)
        )
        floor = max(floor, 60.0)
        markup = (
            (0.96 + 0.30 * days_out / 120) * CARRIER_MARKUP[carrier] * (1 + rng.normal(0, 0.07))
        )
        if departure.weekday() >= 5:
            markup *= 1.08
        price = max(floor * markup, 70.0)

        rows.append(
            {
                "origin": origin,
                "destination": destination,
                "airline_code": carrier,
                "days_until_departure": days_out,
                "current_price_usd": round(float(price), 2),
                "total_stops": stops,
                "departure_dow": departure.weekday(),
                "search_dow": search_date.weekday(),
                "search_date": search_date,
                LABEL: int(floor <= price * (1 - DROP_THRESHOLD_PCT)),
            }
        )
    return pd.DataFrame(rows)[[*FEATURES, "search_date", LABEL]]
