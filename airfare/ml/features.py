"""Feature definitions shared by training and inference.

Keeping one function that turns an offer into a feature row guarantees the
model never sees a different encoding at inference than it saw in training
(the previous version trained on airline *names* and predicted on *codes*).
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from airfare.domain.models import Offer

CATEGORICAL = ["origin", "destination", "airline_code"]
NUMERIC = [
    "days_until_departure",
    "current_price_usd",
    "total_stops",
    "departure_dow",
    "search_dow",
]
FEATURES = CATEGORICAL + NUMERIC
LABEL = "price_drops"

N_DAYS_WINDOW = 7
DROP_THRESHOLD_PCT = 0.05


def feature_row(offer: Offer, search_date: date) -> dict[str, Any]:
    if offer.price.usd is None:
        raise ValueError("cannot build features for an offer without a USD price")
    return {
        "origin": offer.origin,
        "destination": offer.destination,
        "airline_code": offer.airline_code or "UNK",
        "days_until_departure": (offer.departure_date - search_date).days,
        "current_price_usd": offer.price.usd,
        "total_stops": offer.total_stops,
        "departure_dow": offer.departure_date.weekday(),
        "search_dow": search_date.weekday(),
    }


def feature_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)[FEATURES]
