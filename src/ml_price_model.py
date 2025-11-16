"""
ML price model utilities.

Goal (v1):
    Given a route (origin, destination), departure date, current price, and basic context,
    estimate the probability that the price will drop by at least DROP_THRESHOLD_PCT
    within the next N_DAYS_WINDOW days.

This file defines:
    - The ML dataset schema (columns we expect for training)
    - Helpers to build a simple synthetic dataset (for dev/testing)
    - A baseline classifier training pipeline (using scikit-learn)
    - An inference helper that can be called from the Streamlit app later
"""

from __future__ import annotations

from datetime import date
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -----------------------------------------------------------------------------
# Problem definition
# -----------------------------------------------------------------------------

# How far ahead we look when labeling price drops
N_DAYS_WINDOW = 7

# What counts as a "meaningful" drop
DROP_THRESHOLD_PCT = 0.05  # 5%


# This is the schema we ultimately want for training data.
# One row = one observation at a given search_date for a given flight/date combo.
ML_SCHEMA_DOC = """
Expected ML training schema (one row per price observation):

    origin              str   IATA code, e.g. 'SJU'
    destination         str   IATA code, e.g. 'JFK'
    airline             str   Marketing / operating carrier
    region              str   Region label, e.g. 'US', 'EU', 'LATAM'
    departure_date      date  Flight departure date
    search_date         date  Date when this price was observed
    days_until_departure int  (departure_date - search_date).days
    current_price_usd   float Price at search_date
    future_min_price_usd float Minimum price observed in the next N_DAYS_WINDOW days
    price_drops         int   Label: 1 if future_min_price_usd <= current_price_usd * (1 - DROP_THRESHOLD_PCT)
"""


# -----------------------------------------------------------------------------
# Synthetic data generation (dev / testing only)
# -----------------------------------------------------------------------------


def make_synthetic_training_data(n_rows: int = 500) -> pd.DataFrame:
    """
    Create a small synthetic dataset that follows the ML schema shape.
    This is only for development and plumbing; real training will use actual
    historical fare data in the same column format.
    """
    rng = np.random.default_rng(42)

    origins = ["SJU", "JFK", "MCO", "MIA"]
    destinations = ["JFK", "SJU", "MCO", "LAX"]
    airlines = ["JetBlue", "American", "Delta", "Spirit", "United"]
    regions = ["US", "EU", "LATAM"]

    today = date.today()

    rows = []
    for _ in range(n_rows):
        origin = rng.choice(origins)
        destination = rng.choice(destinations)
        airline = rng.choice(airlines)
        region = rng.choice(regions)

        # Sample a departure 3–90 days from "today"
        days_until_dep = int(rng.integers(3, 91))
        departure_date = today.toordinal() + days_until_dep
        departure_date = date.fromordinal(departure_date)

        # Sample a search_date that is before departure (0–20 days earlier)
        days_before_search = int(rng.integers(0, 21))
        search_date = departure_date.toordinal() - days_before_search
        search_date = date.fromordinal(search_date)

        # Base price roughly depends on route distance / randomness
        base_price = rng.normal(300, 80)
        base_price = max(base_price, 80)  # clamp to something reasonable

        # Add some uplift for long-haul routes (very rough)
        if destination in ("LAX",):
            base_price += 150

        # current price with small noise
        current_price = float(base_price + rng.normal(0, 30))

        # Simulate future_min_price as current price plus some movement
        movement = rng.normal(0, 40)  # price can go up or down
        future_min_price = current_price + movement

        # enforce some realism
        future_min_price = max(future_min_price, 50.0)

        price_drops = int(future_min_price <= current_price * (1 - DROP_THRESHOLD_PCT))

        rows.append(
            {
                "origin": origin,
                "destination": destination,
                "airline": airline,
                "region": region,
                "departure_date": departure_date,
                "search_date": search_date,
                "days_until_departure": days_until_dep,
                "current_price_usd": current_price,
                "future_min_price_usd": future_min_price,
                "price_drops": price_drops,
            }
        )

    df = pd.DataFrame(rows)
    return df


# -----------------------------------------------------------------------------
# Feature pipeline + training
# -----------------------------------------------------------------------------


def build_feature_pipeline() -> ColumnTransformer:
    """
    Define how to transform raw columns into model-ready features.
    """
    categorical_cols = ["origin", "destination", "airline", "region"]
    numeric_cols = ["days_until_departure", "current_price_usd"]

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", numeric_transformer, numeric_cols),
        ]
    )

    return preprocessor


def train_baseline_model(df: pd.DataFrame) -> Pipeline:
    """
    Train a baseline classifier that predicts whether price will drop
    in the next N_DAYS_WINDOW days.

    Returns a scikit-learn Pipeline that includes preprocessing + model.
    """
    preprocessor = build_feature_pipeline()

    feature_cols = [
        "origin",
        "destination",
        "airline",
        "region",
        "days_until_departure",
        "current_price_usd",
    ]
    X = df[feature_cols]
    y = df["price_drops"].astype(int)

    clf = LogisticRegression(max_iter=1000)

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )

    model.fit(X, y)
    return model


# -----------------------------------------------------------------------------
# Inference helper
# -----------------------------------------------------------------------------


def build_ml_row_from_offer(offer: Dict[str, Any], search_date: date) -> Dict[str, Any]:
    """
    Convert a current offer (as used in the app) + today's date into a single
    ML row with the expected columns.

    This function is the glue between your search engine and the ML model.
    """
    origin = offer.get("origin")
    destination = offer.get("destination")
    airline = offer.get("airline", "Unknown")
    region = offer.get("region", "US")
    departure_str = offer.get("departure_date")

    if isinstance(departure_str, date):
        departure_date = departure_str
    else:
        # e.g. '2025-12-03'
        departure_date = date.fromisoformat(str(departure_str))

    days_until_departure = (departure_date - search_date).days
    current_price = float(offer.get("total_price_usd", 0.0))

    row = {
        "origin": origin,
        "destination": destination,
        "airline": airline,
        "region": region,
        "departure_date": departure_date,
        "search_date": search_date,
        "days_until_departure": days_until_departure,
        "current_price_usd": current_price,
    }

    return row


def predict_price_drop_probability(
    model: Pipeline, offer: Dict[str, Any], search_date: date
) -> float:
    """
    Given a trained model and a current offer, return the probability that
    the price will drop in the next N_DAYS_WINDOW days.
    """
    ml_row = build_ml_row_from_offer(offer, search_date)
    X = pd.DataFrame([ml_row])

    proba = model.predict_proba(X)[0, 1]  # class 1 = price_drops
    return float(proba)
