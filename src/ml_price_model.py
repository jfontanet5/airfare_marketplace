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
from typing import Dict, Any, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from core.models import Offer

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

def make_synthetic_training_data(n_rows: int = 2000) -> pd.DataFrame:
    """
    Create a synthetic dataset that has structured relationships between features
    and the probability of a future price drop.

    Intuition:
      - Each route has a base fare
      - Each airline and region add a markup factor
      - Farther from departure -> higher markup (more room to drop later)
      - "Future min price" is close to the base fare with some volatility
      - Current price = future min + markup
      - Label is derived from whether future_min_price_usd is at least
        DROP_THRESHOLD_PCT below current_price_usd.
    """
    rng = np.random.default_rng(42)

    # Define some routes with different base fares
    route_base_fares = {
        ("SJU", "JFK"): 320,
        ("SJU", "MCO"): 200,
        ("SJU", "MIA"): 190,
        ("SJU", "LAX"): 480,
        ("JFK", "SJU"): 320,
        ("MCO", "SJU"): 210,
        ("MIA", "SJU"): 195,
    }

    origins = list({r[0] for r in route_base_fares.keys()})
    destinations = list({r[1] for r in route_base_fares.keys()})

    airlines = ["JetBlue", "American", "Delta", "Spirit", "United"]
    # Airline-specific markup factors (e.g. legacy vs low-cost carriers)
    airline_markup = {
        "JetBlue": 1.03,
        "American": 1.07,
        "Delta": 1.08,
        "United": 1.05,
        "Spirit": 0.98,
    }

    regions = ["US", "EU", "LATAM"]
    region_markup = {
        "US": 1.08,
        "EU": 1.02,
        "LATAM": 0.98,
    }

    today = date.today()

    rows = []
    for _ in range(n_rows):
        # Route and carrier
        origin = rng.choice(origins)
        # ensure we pick a valid destination for this origin
        valid_dests = [d for (o, d) in route_base_fares.keys() if o == origin]
        destination = rng.choice(valid_dests)
        airline = rng.choice(airlines)
        region = rng.choice(regions)

        base_fare = route_base_fares.get((origin, destination), 280)

        # Time dimension
        days_until_dep = int(rng.integers(3, 91))  # 3 to 90 days out
        departure_date = today.toordinal() + days_until_dep
        departure_date = date.fromordinal(departure_date)

        # Pick search date a small number of days before departure
        days_before_search = int(rng.integers(0, min(21, days_until_dep)))
        search_date = departure_date.toordinal() - days_before_search
        search_date = date.fromordinal(search_date)

        # Volatility: long-haul routes and EU region can be slightly more volatile
        route_volatility = 1.0
        if destination in ("LAX",):
            route_volatility += 0.3
        if region == "EU":
            route_volatility += 0.1

        # Simulate future min price around the base fare
        future_min_price = base_fare * \
            (1 + rng.normal(0, 0.12) * route_volatility)

        future_min_price = max(future_min_price, 50.0)

        # Markup is larger:
        #  - when days_until_dep is large (booking very early)
        #  - for expensive airlines (legacy carriers)
        #  - in US region vs others
        time_markup_factor = 1 + 0.4 * (days_until_dep / 90.0)  # up to +40%
        carrier_factor = airline_markup[airline]
        region_factor = region_markup[region]

        # Random residual markup
        residual = rng.normal(0, 0.03)  # ±3% noise

        total_markup_multiplier = time_markup_factor * carrier_factor * region_factor
        total_markup_multiplier *= (1 + residual)

        current_price = future_min_price * total_markup_multiplier

        # Clamp to reasonable range
        current_price = max(current_price, 60.0)

        # Label: did price drop by at least DROP_THRESHOLD_PCT?
        # Compute a probability of price drop using a logistic-style weighting
        p = (
            # more days → higher drop chance
            0.50 * (days_until_dep / 90)
            # expensive airlines → higher drop chance
            + 0.25 * (airline_markup[airline] - 1.0)
            # small region influence
            + 0.10 * (region_markup[region] - 1.0)
            + 0.15 * rng.normal(0, 0.2)                    # reduced noise
        )

        # squash to 0–1 (sigmoid-like)
        p = 1 / (1 + np.exp(-p))
        price_drops = int(rng.random() < p)

        rows.append(
            {
                "origin": origin,
                "destination": destination,
                "airline": airline,
                "region": region,
                "departure_date": departure_date,
                "search_date": search_date,
                "days_until_departure": days_until_dep,
                "current_price_usd": float(current_price),
                "future_min_price_usd": float(future_min_price),
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


def build_ml_row_from_offer(offer: Union[Offer, Dict[str, Any]], search_date: date) -> Dict[str, Any]:
    """
    Convert an Offer (preferred) or legacy dict offer + today's date into a single ML row.

    Backward-compatible: supports both Offer objects and dicts.
    """
    # --- Pull fields depending on input type ---
    if isinstance(offer, Offer):
        origin = offer.origin
        destination = offer.destination
        airline = offer.airline or "Unknown"          # keep code for model consistency
        departure_date = offer.departure_date         # already a date
        current_price = float(offer.total_price_usd or 0.0)
        region = "US"  # Legacy feature in synthetic model; remove later when retraining on real history
    else:
        origin = offer.get("origin")
        destination = offer.get("destination")
        airline = offer.get("airline", "Unknown")
        region = offer.get("region", "US")
        departure_str = offer.get("departure_date")

        if isinstance(departure_str, date):
            departure_date = departure_str
        else:
            departure_date = date.fromisoformat(str(departure_str))

        current_price = float(offer.get("total_price_usd", 0.0))

    days_until_departure = (departure_date - search_date).days

    return {
        "origin": origin,
        "destination": destination,
        "airline": airline,
        "region": region,
        "departure_date": departure_date,
        "search_date": search_date,
        "days_until_departure": days_until_departure,
        "current_price_usd": current_price,
    }


def predict_price_drop_probability(
    model: Pipeline, offer: Union[Offer, Dict[str, Any]], search_date: date
) -> float:
    """
    Given a trained model and a current offer, return the probability that
    the price will drop in the next N_DAYS_WINDOW days.
    """
    ml_row = build_ml_row_from_offer(offer, search_date)
    X = pd.DataFrame([ml_row])
    proba = model.predict_proba(X)[0, 1]  # class 1 = price_drops
    return float(proba)
