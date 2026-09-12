from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from airfare.domain.models import Price, SearchQuery
from airfare.ml.dataset import build_dataset, daily_prices
from airfare.ml.features import FEATURES, LABEL
from airfare.ml.train import train_from_observations
from airfare.storage.repository import Observation
from airfare.storage.sqlite import SqlitePriceHistory

from tests.conftest import make_offer


def _obs_frame(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """rows: (signature, search_day 'YYYY-MM-DD', price_usd)."""
    return pd.DataFrame(
        {
            "signature": [r[0] for r in rows],
            "search_ts": [f"{r[1]}T12:00:00+00:00" for r in rows],
            "price_usd": [r[2] for r in rows],
            "origin": "SJU",
            "destination": "JFK",
            "departure_date": "2026-10-20",
            "airline_code": "B6",
            "stops_out": 0,
            "stops_return": 1,
        }
    )


def test_daily_prices_keeps_min_per_day() -> None:
    df = daily_prices(
        _obs_frame([("a", "2026-09-01", 300), ("a", "2026-09-01", 280), ("a", "2026-09-02", 290)])
    )
    assert df["price_usd"].tolist() == [280, 290]


def test_labels_and_censoring() -> None:
    rows = [
        ("a", "2026-09-01", 300),  # future min within 7d = 270 -> drop (>=5%)
        ("a", "2026-09-03", 270),  # future min over 9/04-9/10 = 294 -> no drop
        ("a", "2026-09-08", 295),  # 9/09 is within window -> future 294 -> no drop
        ("a", "2026-09-09", 294),  # no future observation -> censored
        ("a", "2026-09-20", 250),  # isolated -> censored
        ("b", "2026-09-01", 500),  # only observation -> censored
    ]
    ds = build_dataset(_obs_frame(rows))
    assert len(ds) == 3
    assert ds[LABEL].tolist() == [1, 0, 0]
    assert ds["future_min_price_usd"].tolist() == [270, 294, 294]
    assert set(FEATURES).issubset(ds.columns)
    first = ds.iloc[0]
    assert first["days_until_departure"] == (date(2026, 10, 20) - date(2026, 9, 1)).days
    assert first["total_stops"] == 1 and first["search_dow"] == date(2026, 9, 1).weekday()
    assert ds["search_date"].is_monotonic_increasing


def test_empty_input() -> None:
    ds = build_dataset(pd.DataFrame(columns=["signature", "search_ts", "price_usd"]))
    assert ds.empty and LABEL in ds.columns


def test_train_from_observations_requires_enough_rows(tmp_path: Path) -> None:
    history = SqlitePriceHistory(tmp_path / "h.sqlite")
    with pytest.raises(ValueError, match="labeled rows"):
        train_from_observations(tmp_path / "models", history)


def test_train_from_observations_end_to_end(tmp_path: Path) -> None:

    rng = np.random.default_rng(0)
    history = SqlitePriceHistory(tmp_path / "h.sqlite")
    q = SearchQuery("SJU", "JFK", date(2026, 12, 1))
    obs = []
    for sig in range(30):  # 30 itineraries observed daily for 20 days
        base = 200 + sig * 10
        drift = rng.choice([-4.0, 0.0, 3.0])
        for day in range(20):
            price = base + drift * day + rng.normal(0, 3)
            o = replace(
                make_offer(
                    price=price, carrier=["B6", "AA", "DL"][sig % 3], dep_date=date(2026, 12, 1)
                ),
                price=Price(price, "USD", usd=float(price), fx_rate=1.0),
                signature=f"sig{sig}",
            )
            ts = pd.Timestamp(date(2026, 9, 1) + timedelta(days=day), tz="UTC").to_pydatetime()
            obs.append(Observation.from_offer(o, q, ts))
    history.record(obs)
    card = train_from_observations(tmp_path / "models", history)
    assert card.data_source == "observations" and not card.is_synthetic
    assert card.n_train + card.n_valid >= 300
    assert 0.0 <= card.metrics["brier"] <= 0.5
    assert (tmp_path / "models" / "price_drop" / "card.json").exists()
