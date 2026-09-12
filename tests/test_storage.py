import sqlite3
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
from airfare.domain.models import Price, SearchQuery
from airfare.storage.repository import Observation
from airfare.storage.sqlite import SqlitePriceHistory

from tests.conftest import make_offer


def _obs(price: float, ts: datetime, query: SearchQuery) -> Observation:
    o = replace(
        make_offer(price=price),
        price=Price(price, "USD", usd=price, fx_rate=1.0),
        signature=f"sig{price}",
    )
    return Observation.from_offer(o, query, ts)


def test_record_and_trend(tmp_path: Path, query: SearchQuery) -> None:
    repo = SqlitePriceHistory(tmp_path / "h.sqlite")
    t0 = datetime(2026, 9, 1, 12, tzinfo=UTC)
    n = repo.record(
        [_obs(300, t0, query), _obs(280, t0, query), _obs(250, t0 + timedelta(days=1), query)]
    )
    assert n == 3
    assert repo.record([_obs(300, t0, query)]) == 0  # duplicate (signature, ts) ignored
    trend = repo.daily_min_trend("SJU", "JFK", date(2026, 10, 1))
    assert trend["min_price_usd"].tolist() == [280.0, 250.0]
    assert trend["observations"].tolist() == [2, 1]
    stats = repo.route_price_stats("SJU", "JFK")
    assert stats["count"] == 3 and stats["min"] == 250.0
    replay = repo.latest_offers(query)
    assert [o.price.amount for o in replay] == [250.0]
    assert replay[0].signature == "sig250"


def test_migrates_legacy_schema(tmp_path: Path) -> None:
    db = tmp_path / "legacy.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute(
            """CREATE TABLE price_observations (id INTEGER PRIMARY KEY, search_ts TEXT, provider TEXT,
            origin TEXT, destination TEXT, trip_structure TEXT, departure_date TEXT, return_date TEXT,
            passengers INTEGER, max_stops_label TEXT, flexible_dates INTEGER, airline_code TEXT,
            airline_name TEXT, flight_number TEXT, dep_time TEXT, arr_time TEXT, stops_out INTEGER,
            stops_return INTEGER, price_usd REAL, currency TEXT, offer_signature TEXT)"""
        )
        conn.execute(
            "INSERT INTO price_observations VALUES (1,'2026-02-15T20:48:56','amadeus','SJU','JAX','Roundtrip',"
            "'2026-02-26','2026-03-01',1,'Up to 1 stop',0,'B6','JETBLUE','704','2026-02-26T07:00:00',"
            "'2026-02-26T10:00:00',0,0,251.3,'EUR','SJU|JAX|B6|704')"
        )
        conn.execute("CREATE TABLE fx_rates (x INTEGER)")
    repo = SqlitePriceHistory(db)
    df = repo.route_observations("SJU", "JAX")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["price_amount"] == 251.3 and row["currency"] == "EUR"
    assert pd.isna(row["price_usd"])  # legacy EUR amounts are not USD
    with repo.connect() as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "price_observations" not in names and "fx_rates" not in names
    assert SqlitePriceHistory(db).route_observations("SJU", "JAX").shape[0] == 1  # idempotent
