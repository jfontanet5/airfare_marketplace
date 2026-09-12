"""SQLite implementation of :class:`PriceHistoryRepository` with versioned migrations."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from airfare.domain.models import Itinerary, Offer, Price, SearchQuery, Segment
from airfare.storage.repository import Observation

log = logging.getLogger(__name__)

# Each entry is applied once, in order; the version is recorded in schema_version.
MIGRATIONS: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS observations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        search_ts TEXT NOT NULL,
        provider TEXT NOT NULL,
        origin TEXT NOT NULL,
        destination TEXT NOT NULL,
        departure_date TEXT NOT NULL,
        return_date TEXT,
        passengers INTEGER NOT NULL DEFAULT 1,
        airline_code TEXT NOT NULL,
        airline_name TEXT,
        flight_number TEXT,
        dep_at TEXT,
        arr_at TEXT,
        stops_out INTEGER NOT NULL,
        stops_return INTEGER NOT NULL,
        duration_minutes INTEGER,
        price_amount REAL NOT NULL,
        currency TEXT NOT NULL,
        fx_rate REAL,
        price_usd REAL,
        signature TEXT NOT NULL,
        UNIQUE (signature, search_ts)
    );
    CREATE INDEX IF NOT EXISTS ix_obs_route_dep ON observations (origin, destination, departure_date);
    CREATE INDEX IF NOT EXISTS ix_obs_signature_ts ON observations (signature, search_ts);
    CREATE TABLE IF NOT EXISTS fx_rates_daily (
        pair TEXT NOT NULL,
        day_utc TEXT NOT NULL,
        rate REAL NOT NULL,
        source TEXT NOT NULL,
        fetched_at_utc TEXT NOT NULL,
        PRIMARY KEY (pair, day_utc)
    );
    CREATE TABLE IF NOT EXISTS search_cache (
        key TEXT PRIMARY KEY,
        created_at_utc TEXT NOT NULL,
        payload TEXT NOT NULL
    );
    """,
    # v2: migrate legacy `price_observations` (pre-0.5 schema stored native amounts in price_usd).
    """
    INSERT OR IGNORE INTO observations (
        search_ts, provider, origin, destination, departure_date, return_date, passengers,
        airline_code, airline_name, flight_number, dep_at, arr_at, stops_out, stops_return,
        duration_minutes, price_amount, currency, fx_rate, price_usd, signature)
    SELECT search_ts, COALESCE(provider,'unknown'), origin, destination, departure_date, return_date,
           COALESCE(passengers,1), COALESCE(airline_code,''), airline_name, flight_number, dep_time, arr_time,
           COALESCE(stops_out,0), COALESCE(stops_return,0), NULL, price_usd, COALESCE(currency,'USD'),
           NULL, CASE WHEN COALESCE(currency,'USD')='USD' THEN price_usd ELSE NULL END, offer_signature
    FROM price_observations;
    DROP TABLE IF EXISTS price_observations;
    DROP TABLE IF EXISTS fx_rates;
    """,
]


def _iso(v: datetime | date | None) -> str | None:
    return v.isoformat() if v else None


class SqlitePriceHistory:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, detect_types=0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _migrate(self) -> None:
        with self.connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
            row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
            current = int(row[0] or 0)
            legacy_exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='price_observations'"
            ).fetchone()
            for version, sql in enumerate(MIGRATIONS, start=1):
                if version <= current:
                    continue
                if version == 2 and not legacy_exists:
                    conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
                    continue
                log.info("applying migration v%d to %s", version, self.db_path)
                conn.executescript(sql)
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))

    # -- writes ---------------------------------------------------------------

    def record(self, observations: list[Observation]) -> int:
        if not observations:
            return 0
        rows = [
            (
                _iso(o.search_ts), o.provider, o.origin, o.destination, _iso(o.departure_date),
                _iso(o.return_date), o.passengers, o.airline_code, o.airline_name, o.flight_number,
                _iso(o.dep_at), _iso(o.arr_at), o.stops_out, o.stops_return, o.duration_minutes,
                o.price_amount, o.currency, o.fx_rate, o.price_usd, o.signature,
            )
            for o in observations
        ]  # fmt: skip
        with self.connect() as conn:
            cur = conn.executemany(
                """INSERT OR IGNORE INTO observations (
                    search_ts, provider, origin, destination, departure_date, return_date, passengers,
                    airline_code, airline_name, flight_number, dep_at, arr_at, stops_out, stops_return,
                    duration_minutes, price_amount, currency, fx_rate, price_usd, signature
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
            return int(cur.rowcount)

    # -- reads ----------------------------------------------------------------

    def route_observations(
        self, origin: str, destination: str, departure_date: date | None = None
    ) -> pd.DataFrame:
        sql = "SELECT * FROM observations WHERE origin=? AND destination=?"
        params: list[str] = [origin, destination]
        if departure_date:
            sql += " AND departure_date=?"
            params.append(departure_date.isoformat())
        sql += " ORDER BY search_ts ASC"
        with self.connect() as conn:
            df = pd.read_sql_query(sql, conn, params=tuple(params))
        if not df.empty:
            df["search_ts"] = pd.to_datetime(df["search_ts"], utc=True)
        return df

    def daily_min_trend(self, origin: str, destination: str, departure_date: date) -> pd.DataFrame:
        df = self.route_observations(origin, destination, departure_date)
        df = df.dropna(subset=["price_usd"])
        if df.empty:
            return pd.DataFrame(
                columns=["search_day", "min_price_usd", "median_price_usd", "observations"]
            )
        df["search_day"] = df["search_ts"].dt.floor("D")
        out = (
            df.groupby("search_day")["price_usd"]
            .agg(min_price_usd="min", median_price_usd="median", observations="count")
            .reset_index()
        )
        return out

    def route_price_stats(self, origin: str, destination: str) -> dict[str, float]:
        df = self.route_observations(origin, destination).dropna(subset=["price_usd"])
        if df.empty:
            return {}
        s = df["price_usd"]
        return {
            "count": float(len(s)),
            "min": float(s.min()),
            "p25": float(s.quantile(0.25)),
            "median": float(s.median()),
            "p75": float(s.quantile(0.75)),
            "max": float(s.max()),
        }

    def routes(self) -> pd.DataFrame:
        with self.connect() as conn:
            return pd.read_sql_query(
                """SELECT origin, destination, COUNT(*) AS observations,
                          MIN(substr(search_ts,1,10)) AS first_seen, MAX(substr(search_ts,1,10)) AS last_seen
                   FROM observations WHERE price_usd IS NOT NULL
                   GROUP BY origin, destination ORDER BY observations DESC""",
                conn,
            )

    def cheapest_by_departure(self, origin: str, destination: str) -> pd.DataFrame:
        with self.connect() as conn:
            return pd.read_sql_query(
                """SELECT departure_date, MIN(price_usd) AS min_price_usd,
                          MAX(substr(search_ts,1,10)) AS last_seen
                   FROM observations
                   WHERE origin=? AND destination=? AND price_usd IS NOT NULL
                   GROUP BY departure_date ORDER BY departure_date""",
                conn,
                params=(origin, destination),
            )

    def latest_offers(self, query: SearchQuery) -> list[Offer]:
        with self.connect() as conn:
            latest = conn.execute(
                "SELECT MAX(search_ts) FROM observations WHERE origin=? AND destination=?",
                (query.origin, query.destination),
            ).fetchone()[0]
            if not latest:
                return []
            df = pd.read_sql_query(
                "SELECT * FROM observations WHERE origin=? AND destination=? AND search_ts=?",
                conn,
                params=[query.origin, query.destination, latest],
            )
        return [_row_to_offer(r) for r in df.to_dict("records")]


def _row_to_offer(r: dict[Any, Any]) -> Offer:
    """Rebuild a summary Offer from a stored row.

    Only the outbound summary (first departure, final arrival, duration) is stored,
    so replayed offers carry a single synthetic segment per direction.
    """
    dep_at = datetime.fromisoformat(str(r["dep_at"])) if r.get("dep_at") else None
    arr_at = datetime.fromisoformat(str(r["arr_at"])) if r.get("arr_at") else None
    seg = Segment(
        origin=str(r["origin"]),
        destination=str(r["destination"]),
        dep_at=dep_at,
        arr_at=arr_at,
        carrier_code=str(r["airline_code"]) or None,
        carrier_name=str(r["airline_name"]) if r.get("airline_name") else None,
        flight_number=str(r["flight_number"]) if r.get("flight_number") else None,
    )
    raw_duration = r.get("duration_minutes")
    duration = int(raw_duration) if raw_duration is not None and not pd.isna(raw_duration) else None
    return Offer(
        provider="replay",
        origin=str(r["origin"]),
        destination=str(r["destination"]),
        departure_date=date.fromisoformat(str(r["departure_date"])),
        return_date=date.fromisoformat(str(r["return_date"])) if r.get("return_date") else None,
        price=Price(amount=float(r["price_amount"]), currency=str(r["currency"])),
        itineraries=(Itinerary(segments=(seg,), duration_minutes=duration),),
        airline_code=str(r["airline_code"]),
        airline_name=str(r["airline_name"]) if r.get("airline_name") else None,
        signature=str(r["signature"]),
    )
