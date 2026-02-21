# src/sqlite_history_store.py

from __future__ import annotations

import os
import sqlite3
from dataclasses import asdict
from datetime import datetime, date
from typing import List, Optional

import pandas as pd

from core.models import Offer


DEFAULT_DB_PATH = os.path.join("data", "price_history.sqlite")


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def _dt_to_str(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def _d_to_str(d: Optional[date]) -> Optional[str]:
    return d.isoformat() if d else None


def offer_signature(offer: Offer) -> str:
    """
    Stable signature for an itinerary based on segment chain.
    Same physical flight plan -> same signature.
    """
    parts: List[str] = []
    for it in (offer.itineraries or []):
        seg_parts: List[str] = []
        for s in (it.segments or []):
            dep = s.dep_at.isoformat() if s.dep_at else ""
            seg_parts.append("|".join([
                s.origin or "",
                s.destination or "",
                s.carrier_code or "",
                s.flight_number or "",
                dep,
            ]))
        parts.append(">".join(seg_parts))

    sig = "||".join(parts).strip()
    if sig:
        return sig

    # Fallback for offers without segments populated (should be rare now)
    return "|".join([
        offer.origin or "",
        offer.destination or "",
        _d_to_str(getattr(offer, "departure_date", None)) or "",
        _d_to_str(getattr(offer, "return_date", None)) or "",
        offer.airline or "",
        str(offer.stops_out or 0),
        str(offer.stops_return or 0),
    ])


class SqlitePriceHistoryStore:
    """
    SQLite-backed price observation store.
    Designed to be swapped later for Postgres with the same interface.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        _ensure_parent_dir(self.db_path)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS price_observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    search_ts TEXT NOT NULL,
                    provider TEXT,
                    origin TEXT NOT NULL,
                    destination TEXT NOT NULL,
                    trip_structure TEXT NOT NULL,
                    departure_date TEXT NOT NULL,
                    return_date TEXT,
                    passengers INTEGER,
                    max_stops_label TEXT,
                    flexible_dates INTEGER,

                    airline_code TEXT,
                    airline_name TEXT,
                    flight_number TEXT,
                    dep_time TEXT,
                    arr_time TEXT,

                    stops_out INTEGER,
                    stops_return INTEGER,

                    price_usd REAL NOT NULL,
                    currency TEXT,

                    offer_signature TEXT NOT NULL
                );
                """
            )

            # Helpful indexes for common queries
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_route_dep ON price_observations(origin, destination, departure_date);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_search_ts ON price_observations(search_ts);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_signature ON price_observations(offer_signature);"
            )

    def append_offers(
        self,
        offers: List[Offer],
        *,
        origin: str,
        destination: str,
        trip_structure: str,
        departure_date: date,
        return_date: Optional[date],
        passengers: int,
        max_stops_label: str,
        flexible_dates: bool,
        search_ts: Optional[datetime] = None,
        top_n: int = 30,
    ) -> int:
        """
        Append observations for up to top_n offers by price (to control volume).
        Returns number of rows inserted.
        """
        if not offers:
            return 0

        search_ts = search_ts or datetime.utcnow()

        # Control volume: store only top N cheapest offers
        offers_sorted = sorted(
            offers, key=lambda o: float(o.total_price_usd or 1e18))
        offers_to_log = offers_sorted[: max(1, int(top_n))]

        rows = []
        for o in offers_to_log:
            first_seg = None
            if o.itineraries and o.itineraries[0].segments:
                first_seg = o.itineraries[0].segments[0]

            rows.append(
                (
                    search_ts.isoformat(),
                    o.provider,
                    origin,
                    destination,
                    trip_structure,
                    departure_date.isoformat(),
                    return_date.isoformat() if return_date else None,
                    int(passengers),
                    max_stops_label,
                    1 if flexible_dates else 0,
                    (o.airline or None),
                    (o.airline_name or None),
                    (first_seg.flight_number if first_seg else None),
                    (_dt_to_str(first_seg.dep_at) if first_seg else None),
                    (_dt_to_str(first_seg.arr_at) if first_seg else None),
                    int(o.stops_out or 0),
                    int(o.stops_return or 0),
                    float(o.total_price_usd or 0.0),
                    (o.currency or "USD"),
                    offer_signature(o),
                )
            )

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO price_observations (
                    search_ts, provider, origin, destination, trip_structure,
                    departure_date, return_date, passengers, max_stops_label, flexible_dates,
                    airline_code, airline_name, flight_number, dep_time, arr_time,
                    stops_out, stops_return,
                    price_usd, currency,
                    offer_signature
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                rows,
            )

        return len(rows)

    def get_route_history(
        self,
        origin: str,
        destination: str,
        departure_date: date,
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Return historical observations for the route + departure_date.
        """
        with self._connect() as conn:
            df = pd.read_sql_query(
                """
                SELECT
                    search_ts AS search_datetime,
                    price_usd,
                    currency,
                    airline_name,
                    airline_code,
                    flight_number,
                    dep_time,
                    arr_time,
                    stops_out,
                    stops_return,
                    offer_signature
                FROM price_observations
                WHERE origin = ?
                  AND destination = ?
                  AND departure_date = ?
                ORDER BY search_ts ASC
                LIMIT ?;
                """,
                conn,
                params=(origin, destination,
                        departure_date.isoformat(), int(limit)),
            )
        return df

    def get_market_trend(
        self,
        origin: str,
        destination: str,
        departure_date: date,
        limit_days: int = 365,
    ) -> pd.DataFrame:
        """
        Market trend = daily minimum observed price for route + departure_date.
        (NOTE: currently operates on stored numeric values as-is.)
        """
        with self._connect() as conn:
            df = pd.read_sql_query(
                """
                SELECT
                    substr(search_ts, 1, 10) AS search_day,   -- YYYY-MM-DD
                    MIN(price_usd) AS min_price_usd,
                    COUNT(*) AS observations
                FROM price_observations
                WHERE origin = ?
                AND destination = ?
                AND departure_date = ?
                GROUP BY substr(search_ts, 1, 10)
                ORDER BY search_day ASC
                LIMIT ?;
                """,
                conn,
                params=(origin, destination,
                        departure_date.isoformat(), int(limit_days)),
            )
        return df

    def get_market_trend_usd_dual(
        self,
        origin: str,
        destination: str,
        departure_date: date,
        fx_service,
        limit: int = 5000,
    ) -> tuple[str, pd.DataFrame]:
        """
        Dual-mode market trend with USD conversion at extraction time.

        Returns (mode, df):
          - mode == "raw":  df columns: [search_datetime, price_usd, currency, observations]
          - mode == "daily": df columns: [search_day, min_price_usd, observations]
        """
        with self._connect() as conn:
            df = pd.read_sql_query(
                """
                SELECT search_ts, price_usd, currency
                FROM price_observations
                WHERE origin = ?
                  AND destination = ?
                  AND departure_date = ?
                ORDER BY search_ts ASC
                LIMIT ?;
                """,
                conn,
                params=(origin, destination,
                        departure_date.isoformat(), int(limit)),
            )

        if df.empty:
            return "raw", df

        # Parse timestamps (your stored search_ts is naive ISO; treat as UTC)
        df["search_ts"] = pd.to_datetime(
            df["search_ts"], errors="coerce", utc=True)
        df = df.dropna(subset=["search_ts"])

        if df.empty:
            return "raw", df

        # Convert to USD (stored price_usd currently holds native amount)
        def _to_usd(row) -> float:
            amt = float(row["price_usd"])
            cur = str(row["currency"] or "USD")
            ts = row["search_ts"].to_pydatetime()  # tz-aware UTC
            return float(fx_service.amount_to_usd(amt, cur, ts))

        df["price_usd"] = df.apply(_to_usd, axis=1)

        # Decide mode based on distinct days present
        df["search_day"] = df["search_ts"].dt.strftime("%Y-%m-%d")
        n_days = int(df["search_day"].nunique())

        if n_days <= 1:
            out = df.rename(columns={"search_ts": "search_datetime"}).copy()
            out["observations"] = 1
            return "raw", out[["search_datetime", "price_usd", "currency", "observations"]]

        daily = (
            df.groupby("search_day", as_index=False)
            .agg(min_price_usd=("price_usd", "min"), observations=("price_usd", "count"))
            .sort_values("search_day")
        )
        return "daily", daily
