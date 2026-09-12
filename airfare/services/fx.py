"""Daily FX rates to USD with a SQLite cache and two sources.

Twelve Data is used when an API key is configured; otherwise (or on failure)
the keyless Frankfurter API (ECB reference rates) is used. Rates are keyed by
UTC day so a price observed on a given day is always converted the same way.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Protocol

import requests

log = logging.getLogger(__name__)


class FxUnavailableError(Exception):
    """No source could provide a rate."""


class FxSource(Protocol):
    name: str

    def rate_to_usd(self, currency: str, day: date) -> float: ...


class FrankfurterSource:
    """ECB reference rates via https://frankfurter.dev — no key required."""

    name = "frankfurter"

    def __init__(self, timeout: float = 10.0, session: requests.Session | None = None) -> None:
        self.timeout = timeout
        self.http = session or requests.Session()

    def rate_to_usd(self, currency: str, day: date) -> float:
        # Frankfurter returns the latest available rate on or before the requested day.
        resp = self.http.get(
            f"https://api.frankfurter.dev/v1/{day.isoformat()}",
            params={"base": currency, "symbols": "USD"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return float(resp.json()["rates"]["USD"])


class TwelveDataSource:
    name = "twelvedata"

    def __init__(
        self, api_key: str, timeout: float = 10.0, session: requests.Session | None = None
    ) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.http = session or requests.Session()

    def rate_to_usd(self, currency: str, day: date) -> float:
        start = (day - timedelta(days=7)).isoformat()
        resp = self.http.get(
            "https://api.twelvedata.com/time_series",
            params={
                "symbol": f"{currency}/USD",
                "interval": "1day",
                "start_date": start,
                "end_date": day.isoformat(),
                "timezone": "UTC",
                "apikey": self.api_key,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") == "error":
            raise ValueError(payload.get("message", "twelvedata error"))
        values = sorted(payload.get("values") or [], key=lambda v: v["datetime"])
        eligible = [v for v in values if v["datetime"][:10] <= day.isoformat()]
        if not eligible:
            raise ValueError(f"no {currency}/USD close on or before {day}")
        return float(eligible[-1]["close"])


class FxService:
    def __init__(self, db_path: Path | str, sources: list[FxSource]) -> None:
        self.db_path = Path(db_path)
        self.sources = sources
        self._mem: dict[tuple[str, str], float] = {}

    @classmethod
    def from_settings(cls, db_path: Path | str, twelvedata_api_key: str = "") -> FxService:
        sources: list[FxSource] = []
        if twelvedata_api_key:
            sources.append(TwelveDataSource(twelvedata_api_key))
        sources.append(FrankfurterSource())
        return cls(db_path, sources)

    # -- cache ----------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """CREATE TABLE IF NOT EXISTS fx_rates_daily (
                pair TEXT NOT NULL, day_utc TEXT NOT NULL, rate REAL NOT NULL,
                source TEXT NOT NULL, fetched_at_utc TEXT NOT NULL, PRIMARY KEY (pair, day_utc))"""
        )
        return conn

    def _cache_get(self, pair: str, day: str) -> float | None:
        if (pair, day) in self._mem:
            return self._mem[(pair, day)]
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT rate FROM fx_rates_daily WHERE pair=? AND day_utc=?", (pair, day)
            ).fetchone()
        if row:
            self._mem[(pair, day)] = float(row[0])
            return float(row[0])
        return None

    def _cache_put(self, pair: str, day: str, rate: float, source: str) -> None:
        self._mem[(pair, day)] = rate
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO fx_rates_daily VALUES (?,?,?,?,?)",
                (pair, day, rate, source, datetime.now(UTC).isoformat()),
            )

    # -- api ------------------------------------------------------------------

    def rate_to_usd(self, currency: str, at: datetime | date | None = None) -> float:
        """USD per 1 unit of ``currency`` on the UTC day of ``at`` (default: today)."""
        cur = currency.upper().strip()
        if cur == "USD":
            return 1.0
        day = _utc_day(at)
        pair = f"{cur}/USD"
        cached = self._cache_get(pair, day.isoformat())
        if cached is not None:
            return cached
        errors: list[str] = []
        for src in self.sources:
            try:
                rate = src.rate_to_usd(cur, day)
            except Exception as e:
                errors.append(f"{src.name}: {e}")
                continue
            self._cache_put(pair, day.isoformat(), rate, src.name)
            return rate
        raise FxUnavailableError(f"No FX source could price {pair} on {day}: " + "; ".join(errors))

    def to_usd(self, amount: float, currency: str, at: datetime | date | None = None) -> float:
        return amount * self.rate_to_usd(currency, at)


def _utc_day(at: datetime | date | None) -> date:
    if at is None:
        return datetime.now(UTC).date()
    if isinstance(at, datetime):
        return (at if at.tzinfo else at.replace(tzinfo=UTC)).astimezone(UTC).date()
    return at
