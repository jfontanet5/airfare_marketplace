# src/services/fx_rate_services.py

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Tuple

import requests


class FxRateService:
    """
    Daily FX using Twelve Data with a SQLite cache.

    Contract:
      - get_rate_to_usd(currency, ts) -> USD per 1 unit of `currency` (daily close, UTC day)
      - amount_to_usd(amount, currency, ts) -> amount in USD
    """

    def __init__(
        self,
        db_path: str = os.path.join("data", "price_history.sqlite"),
        api_key: Optional[str] = None,
        timeout_s: int = 20,
        lookback_days: int = 10,
    ):
        self.db_path = db_path
        self.api_key = api_key or os.getenv("TWELVEDATA_API_KEY", "").strip()
        if not self.api_key:
            raise ValueError("Missing TWELVEDATA_API_KEY in environment/.env")

        self.timeout_s = timeout_s
        self.lookback_days = max(2, int(lookback_days))
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
                CREATE TABLE IF NOT EXISTS fx_rates_daily (
                    pair TEXT NOT NULL,
                    day_utc TEXT NOT NULL,          -- YYYY-MM-DD
                    rate REAL NOT NULL,             -- USD per 1 unit of base currency
                    source TEXT NOT NULL,
                    fetched_at_utc TEXT NOT NULL,
                    PRIMARY KEY (pair, day_utc)
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_fx_day ON fx_rates_daily(day_utc);"
            )

    @staticmethod
    def _to_utc(ts: datetime) -> datetime:
        # Treat naive timestamps as UTC (matches your search_ts logging).
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)

    def _cache_get_daily(self, pair: str, day_utc: str) -> Optional[float]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT rate FROM fx_rates_daily WHERE pair = ? AND day_utc = ?;",
                (pair, day_utc),
            )
            row = cur.fetchone()
            return float(row[0]) if row else None

    def _cache_put_daily(self, pair: str, day_utc: str, rate: float, source: str = "twelvedata") -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO fx_rates_daily (pair, day_utc, rate, source, fetched_at_utc)
                VALUES (?, ?, ?, ?, ?);
                """,
                (pair, day_utc, float(rate), source, now),
            )

    def _candidate_symbols(self, pair: str) -> List[str]:
        """
        Twelve Data often supports 'EUR/USD'. Some providers also accept 'EURUSD'.
        We try the canonical form first, then a no-slash fallback.
        """
        p = pair.strip()
        if "/" in p:
            return [p, p.replace("/", "")]
        return [p]

    def _fetch_daily_series_from_twelvedata(
        self, symbol: str, start_day_utc: str, end_day_utc: str
    ) -> List[Dict[str, Any]]:
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": symbol,
            "interval": "1day",
            # For daily interval, TwelveData is happiest with YYYY-MM-DD (no time component)
            "start_date": start_day_utc,
            "end_date": end_day_utc,
            "apikey": self.api_key,
            "timezone": "UTC",
            "format": "JSON",
            "outputsize": 5000,
        }

        r = requests.get(url, params=params, timeout=self.timeout_s)
        r.raise_for_status()
        payload: Dict[str, Any] = r.json()

        if str(payload.get("status", "")).lower() == "error":
            raise ValueError(
                f"TwelveData error for {symbol}: {payload.get('message')}")

        return payload.get("values") or []

    @staticmethod
    def _pick_rate_for_day(values: List[Dict[str, Any]], day_utc: str) -> Optional[float]:
        """
        values items have:
          - datetime: 'YYYY-MM-DD'
          - close: string
        We select the close for the most recent day <= day_utc.
        """
        if not values:
            return None

        # Normalize and sort ascending by datetime
        cleaned: List[Tuple[str, float]] = []
        for v in values:
            d = v.get("datetime")
            c = v.get("close")
            if not d or c is None:
                continue
            try:
                cleaned.append((str(d), float(c)))
            except Exception:
                continue

        cleaned.sort(key=lambda x: x[0])
        # Find last <= target day
        candidates = [rate for (d, rate) in cleaned if d <= day_utc]
        if candidates:
            return candidates[-1]
        # If nothing <= day, return earliest (should be rare)
        return cleaned[0][1] if cleaned else None

    def _fetch_daily_rate_from_twelvedata(self, pair: str, day_utc: str) -> float:
        """
        Fetch a daily FX close for `pair` on `day_utc`.
        If data isn't available on that exact date (e.g., weekend/holiday),
        we take the most recent available close BEFORE that day within lookback window.
        """
        end_dt = datetime.strptime(
            day_utc, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        start_dt = end_dt - timedelta(days=self.lookback_days)
        start_day = start_dt.strftime("%Y-%m-%d")
        end_day = end_dt.strftime("%Y-%m-%d")

        last_err: Optional[Exception] = None
        for sym in self._candidate_symbols(pair):
            try:
                values = self._fetch_daily_series_from_twelvedata(
                    sym, start_day, end_day)
                rate = self._pick_rate_for_day(values, day_utc)
                if rate is None:
                    raise RuntimeError(
                        f"No daily FX values returned for {sym} in {start_day}..{end_day}")
                return float(rate)
            except Exception as e:
                last_err = e

        raise RuntimeError(
            f"Unable to fetch FX for {pair} on {day_utc}. Last error: {last_err}")

    def _resolve_pair_to_usd(self, currency: str) -> str:
        c = currency.upper().strip()
        if c == "USD":
            return "USD/USD"
        # We want USD per 1 unit of currency, i.e., EUR/USD
        return f"{c}/USD"

    def get_rate_to_usd(self, currency: str, ts: datetime) -> float:
        c = currency.upper().strip() if currency else "USD"
        if c == "USD":
            return 1.0

        ts_utc = self._to_utc(ts)
        day_utc = ts_utc.strftime("%Y-%m-%d")

        pair = self._resolve_pair_to_usd(c)

        cached = self._cache_get_daily(pair, day_utc)
        if cached is not None:
            return float(cached)

        rate = self._fetch_daily_rate_from_twelvedata(pair, day_utc)
        self._cache_put_daily(pair, day_utc, rate)
        return float(rate)

    def amount_to_usd(self, amount: float, currency: str, ts: datetime) -> float:
        rate = self.get_rate_to_usd(currency, ts)
        return float(amount) * float(rate)
