"""Storage contract for price observations.

An *observation* is "we saw this itinerary at this price at this moment".
The SQLite implementation is the default; the protocol exists so a Postgres or
warehouse-backed implementation can replace it without touching callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol

import pandas as pd

from airfare.domain.models import Offer, SearchQuery


@dataclass(frozen=True, slots=True)
class Observation:
    search_ts: datetime
    provider: str
    origin: str
    destination: str
    departure_date: date
    return_date: date | None
    passengers: int
    airline_code: str
    airline_name: str | None
    flight_number: str | None
    dep_at: datetime | None
    arr_at: datetime | None
    stops_out: int
    stops_return: int
    duration_minutes: int | None
    price_amount: float
    currency: str
    fx_rate: float | None
    price_usd: float | None
    signature: str

    @classmethod
    def from_offer(cls, offer: Offer, query: SearchQuery, search_ts: datetime) -> Observation:
        out = offer.outbound
        first = out.segments[0] if out and out.segments else None
        return cls(
            search_ts=search_ts,
            provider=offer.provider,
            origin=offer.origin,
            destination=offer.destination,
            departure_date=offer.departure_date,
            return_date=offer.return_date,
            passengers=query.passengers,
            airline_code=offer.airline_code,
            airline_name=offer.airline_name,
            flight_number=first.flight_number if first else None,
            dep_at=out.dep_at if out else None,
            arr_at=out.arr_at if out else None,
            stops_out=offer.stops_out,
            stops_return=offer.stops_return,
            duration_minutes=offer.total_duration_minutes,
            price_amount=offer.price.amount,
            currency=offer.price.currency,
            fx_rate=offer.price.fx_rate,
            price_usd=offer.price.usd,
            signature=offer.signature,
        )


class PriceHistoryRepository(Protocol):
    def record(self, observations: list[Observation]) -> int:
        """Persist observations; returns the number of rows written."""

    def route_observations(
        self, origin: str, destination: str, departure_date: date | None = None
    ) -> pd.DataFrame:
        """All observations for a route (optionally one departure date), oldest first."""

    def daily_min_trend(self, origin: str, destination: str, departure_date: date) -> pd.DataFrame:
        """Columns: search_day, min_price_usd, median_price_usd, observations."""

    def route_price_stats(self, origin: str, destination: str) -> dict[str, float]:
        """Summary of observed USD prices for a route: count, min, p25, median, p75, max."""

    def latest_offers(self, query: SearchQuery) -> list[Offer]:
        """Most recent snapshot matching the query, rebuilt as Offers (for the Replay provider)."""
