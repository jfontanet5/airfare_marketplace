"""Deterministic offline provider.

Produces realistic-looking itineraries (segments, times, durations, layovers),
price variation across dates, a mix of carriers and currencies, and a couple of
deliberate duplicates so dedup and FX normalization can be exercised without
any network access. Output is a pure function of the query.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise

from airfare.domain.models import Itinerary, Offer, Price, SearchQuery, Segment
from airfare.providers.base import FlightSearchProvider

_CARRIERS: list[tuple[str, str, float]] = [  # code, name, fare multiplier
    ("B6", "JetBlue Airways", 1.00),
    ("AA", "American Airlines", 1.08),
    ("DL", "Delta Air Lines", 1.10),
    ("UA", "United Airlines", 1.06),
    ("NK", "Spirit Airlines", 0.82),
    ("IB", "Iberia", 1.04),
]
_HUBS = ["MIA", "JFK", "ATL", "CLT", "FLL"]


def _seed(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def _base_fare(origin: str, destination: str) -> float:
    # Stable pseudo-distance from the route string; roughly $180-$780.
    return 180 + (_seed(origin, destination) % 600)


def _itinerary(
    origin: str, destination: str, day: date, *, stops: int, carrier: str, hour: int, seed: int
) -> Itinerary:
    legs = [origin, *(_HUBS[(seed + i) % len(_HUBS)] for i in range(stops)), destination]
    legs = [
        leg if leg not in (origin, destination) or i in (0, len(legs) - 1) else "DFW"
        for i, leg in enumerate(legs)
    ]
    segments: list[Segment] = []
    cursor = datetime(day.year, day.month, day.day, hour, 5 + seed % 50, tzinfo=UTC)
    for i, (a, b) in enumerate(pairwise(legs)):
        flight_minutes = 95 + (_seed(a, b) % 200)
        arr = cursor + timedelta(minutes=flight_minutes)
        segments.append(
            Segment(
                origin=a,
                destination=b,
                dep_at=cursor,
                arr_at=arr,
                carrier_code=carrier,
                carrier_name=next(n for c, n, _ in _CARRIERS if c == carrier),
                flight_number=str(100 + (seed + i * 7) % 900),
                aircraft_code="32N" if i % 2 == 0 else "738",
            )
        )
        cursor = arr + timedelta(minutes=55 + (seed + i) % 90)  # layover
    total = int((segments[-1].arr_at - segments[0].dep_at).total_seconds() // 60)  # type: ignore[operator]
    return Itinerary(segments=tuple(segments), duration_minutes=total)


class MockProvider(FlightSearchProvider):
    name = "mock"

    def _offers_for_day(self, q: SearchQuery, offset: int) -> list[Offer]:
        dep = q.departure_date + timedelta(days=offset)
        ret = dep + q.trip_length if q.trip_length is not None else None
        base = _base_fare(q.origin, q.destination)
        # Demand curve: weekends and near-term departures cost more.
        days_out = max(0, (dep - date.today()).days)
        demand = 1.0 + (0.25 if dep.weekday() >= 5 else 0.0) + max(0.0, 0.35 - days_out * 0.005)
        offers: list[Offer] = []

        for idx, (code, name, mult) in enumerate(_CARRIERS):
            stops = idx % 3  # 0,1,2,0,1,2
            seed = _seed(q.origin, q.destination, dep, code)
            price_amount = round(base * mult * demand * (1 + (seed % 17) / 100), 2)
            currency = "EUR" if code == "IB" else "USD"
            if currency == "EUR":
                price_amount = round(price_amount * 0.86, 2)
            itineraries = [
                _itinerary(
                    q.origin,
                    q.destination,
                    dep,
                    stops=stops,
                    carrier=code,
                    hour=6 + idx * 2,
                    seed=seed,
                )
            ]
            if ret:
                itineraries.append(
                    _itinerary(
                        q.destination,
                        q.origin,
                        ret,
                        stops=stops,
                        carrier=code,
                        hour=9 + idx,
                        seed=seed + 1,
                    )
                )
            offers.append(
                Offer(
                    provider=self.name,
                    origin=q.origin,
                    destination=q.destination,
                    departure_date=dep,
                    return_date=ret,
                    price=Price(amount=price_amount * q.passengers, currency=currency),
                    itineraries=tuple(itineraries),
                    airline_code=code,
                    airline_name=name,
                )
            )

        # A duplicate of the first offer at a slightly worse price — dedup must drop it.
        first = offers[0]
        offers.append(
            Offer(
                provider=first.provider,
                origin=first.origin,
                destination=first.destination,
                departure_date=first.departure_date,
                return_date=first.return_date,
                price=Price(
                    amount=round(first.price.amount * 1.04, 2), currency=first.price.currency
                ),
                itineraries=first.itineraries,
                airline_code=first.airline_code,
                airline_name=first.airline_name,
            )
        )
        return offers

    def search(self, query: SearchQuery) -> list[Offer]:
        offsets = range(-3, 4) if query.flexible_dates else [0]
        return [o for off in offsets for o in self._offers_for_day(query, off)]
