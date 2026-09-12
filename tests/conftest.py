from __future__ import annotations

from datetime import UTC, date, datetime
from itertools import pairwise
from pathlib import Path

import pytest
from airfare.domain.models import Itinerary, Offer, Price, SearchQuery, Segment
from airfare.services.fx import FxService, FxSource

FIXTURES = Path(__file__).parent / "fixtures"


class StaticFx:
    name = "static"

    def __init__(self, rates: dict[str, float]) -> None:
        self.rates = rates
        self.calls = 0

    def rate_to_usd(self, currency: str, day: date) -> float:
        self.calls += 1
        return self.rates[currency]


@pytest.fixture
def fx(tmp_path: Path) -> FxService:
    src: FxSource = StaticFx({"EUR": 1.10, "GBP": 1.30})
    return FxService(tmp_path / "fx.sqlite", [src])


@pytest.fixture
def query() -> SearchQuery:
    return SearchQuery(
        "SJU", "JFK", date(2026, 10, 1), date(2026, 10, 5), passengers=1, max_stops=2
    )


def make_offer(
    price: float = 300.0,
    currency: str = "USD",
    stops: int = 0,
    carrier: str = "B6",
    dep_date: date = date(2026, 10, 1),
    dep_hour: int = 8,
    duration: int | None = 240,
    provider: str = "test",
) -> Offer:
    legs = ["SJU", *(["MIA"] * stops), "JFK"]
    segs = []
    t = datetime(dep_date.year, dep_date.month, dep_date.day, dep_hour, tzinfo=UTC)
    for i, (a, b) in enumerate(pairwise(legs)):
        segs.append(
            Segment(a, b, t, t.replace(hour=t.hour + 2), carrier, None, str(100 + i), None, None)
        )
        t = t.replace(hour=t.hour + 3)
    return Offer(
        provider=provider,
        origin="SJU",
        destination="JFK",
        departure_date=dep_date,
        return_date=None,
        price=Price(price, currency),
        itineraries=(Itinerary(tuple(segs), duration),),
        airline_code=carrier,
        airline_name=None,
    )
