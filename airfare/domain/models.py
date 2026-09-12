"""Canonical domain model.

Every provider is translated into these types at the boundary
(see :mod:`airfare.services.normalization`). Nothing downstream — scoring,
storage, ML, UI — ever sees a provider payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import StrEnum


class TripType(StrEnum):
    ROUNDTRIP = "roundtrip"
    ONE_WAY = "one_way"


@dataclass(frozen=True, slots=True)
class SearchQuery:
    """What the user asked for. Pure search intent — no UI routing flags."""

    origin: str
    destination: str
    departure_date: date
    return_date: date | None = None
    passengers: int = 1
    max_stops: int = 1
    flexible_dates: bool = False
    airlines: frozenset[str] = frozenset()  # IATA carrier codes; empty = any

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", self.origin.strip().upper())
        object.__setattr__(self, "destination", self.destination.strip().upper())
        if len(self.origin) != 3 or len(self.destination) != 3:
            raise ValueError("origin and destination must be 3-letter IATA codes")
        if self.origin == self.destination:
            raise ValueError("origin and destination must differ")
        if self.passengers < 1:
            raise ValueError("passengers must be >= 1")
        if self.max_stops < 0:
            raise ValueError("max_stops must be >= 0")
        if self.return_date is not None and self.return_date < self.departure_date:
            raise ValueError("return_date must be on or after departure_date")

    @property
    def trip_type(self) -> TripType:
        return TripType.ROUNDTRIP if self.return_date else TripType.ONE_WAY

    @property
    def trip_length(self) -> timedelta | None:
        return (self.return_date - self.departure_date) if self.return_date else None

    def shifted(self, days: int) -> SearchQuery:
        """Same query with both dates moved by ``days`` (used for flexible-date fan-out)."""
        dep = self.departure_date + timedelta(days=days)
        ret = dep + self.trip_length if self.trip_length is not None else None
        return SearchQuery(
            origin=self.origin,
            destination=self.destination,
            departure_date=dep,
            return_date=ret,
            passengers=self.passengers,
            max_stops=self.max_stops,
            flexible_dates=False,
            airlines=self.airlines,
        )


@dataclass(frozen=True, slots=True)
class Segment:
    """A single flight leg."""

    origin: str
    destination: str
    dep_at: datetime | None = None
    arr_at: datetime | None = None
    carrier_code: str | None = None
    carrier_name: str | None = None
    flight_number: str | None = None
    operating_carrier_code: str | None = None
    aircraft_code: str | None = None

    @property
    def duration(self) -> timedelta | None:
        if self.dep_at and self.arr_at:
            return self.arr_at - self.dep_at
        return None


@dataclass(frozen=True, slots=True)
class Itinerary:
    """One direction of travel."""

    segments: tuple[Segment, ...]
    duration_minutes: int | None = None

    @property
    def stops(self) -> int:
        return max(0, len(self.segments) - 1)

    @property
    def origin(self) -> str | None:
        return self.segments[0].origin if self.segments else None

    @property
    def destination(self) -> str | None:
        return self.segments[-1].destination if self.segments else None

    @property
    def dep_at(self) -> datetime | None:
        return self.segments[0].dep_at if self.segments else None

    @property
    def arr_at(self) -> datetime | None:
        return self.segments[-1].arr_at if self.segments else None

    def layovers(self) -> list[timedelta]:
        out: list[timedelta] = []
        for prev, nxt in zip(self.segments, self.segments[1:], strict=False):
            if prev.arr_at and nxt.dep_at:
                out.append(nxt.dep_at - prev.arr_at)
        return out


@dataclass(frozen=True, slots=True)
class Price:
    """Money as quoted by the provider plus its USD normalization.

    ``usd`` is ``None`` only when FX was unavailable; consumers must handle it.
    """

    amount: float
    currency: str
    usd: float | None = None
    fx_rate: float | None = None  # USD per 1 unit of ``currency``
    fx_as_of: date | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "currency", self.currency.upper())
        if self.amount < 0:
            raise ValueError("amount must be >= 0")

    @property
    def is_normalized(self) -> bool:
        return self.usd is not None


@dataclass(frozen=True, slots=True)
class Offer:
    """A bookable option: one or two itineraries at a price."""

    provider: str
    origin: str
    destination: str
    departure_date: date
    return_date: date | None
    price: Price
    itineraries: tuple[Itinerary, ...]
    airline_code: str
    airline_name: str | None = None
    signature: str = ""  # filled by normalization; stable across searches
    purchase_url: str | None = None

    @property
    def trip_type(self) -> TripType:
        return TripType.ROUNDTRIP if self.return_date else TripType.ONE_WAY

    @property
    def outbound(self) -> Itinerary | None:
        return self.itineraries[0] if self.itineraries else None

    @property
    def inbound(self) -> Itinerary | None:
        return self.itineraries[1] if len(self.itineraries) > 1 else None

    @property
    def stops_out(self) -> int:
        return self.outbound.stops if self.outbound else 0

    @property
    def stops_return(self) -> int:
        return self.inbound.stops if self.inbound else 0

    @property
    def total_stops(self) -> int:
        return self.stops_out + self.stops_return

    @property
    def total_duration_minutes(self) -> int | None:
        mins = [it.duration_minutes for it in self.itineraries]
        if not mins or any(m is None for m in mins):
            return None
        return sum(m for m in mins if m is not None)

    @property
    def airline_display(self) -> str:
        return self.airline_name or self.airline_code or "Unknown airline"


@dataclass(frozen=True, slots=True)
class ScoredOffer:
    offer: Offer
    score: float
    reasons: tuple[str, ...] = field(default_factory=tuple)
