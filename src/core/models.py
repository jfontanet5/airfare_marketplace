# src/core/models.py

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional, List, Dict, Any


@dataclass(frozen=True)
class SearchParams:
    origin: str
    destination: str
    trip_structure: str  # "Roundtrip" | "One-way"
    departure_date: date
    return_date: Optional[date]
    optimization_mode: str
    passengers: int
    max_stops: str
    airlines: List[str]
    multicity: bool
    flexible_dates: bool
    use_real_data: bool


@dataclass(frozen=True)
class Segment:
    """A single flight leg."""

    origin: str
    destination: str
    dep_at: Optional[datetime] = None
    arr_at: Optional[datetime] = None
    carrier_code: Optional[str] = None  # e.g. "AA"
    carrier_name: Optional[str] = None  # e.g. "American Airlines"
    flight_number: Optional[str] = None  # e.g. "1234"
    operating_carrier_code: Optional[str] = None
    operating_carrier_name: Optional[str] = None
    aircraft_code: Optional[str] = None


@dataclass(frozen=True)
class Itinerary:
    """A collection of segments representing one direction of travel."""

    direction: str  # "OUT" | "RETURN"
    segments: List[Segment] = field(default_factory=list)
    duration_minutes: Optional[int] = None


@dataclass
class Offer:
    """Canonical offer representation used throughout the system.

    Phase 3 note:
      - Keep legacy fields (stops_out, stops_return, airline, score, raw) to avoid
        breaking the current UI/scoring while we migrate to object-based scoring.
      - New fields (itineraries, airline_name, purchase_url) enable richer UI, dedup,
        storage, and ML training on real observations.
    """

    provider: str
    origin: str
    destination: str
    trip_structure: str
    departure_date: date
    return_date: Optional[date]

    # Legacy summary fields (kept for backward compatibility)
    airline: str
    stops_out: int
    stops_return: int
    total_price_usd: float

    # New richer fields (Phase 3)
    itineraries: List[Itinerary] = field(default_factory=list)
    airline_name: Optional[str] = None
    purchase_url: Optional[str] = None

    # Metadata
    currency: str = "USD"
    score: float = 0.0
    raw: Optional[Dict[str, Any]] = None
    # stable identity for dedup/history (computed later)
    offer_signature: Optional[str] = None


@dataclass(frozen=True)
class ScoredOffer:
    """Non-mutating scored wrapper so we stop writing score into offers/dicts."""

    offer: Offer
    score: float
    reasons: List[str] = field(default_factory=list)
