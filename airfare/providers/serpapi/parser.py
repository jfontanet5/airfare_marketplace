"""Pure functions that turn SerpApi Google Flights JSON into domain objects.

Google Flights prices a round trip as a whole: the outbound search returns
outbound itineraries each carrying the total round-trip price for its cheapest
return pairing, and a ``departure_token`` that fetches the matching return
options. The parser therefore produces an Offer per outbound itinerary; the
provider may attach the return leg for the top options.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from airfare.domain.models import Itinerary, Offer, Price, Segment

_FLIGHT_NUMBER_RE = re.compile(r"^\s*([A-Z0-9]{2})\s*(\d+)\s*$")


def parse_time(value: str | None) -> datetime | None:
    """'2026-10-01 07:00' -> naive local datetime (Google reports airport-local times)."""
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M")
    except ValueError:
        return None


def split_flight_number(value: str | None) -> tuple[str | None, str | None]:
    """'B6 704' -> ('B6', '704')."""
    if not value:
        return None, None
    m = _FLIGHT_NUMBER_RE.match(value)
    return (m.group(1), m.group(2)) if m else (None, value.strip())


def _segment(raw: dict[str, Any]) -> Segment:
    dep, arr = raw.get("departure_airport") or {}, raw.get("arrival_airport") or {}
    code, number = split_flight_number(raw.get("flight_number"))
    return Segment(
        origin=str(dep.get("id", "")),
        destination=str(arr.get("id", "")),
        dep_at=parse_time(dep.get("time")),
        arr_at=parse_time(arr.get("time")),
        carrier_code=code,
        carrier_name=raw.get("airline"),
        flight_number=number,
        aircraft_code=raw.get("airplane"),
    )


def parse_itinerary(raw: dict[str, Any]) -> Itinerary:
    segments = tuple(_segment(s) for s in raw.get("flights") or [])
    total = raw.get("total_duration")
    return Itinerary(segments=segments, duration_minutes=int(total) if total else None)


def parse_offer(
    raw: dict[str, Any], origin: str, destination: str, return_date: date | None, currency: str
) -> Offer | None:
    outbound = parse_itinerary(raw)
    if not outbound.segments or raw.get("price") is None:
        return None
    first = outbound.segments[0]
    dep_date = first.dep_at.date() if first.dep_at else date.min
    return Offer(
        provider="serpapi",
        origin=origin,
        destination=destination,
        departure_date=dep_date,
        return_date=return_date,
        price=Price(amount=float(raw["price"]), currency=currency),
        itineraries=(outbound,),
        airline_code=first.carrier_code or "",
        airline_name=first.carrier_name,
        purchase_url=None,
    )


def parse_response(
    payload: dict[str, Any], origin: str, destination: str, return_date: date | None, currency: str
) -> list[tuple[Offer, str | None]]:
    """Return (offer, departure_token) pairs; token is needed to fetch the return leg."""
    out: list[tuple[Offer, str | None]] = []
    for group in ("best_flights", "other_flights"):
        for raw in payload.get(group) or []:
            offer = parse_offer(raw, origin, destination, return_date, currency)
            if offer is not None:
                out.append((offer, raw.get("departure_token")))
    return out


def parse_return_options(payload: dict[str, Any]) -> list[tuple[Itinerary, float]]:
    """Return-leg search results as (itinerary, total_round_trip_price)."""
    out: list[tuple[Itinerary, float]] = []
    for group in ("best_flights", "other_flights"):
        for raw in payload.get(group) or []:
            it = parse_itinerary(raw)
            if it.segments and raw.get("price") is not None:
                out.append((it, float(raw["price"])))
    return out


def parse_price_insights(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Google's own read on the fare level: lowest_price, price_level, typical_price_range."""
    insights = payload.get("price_insights")
    return dict(insights) if isinstance(insights, dict) else None
