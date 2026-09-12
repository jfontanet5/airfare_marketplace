"""Pure functions that turn Amadeus Flight Offers Search JSON into domain objects."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from airfare.domain.models import Itinerary, Offer, Price, Segment

_DURATION_RE = re.compile(r"^P(?:(?P<d>\d+)D)?(?:T(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?)?$")


def parse_iso_duration_minutes(value: str | None) -> int | None:
    """'PT6H30M' -> 390; 'P1DT2H' -> 1560; anything unparseable -> None."""
    if not value:
        return None
    m = _DURATION_RE.match(value)
    if not m:
        return None
    d, h, mi = (int(m.group(k) or 0) for k in ("d", "h", "m"))
    return d * 1440 + h * 60 + mi


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _segment(raw: dict[str, Any], carriers: dict[str, str]) -> Segment:
    dep, arr = raw.get("departure") or {}, raw.get("arrival") or {}
    carrier = raw.get("carrierCode")
    op = (raw.get("operating") or {}).get("carrierCode")
    return Segment(
        origin=str(dep.get("iataCode", "")),
        destination=str(arr.get("iataCode", "")),
        dep_at=parse_datetime(dep.get("at")),
        arr_at=parse_datetime(arr.get("at")),
        carrier_code=str(carrier) if carrier else None,
        carrier_name=carriers.get(str(carrier)) if carrier else None,
        flight_number=str(raw["number"]) if raw.get("number") is not None else None,
        operating_carrier_code=str(op) if op else None,
        aircraft_code=(raw.get("aircraft") or {}).get("code"),
    )


def _itinerary(raw: dict[str, Any], carriers: dict[str, str]) -> Itinerary:
    return Itinerary(
        segments=tuple(_segment(s, carriers) for s in raw.get("segments") or []),
        duration_minutes=parse_iso_duration_minutes(raw.get("duration")),
    )


def _airline_code(raw: dict[str, Any], itineraries: tuple[Itinerary, ...]) -> str:
    validating = raw.get("validatingAirlineCodes") or []
    if validating:
        return str(validating[0])
    if itineraries and itineraries[0].segments:
        return itineraries[0].segments[0].carrier_code or ""
    return ""


def parse_offer(
    raw: dict[str, Any], carriers: dict[str, str], origin: str, destination: str
) -> Offer:
    itineraries = tuple(_itinerary(it, carriers) for it in raw.get("itineraries") or [])
    price_raw = raw.get("price") or {}
    amount = float(price_raw.get("grandTotal") or price_raw.get("total") or 0.0)
    code = _airline_code(raw, itineraries)

    dep_date: date = (
        itineraries[0].dep_at.date() if itineraries and itineraries[0].dep_at else date.min
    )
    ret_date: date | None = (
        itineraries[1].dep_at.date() if len(itineraries) > 1 and itineraries[1].dep_at else None
    )
    return Offer(
        provider="amadeus",
        origin=origin,
        destination=destination,
        departure_date=dep_date,
        return_date=ret_date,
        price=Price(amount=amount, currency=str(price_raw.get("currency", "USD"))),
        itineraries=itineraries,
        airline_code=code,
        airline_name=carriers.get(code),
    )


def parse_response(payload: dict[str, Any], origin: str, destination: str) -> list[Offer]:
    carriers: dict[str, str] = (payload.get("dictionaries") or {}).get("carriers") or {}
    return [parse_offer(o, carriers, origin, destination) for o in payload.get("data") or []]
