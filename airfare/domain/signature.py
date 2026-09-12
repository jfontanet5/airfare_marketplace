"""Deterministic offer identity.

Two offers that describe the same physical flight plan (same segments, same
carriers, same flight numbers, same departure times) share a signature, no
matter which provider or search produced them. That makes it possible to
deduplicate results and to track one itinerary's price over time.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from datetime import datetime

from airfare.domain.models import Offer


def _segment_key(
    origin: str, destination: str, carrier: str | None, number: str | None, dep: datetime | None
) -> str:
    return "|".join(
        [origin, destination, carrier or "", number or "", dep.isoformat() if dep else ""]
    )


def offer_signature(offer: Offer) -> str:
    """Stable, provider-independent signature. Short SHA-1 of the segment chain."""
    itinerary_keys = [
        ">".join(
            _segment_key(s.origin, s.destination, s.carrier_code, s.flight_number, s.dep_at)
            for s in it.segments
        )
        for it in offer.itineraries
    ]
    raw = "||".join(itinerary_keys).strip("|")
    if not raw:
        # No segment detail (e.g. a coarse mock offer): fall back to summary fields.
        raw = "|".join(
            [
                offer.origin,
                offer.destination,
                offer.departure_date.isoformat(),
                offer.return_date.isoformat() if offer.return_date else "",
                offer.airline_code,
                str(offer.stops_out),
                str(offer.stops_return),
            ]
        )
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _rank_key(offer: Offer) -> tuple[float, int, datetime]:
    usd = offer.price.usd if offer.price.usd is not None else float("inf")
    dep = offer.outbound.dep_at if offer.outbound and offer.outbound.dep_at else datetime.max
    return (usd, offer.total_stops, dep.replace(tzinfo=None))


def dedup_offers(offers: Iterable[Offer]) -> list[Offer]:
    """Keep the best-priced offer for each signature, preserving first-seen order."""
    best: dict[str, Offer] = {}
    for o in offers:
        sig = o.signature or offer_signature(o)
        current = best.get(sig)
        if current is None or _rank_key(o) < _rank_key(current):
            best[sig] = o
    return list(best.values())
