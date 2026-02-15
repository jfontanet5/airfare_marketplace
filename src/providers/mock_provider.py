# src/providers/mock_provider.py

from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, List

from core.models import Offer, SearchParams
from providers.base import FlightSearchProvider


def generate_dummy_offers(search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Simulate multi-region, multi-provider offers.

    Respects:
      - trip_structure (Roundtrip vs One-way)
      - optimization_mode (Optimal vs Traditional)
      - max_stops (Nonstop / up to 1 / up to 2+)
      - flexible_dates (±3 days around selected date)

    Returns legacy-style dicts (used as raw payload for MockProvider).
    """
    origin = search_params["origin"]
    destination = search_params["destination"]
    trip_structure = search_params["trip_structure"]
    optimization_mode = search_params["optimization_mode"]
    max_stops_label = search_params["max_stops"]
    flexible_dates = search_params["flexible_dates"]
    departure_date = search_params["departure_date"]  # datetime.date
    return_date = search_params.get("return_date")    # datetime.date or None

    # Map label -> numeric limit
    if "Nonstop" in max_stops_label:
        max_allowed_stops = 0
    elif "1 stop" in max_stops_label:
        max_allowed_stops = 1
    else:
        max_allowed_stops = 2

    base_price = 300

    # Determine which day offsets we consider
    day_offsets = range(-3, 4) if flexible_dates else [0]

    offers: List[Dict[str, Any]] = []

    for offset in day_offsets:
        dep_variant = departure_date + timedelta(days=offset)
        ret_variant = (
            return_date + timedelta(days=offset)
            if (return_date and trip_structure == "Roundtrip")
            else None
        )

        price_adjustment = offset * 5
        day_base_price = base_price + price_adjustment

        daily_offers = [
            {
                "option_id": f"US-A-{offset:+d}",
                "region": "US",
                "provider": "mock",
                "description": f"{trip_structure} · Single carrier · US region",
                "total_price_usd": day_base_price,
                "stops_out": 0,
                "stops_return": 0 if trip_structure == "Roundtrip" else None,
                "trip_structure": trip_structure,
                "departure_date": dep_variant.isoformat(),
                "return_date": ret_variant.isoformat() if ret_variant else None,
                "date_offset_days": offset,
                "airline": "AA",
                "currency": "USD",
            },
            {
                "option_id": f"US-B-{offset:+d}",
                "region": "US",
                "provider": "mock",
                "description": f"{trip_structure} · 1 stop · US region",
                "total_price_usd": day_base_price - 15,
                "stops_out": 1,
                "stops_return": 0 if trip_structure == "Roundtrip" else None,
                "trip_structure": trip_structure,
                "departure_date": dep_variant.isoformat(),
                "return_date": ret_variant.isoformat() if ret_variant else None,
                "date_offset_days": offset,
                "airline": "DL",
                "currency": "USD",
            },
            {
                "option_id": f"EU-A-{offset:+d}",
                "region": "EU",
                "provider": "mock",
                "description": f"{trip_structure} · Mixed carriers · EU region",
                "total_price_usd": day_base_price - 35,
                "stops_out": 1,
                "stops_return": 1 if trip_structure == "Roundtrip" else None,
                "trip_structure": trip_structure,
                "departure_date": dep_variant.isoformat(),
                "return_date": ret_variant.isoformat() if ret_variant else None,
                "date_offset_days": offset,
                "airline": "UA",
                "currency": "USD",
            },
            {
                "option_id": f"LATAM-A-{offset:+d}",
                "region": "LATAM",
                "provider": "mock",
                "description": f"{trip_structure} · Aggressive pricing · LATAM region",
                "total_price_usd": day_base_price - 45,
                "stops_out": 2,
                "stops_return": 2 if trip_structure == "Roundtrip" else None,
                "trip_structure": trip_structure,
                "departure_date": dep_variant.isoformat(),
                "return_date": ret_variant.isoformat() if ret_variant else None,
                "date_offset_days": offset,
                "airline": "B6",
                "currency": "USD",
            },
        ]

        offers.extend(daily_offers)

    # --- Apply optimization_mode behavior ---
    if optimization_mode == "Traditional" and trip_structure == "Roundtrip":
        offers = [o for o in offers if o["trip_structure"] == "Roundtrip"]

    # --- Apply max_stops filter ---
    def passes_stops_filter(offer: Dict[str, Any]) -> bool:
        out_stops = offer.get("stops_out") or 0
        ret_stops = offer.get("stops_return")
        if trip_structure == "One-way":
            return out_stops <= max_allowed_stops
        ret_stops_val = ret_stops or 0
        return out_stops <= max_allowed_stops and ret_stops_val <= max_allowed_stops

    offers = [o for o in offers if passes_stops_filter(o)]

    return offers


class MockProvider(FlightSearchProvider):
    """
    Deterministic offline provider for dev/testing.
    Generates legacy dict offers and wraps them into Offer objects.
    """

    def search(self, params: SearchParams) -> List[Offer]:
        offers_dicts = generate_dummy_offers(params.__dict__)

        offers: List[Offer] = []
        for d in offers_dicts:
            offers.append(
                Offer(
                    provider=d.get("provider", "mock"),
                    origin=params.origin,
                    destination=params.destination,
                    trip_structure=params.trip_structure,
                    departure_date=params.departure_date,
                    return_date=params.return_date,
                    airline=d.get("airline", "") or "",
                    stops_out=int(d.get("stops_out", 0) or 0),
                    stops_return=int(d.get("stops_return", 0) or 0),
                    total_price_usd=float(
                        d.get("total_price_usd", 0.0) or 0.0),
                    currency="USD",
                    score=float(d.get("score", 0.0) or 0.0),
                    raw=d,
                )
            )

        return offers
