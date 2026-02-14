# src/providers/mock_provider.py

from typing import List
from providers.base import FlightSearchProvider
from core.models import SearchParams, Offer
from engine import generate_dummy_offers


class MockProvider(FlightSearchProvider):

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
                    airline=d.get("airline", ""),
                    stops_out=int(d.get("stops_out", 0) or 0),
                    stops_return=int(d.get("stops_return", 0) or 0),
                    total_price_usd=float(
                        d.get("total_price_usd", d.get(
                            "price_usd", 0.0)) or 0.0
                    ),
                    currency=d.get("currency", "USD"),
                    score=float(d.get("score", 0.0) or 0.0),
                    raw=d,
                )
            )

        return offers
