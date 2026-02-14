# src/providers/csv_provider.py

from typing import List
from providers.base import FlightSearchProvider
from core.models import SearchParams, Offer
from data_access import load_flight_offers


class CSVProvider(FlightSearchProvider):
    def search(self, params: SearchParams) -> List[Offer]:
        df = load_flight_offers(params.__dict__)
        if df is None or df.empty:
            return []

        rows = df.to_dict(orient="records")
        offers: List[Offer] = []

        for d in rows:
            offers.append(
                Offer(
                    provider=d.get("provider", "csv"),
                    origin=d.get("origin", params.origin),
                    destination=d.get("destination", params.destination),
                    trip_structure=d.get(
                        "trip_structure", params.trip_structure),
                    departure_date=d.get(
                        "departure_date", params.departure_date),
                    return_date=d.get("return_date", params.return_date),
                    airline=d.get("airline", ""),
                    stops_out=int(d.get("stops_out", 0) or 0),
                    stops_return=int(d.get("stops_return", 0) or 0),
                    total_price_usd=float(
                        d.get("total_price_usd", 0.0) or 0.0),
                    currency=d.get("currency", "USD"),
                    score=float(d.get("score", 0.0) or 0.0),
                    # IMPORTANT: keep raw so legacy functions have region, etc.
                    raw=d,
                )
            )

        return offers
