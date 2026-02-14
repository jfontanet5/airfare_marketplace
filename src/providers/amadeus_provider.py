# src/providers/amadeus_provider.py

from typing import List, Optional
from datetime import timedelta
from providers.base import FlightSearchProvider
from core.models import SearchParams, Offer
from services.amadeus_client import AmadeusClient


def _max_stops_from_label(label: str) -> int:
    if "Nonstop" in label:
        return 0
    if "1 stop" in label:
        return 1
    return 2


def _count_stops(itinerary: dict) -> int:
    # Stops = number of segments - 1
    segments = itinerary.get("segments", []) or []
    return max(0, len(segments) - 1)


def _pick_airline(offer: dict) -> str:
    # Prefer validating airline codes if present
    vac = offer.get("validatingAirlineCodes")
    if isinstance(vac, list) and vac:
        return str(vac[0])
    # Fall back to first segment carrier
    itineraries = offer.get("itineraries", []) or []
    if itineraries and itineraries[0].get("segments"):
        return str(itineraries[0]["segments"][0].get("carrierCode", ""))
    return ""


class AmadeusProvider(FlightSearchProvider):
    """
    Live provider (Amadeus Self-Service Flight Offers Search).
    """

    def __init__(self, client: Optional[AmadeusClient] = None, max_results: int = 25):
        self.client = client or AmadeusClient()
        self.max_results = max_results

    def _search_one(self, params: SearchParams) -> List[Offer]:
        # Amadeus Flight Offers Search (GET)
        # /v2/shopping/flight-offers
        query = {
            "originLocationCode": params.origin,
            "destinationLocationCode": params.destination,
            "departureDate": params.departure_date.isoformat(),
            "adults": max(1, int(params.passengers)),
            "max": self.max_results,
        }

        if params.trip_structure == "Roundtrip" and params.return_date:
            query["returnDate"] = params.return_date.isoformat()

        # Nonstop filter (Amadeus supports nonStop=true/false)
        max_stops = _max_stops_from_label(params.max_stops)
        if max_stops == 0:
            query["nonStop"] = "true"

        # Airline filter (if not Any)
        if params.airlines and "Any" not in params.airlines:
            # NOTE: Amadeus expects carrier codes (e.g. "B6"), not names.
            # For now, we do post-filter by matching the airline code we parse, if present.
            # (Later improvement: map airline names -> IATA codes.)
            pass

        payload = self.client.get("/v2/shopping/flight-offers", query)
        data = payload.get("data", []) or []

        offers: List[Offer] = []
        for o in data:
            price = o.get("price", {}) or {}
            total_str = price.get("grandTotal") or price.get("total") or "0"
            currency = price.get("currency", "USD")

            itineraries = o.get("itineraries", []) or []
            stops_out = _count_stops(itineraries[0]) if len(
                itineraries) >= 1 else 0
            stops_ret = _count_stops(itineraries[1]) if len(
                itineraries) >= 2 else 0

            offer = Offer(
                provider="amadeus",
                origin=params.origin,
                destination=params.destination,
                trip_structure=params.trip_structure,
                departure_date=params.departure_date,
                return_date=params.return_date,
                airline=_pick_airline(o),
                stops_out=stops_out,
                stops_return=stops_ret,
                total_price_usd=float(total_str),
                currency=str(currency),
                score=0.0,
                raw=o,
            )
            offers.append(offer)

        # Post-filter stops if max is 1 or 2+ (because Amadeus only has nonStop switch)
        max_allowed = _max_stops_from_label(params.max_stops)
        if max_allowed >= 1:
            offers = [
                of for of in offers
                if (of.stops_out <= max_allowed) and (of.stops_return <= max_allowed)
            ]

        return offers

    def search(self, params: SearchParams) -> List[Offer]:
        # If flexible_dates enabled, do 7 calls (±3 days) with same trip length
        if not params.flexible_dates:
            return self._search_one(params)

        results: List[Offer] = []
        trip_len = None
        if params.trip_structure == "Roundtrip" and params.return_date:
            trip_len = (params.return_date - params.departure_date).days

        for delta in range(-3, 4):
            dep = params.departure_date + timedelta(days=delta)
            ret = None
            if trip_len is not None:
                ret = dep + timedelta(days=trip_len)

            p2 = SearchParams(
                origin=params.origin,
                destination=params.destination,
                trip_structure=params.trip_structure,
                departure_date=dep,
                return_date=ret,
                optimization_mode=params.optimization_mode,
                passengers=params.passengers,
                max_stops=params.max_stops,
                airlines=params.airlines,
                multicity=params.multicity,
                flexible_dates=False,   # prevent recursion
                use_real_data=params.use_real_data,
            )
            results.extend(self._search_one(p2))

        return results
