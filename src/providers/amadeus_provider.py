# src/providers/amadeus_provider.py

from __future__ import annotations

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from providers.base import FlightSearchProvider
from core.models import SearchParams, Offer, Itinerary, Segment
from services.amadeus_client import AmadeusClient


def _max_stops_from_label(label: str) -> int:
    if "Nonstop" in label:
        return 0
    if "1 stop" in label:
        return 1
    return 2


def _count_stops(itinerary: dict) -> int:
    segments = itinerary.get("segments", []) or []
    return max(0, len(segments) - 1)


def _pick_airline_code(offer: dict) -> str:
    # Prefer validating airline codes if present
    vac = offer.get("validatingAirlineCodes")
    if isinstance(vac, list) and vac:
        return str(vac[0])

    itineraries = offer.get("itineraries", []) or []
    if itineraries and itineraries[0].get("segments"):
        return str(itineraries[0]["segments"][0].get("carrierCode", ""))

    return ""


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    """
    Parse Amadeus datetime strings like:
      - '2026-02-15T10:30:00'
      - '2026-02-15T10:30:00Z'
      - '2026-02-15T10:30:00+00:00'
    """
    if not value:
        return None
    try:
        v = str(value).replace("Z", "+00:00")
        return datetime.fromisoformat(v)
    except Exception:
        return None


def _parse_iso8601_duration_minutes(duration: Optional[str]) -> Optional[int]:
    """
    Parse durations like 'PT6H30M' into total minutes.
    """
    if not duration or not isinstance(duration, str):
        return None
    if not duration.startswith("PT"):
        return None

    hours = 0
    minutes = 0
    tmp = duration[2:]  # strip 'PT'

    # Very small parser; Amadeus typically uses H and M (sometimes S)
    num = ""
    for ch in tmp:
        if ch.isdigit():
            num += ch
            continue
        if ch == "H" and num:
            hours = int(num)
            num = ""
        elif ch == "M" and num:
            minutes = int(num)
            num = ""
        else:
            # ignore unknown tokens (e.g., seconds)
            num = ""

    return hours * 60 + minutes


def _build_itineraries(
    offer_raw: Dict[str, Any],
    carriers_dict: Dict[str, str],
) -> List[Itinerary]:
    """
    Convert Amadeus offer['itineraries'] into canonical Itinerary/Segment objects.
    """
    itineraries_raw = offer_raw.get("itineraries", []) or []
    out: List[Itinerary] = []

    for idx, it in enumerate(itineraries_raw):
        direction = "OUT" if idx == 0 else "RETURN"

        duration_minutes = _parse_iso8601_duration_minutes(it.get("duration"))

        segs: List[Segment] = []
        for seg in it.get("segments", []) or []:
            dep = seg.get("departure", {}) or {}
            arr = seg.get("arrival", {}) or {}

            carrier_code = seg.get("carrierCode")
            carrier_name = carriers_dict.get(
                str(carrier_code), None) if carrier_code else None

            operating = seg.get("operating", {}) or {}
            op_code = operating.get("carrierCode")
            op_name = carriers_dict.get(
                str(op_code), None) if op_code else None

            aircraft = seg.get("aircraft", {}) or {}
            aircraft_code = aircraft.get("code")

            segs.append(
                Segment(
                    origin=str(dep.get("iataCode", "")),
                    destination=str(arr.get("iataCode", "")),
                    dep_at=_parse_dt(dep.get("at")),
                    arr_at=_parse_dt(arr.get("at")),
                    carrier_code=str(carrier_code) if carrier_code else None,
                    carrier_name=carrier_name,
                    flight_number=str(seg.get("number")) if seg.get(
                        "number") is not None else None,
                    operating_carrier_code=str(op_code) if op_code else None,
                    operating_carrier_name=op_name,
                    aircraft_code=str(
                        aircraft_code) if aircraft_code else None,
                )
            )

        out.append(Itinerary(direction=direction, segments=segs,
                   duration_minutes=duration_minutes))

    return out


class AmadeusProvider(FlightSearchProvider):
    """
    Live provider (Amadeus Self-Service Flight Offers Search).
    """

    def __init__(self, client: Optional[AmadeusClient] = None, max_results: int = 25):
        self.client = client or AmadeusClient()
        self.max_results = max_results

    def _search_one(self, params: SearchParams) -> List[Offer]:
        query = {
            "originLocationCode": params.origin,
            "destinationLocationCode": params.destination,
            "departureDate": params.departure_date.isoformat(),
            "adults": max(1, int(params.passengers)),
            "max": self.max_results,
        }

        if params.trip_structure == "Roundtrip" and params.return_date:
            query["returnDate"] = params.return_date.isoformat()

        max_stops = _max_stops_from_label(params.max_stops)
        if max_stops == 0:
            query["nonStop"] = "true"

        payload = self.client.get("/v2/shopping/flight-offers", query)

        data = payload.get("data", []) or []

        dictionaries = payload.get("dictionaries", {}) or {}
        carriers_dict = dictionaries.get("carriers", {}) or {}

        offers: List[Offer] = []

        for o in data:
            price = o.get("price", {}) or {}
            total_str = price.get("grandTotal") or price.get("total") or "0"
            currency = price.get("currency", "USD")

            itineraries_raw = o.get("itineraries", []) or []
            stops_out = _count_stops(itineraries_raw[0]) if len(
                itineraries_raw) >= 1 else 0
            stops_ret = _count_stops(itineraries_raw[1]) if len(
                itineraries_raw) >= 2 else 0

            airline_code = _pick_airline_code(o)
            airline_name = carriers_dict.get(
                str(airline_code), None) if airline_code else None

            itineraries_obj = _build_itineraries(o, carriers_dict)

            offer = Offer(
                provider="amadeus",
                origin=params.origin,
                destination=params.destination,
                trip_structure=params.trip_structure,
                departure_date=params.departure_date,
                return_date=params.return_date,

                # legacy summary fields
                airline=airline_code,
                stops_out=stops_out,
                stops_return=stops_ret,
                total_price_usd=float(total_str),

                # Phase 3 richer fields
                itineraries=itineraries_obj,
                airline_name=airline_name,
                purchase_url=None,

                currency=str(currency),
                score=0.0,
                raw=o,
            )
            offers.append(offer)

        # Post-filter stops if max is 1 or 2+ (Amadeus only supports nonStop switch)
        max_allowed = _max_stops_from_label(params.max_stops)
        if max_allowed >= 1:
            offers = [
                of for of in offers
                if (of.stops_out <= max_allowed) and (of.stops_return <= max_allowed)
            ]

        return offers

    def search(self, params: SearchParams) -> List[Offer]:
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
                flexible_dates=False,
                use_real_data=params.use_real_data,
            )
            results.extend(self._search_one(p2))

        return results
