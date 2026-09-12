"""Live provider backed by the Amadeus Flight Offers Search API."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from airfare.domain.models import Offer, SearchQuery
from airfare.providers.amadeus.client import AmadeusClient
from airfare.providers.amadeus.parser import parse_response
from airfare.providers.base import FlightSearchProvider

log = logging.getLogger(__name__)


class AmadeusProvider(FlightSearchProvider):
    name = "amadeus"

    def __init__(
        self, client: AmadeusClient, max_results: int = 50, flexible_window_days: int = 3
    ) -> None:
        self.client = client
        self.max_results = max_results
        self.window = flexible_window_days

    def _build_params(self, q: SearchQuery) -> dict[str, Any]:
        params: dict[str, Any] = {
            "originLocationCode": q.origin,
            "destinationLocationCode": q.destination,
            "departureDate": q.departure_date.isoformat(),
            "adults": q.passengers,
            "max": self.max_results,
        }
        if q.return_date:
            params["returnDate"] = q.return_date.isoformat()
        if q.max_stops == 0:
            params["nonStop"] = "true"
        if q.airlines:
            params["includedAirlineCodes"] = ",".join(sorted(q.airlines))
        return params

    def _search_one(self, q: SearchQuery) -> list[Offer]:
        payload = self.client.get("/v2/shopping/flight-offers", self._build_params(q))
        offers = parse_response(payload, q.origin, q.destination)
        log.info(
            "amadeus %s-%s %s: %d offers", q.origin, q.destination, q.departure_date, len(offers)
        )
        return offers

    def search(self, query: SearchQuery) -> list[Offer]:
        if not query.flexible_dates:
            return self._search_one(query)
        shifted = [query.shifted(d) for d in range(-self.window, self.window + 1)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            batches = list(pool.map(self._search_one, shifted))
        return [o for batch in batches for o in batch]
