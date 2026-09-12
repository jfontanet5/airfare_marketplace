"""Live provider backed by SerpApi's Google Flights engine.

Each outbound search costs one SerpApi request; each return-leg lookup costs
one more. ``return_legs_top_n`` bounds the extra spend: the cheapest N outbound
options get their matching return itinerary attached, the rest carry the total
round-trip price with the return leg left to be chosen at booking.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

from airfare.domain.models import Offer, SearchQuery
from airfare.providers.base import FlightSearchProvider
from airfare.providers.serpapi.client import SerpApiClient
from airfare.providers.serpapi.parser import parse_response, parse_return_options

log = logging.getLogger(__name__)

# SerpApi `stops` filter: 0 any, 1 nonstop, 2 one stop or fewer, 3 two stops or fewer
_STOPS_PARAM = {0: 1, 1: 2, 2: 3}


class SerpApiProvider(FlightSearchProvider):
    name = "serpapi"

    def __init__(
        self,
        client: SerpApiClient,
        currency: str = "USD",
        flexible_window_days: int = 3,
        return_legs_top_n: int = 0,
    ) -> None:
        self.client = client
        self.currency = currency
        self.window = flexible_window_days
        self.return_legs_top_n = return_legs_top_n

    def _base_params(self, q: SearchQuery) -> dict[str, Any]:
        params: dict[str, Any] = {
            "engine": "google_flights",
            "departure_id": q.origin,
            "arrival_id": q.destination,
            "outbound_date": q.departure_date.isoformat(),
            "type": 1 if q.return_date else 2,
            "adults": q.passengers,
            "currency": self.currency,
            "hl": "en",
            "gl": "us",
            "stops": _STOPS_PARAM.get(q.max_stops, 0),
        }
        if q.return_date:
            params["return_date"] = q.return_date.isoformat()
        if q.airlines:
            params["include_airlines"] = ",".join(sorted(q.airlines))
        return params

    def _attach_return(self, q: SearchQuery, offer: Offer, token: str) -> Offer:
        payload = self.client.search({**self._base_params(q), "departure_token": token})
        options = parse_return_options(payload)
        if not options:
            return offer
        inbound, total = min(options, key=lambda x: x[1])
        return replace(
            offer,
            itineraries=(*offer.itineraries, inbound),
            price=replace(offer.price, amount=total),
        )

    def _search_one(self, q: SearchQuery) -> list[Offer]:
        pairs = parse_response(
            self.client.search(self._base_params(q)),
            q.origin,
            q.destination,
            q.return_date,
            self.currency,
        )
        log.info(
            "serpapi %s-%s %s: %d itineraries",
            q.origin,
            q.destination,
            q.departure_date,
            len(pairs),
        )
        if not q.return_date or self.return_legs_top_n <= 0:
            return [o for o, _ in pairs]

        pairs.sort(key=lambda p: p[0].price.amount)
        head = [(o, t) for o, t in pairs[: self.return_legs_top_n] if t]
        tail = [o for o, _ in pairs[len(head) :]] if head else [o for o, _ in pairs]
        with ThreadPoolExecutor(max_workers=3) as pool:
            enriched = list(pool.map(lambda p: self._attach_return(q, p[0], p[1]), head))
        return [*enriched, *tail]

    def search(self, query: SearchQuery) -> list[Offer]:
        if not query.flexible_dates:
            return self._search_one(query)
        shifted = [query.shifted(d) for d in range(-self.window, self.window + 1)]
        with ThreadPoolExecutor(max_workers=3) as pool:
            batches = list(pool.map(self._search_one, shifted))
        return [o for batch in batches for o in batch]
