"""Offline provider that replays the most recent stored snapshot for a route.

Useful for demos without API keys and for reproducing a past search.
"""

from __future__ import annotations

from airfare.domain.models import Offer, SearchQuery
from airfare.providers.base import FlightSearchProvider
from airfare.storage.repository import PriceHistoryRepository


class ReplayProvider(FlightSearchProvider):
    name = "replay"

    def __init__(self, history: PriceHistoryRepository) -> None:
        self.history = history

    def search(self, query: SearchQuery) -> list[Offer]:
        return self.history.latest_offers(query)
