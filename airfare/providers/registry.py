"""Build providers by name so callers never import concrete classes."""

from __future__ import annotations

from enum import StrEnum

from airfare.config import Settings
from airfare.providers.base import FlightSearchProvider, ProviderAuthError
from airfare.providers.mock import MockProvider
from airfare.providers.replay import ReplayProvider
from airfare.providers.serpapi import SerpApiClient, SerpApiProvider
from airfare.storage.repository import PriceHistoryRepository


class ProviderName(StrEnum):
    MOCK = "mock"
    REPLAY = "replay"
    SERPAPI = "serpapi"


def available_providers(settings: Settings) -> list[ProviderName]:
    names = [ProviderName.MOCK, ProviderName.REPLAY]
    if settings.live_configured:
        names.append(ProviderName.SERPAPI)
    return names


def build_provider(
    name: ProviderName | str, settings: Settings, history: PriceHistoryRepository | None = None
) -> FlightSearchProvider:
    match ProviderName(name):
        case ProviderName.MOCK:
            return MockProvider()
        case ProviderName.REPLAY:
            if history is None:
                raise ValueError("replay provider requires a history repository")
            return ReplayProvider(history)
        case ProviderName.SERPAPI:
            if not settings.live_configured:
                raise ProviderAuthError("Set SERPAPI_API_KEY to use live search")
            return SerpApiProvider(
                SerpApiClient(settings.serpapi_api_key),
                currency=settings.serpapi_currency,
                flexible_window_days=settings.flexible_window_days,
                return_legs_top_n=settings.serpapi_return_legs_top_n,
            )
