"""Build providers by name so callers never import concrete classes."""

from __future__ import annotations

from enum import StrEnum

from airfare.config import Settings
from airfare.providers.amadeus import AmadeusClient, AmadeusProvider
from airfare.providers.base import FlightSearchProvider, ProviderAuthError
from airfare.providers.mock import MockProvider
from airfare.providers.replay import ReplayProvider
from airfare.storage.repository import PriceHistoryRepository


class ProviderName(StrEnum):
    MOCK = "mock"
    REPLAY = "replay"
    AMADEUS = "amadeus"


def available_providers(settings: Settings) -> list[ProviderName]:
    names = [ProviderName.MOCK, ProviderName.REPLAY]
    if settings.amadeus_configured:
        names.append(ProviderName.AMADEUS)
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
        case ProviderName.AMADEUS:
            if not settings.amadeus_configured:
                raise ProviderAuthError(
                    "Set AMADEUS_CLIENT_ID and AMADEUS_CLIENT_SECRET to use live search"
                )
            client = AmadeusClient(
                settings.amadeus_client_id, settings.amadeus_client_secret, settings.amadeus_env
            )
            return AmadeusProvider(client, flexible_window_days=settings.flexible_window_days)
