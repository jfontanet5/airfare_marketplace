"""Provider contract and error taxonomy."""

from __future__ import annotations

from abc import ABC, abstractmethod

from airfare.domain.models import Offer, SearchQuery


class ProviderError(Exception):
    """Base class for provider failures the UI can present to a user."""


class ProviderAuthError(ProviderError):
    """Missing or rejected credentials."""


class ProviderRateLimitedError(ProviderError):
    """The upstream API asked us to slow down."""


class ProviderUnavailableError(ProviderError):
    """Network or upstream outage after retries were exhausted."""


class FlightSearchProvider(ABC):
    """Returns *raw* offers: priced in the provider's currency, not yet normalized."""

    name: str = "base"

    @abstractmethod
    def search(self, query: SearchQuery) -> list[Offer]: ...
