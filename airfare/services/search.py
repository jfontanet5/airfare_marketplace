"""Application service: run a search end to end.

provider.search → normalize → score → persist observations → SearchResult.
The UI (or a future API) calls only this.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from airfare.domain.models import Offer, ScoredOffer, SearchQuery
from airfare.domain.scoring import DEFAULT_WEIGHTS, ScoringWeights, cheapest, score_offers
from airfare.providers.base import FlightSearchProvider
from airfare.services.fx import FxService
from airfare.services.normalization import normalize_offers
from airfare.storage.repository import Observation, PriceHistoryRepository

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SearchResult:
    query: SearchQuery
    provider: str
    searched_at: datetime
    offers: list[Offer]
    ranked: list[ScoredOffer]
    warnings: list[str] = field(default_factory=list)
    observations_persisted: int = 0

    @property
    def recommended(self) -> ScoredOffer | None:
        return self.ranked[0] if self.ranked and self.ranked[0].score != float("inf") else None

    @property
    def cheapest(self) -> Offer | None:
        return cheapest(self.offers)


class SearchService:
    def __init__(
        self,
        provider: FlightSearchProvider,
        fx: FxService | None,
        history: PriceHistoryRepository | None,
        weights: ScoringWeights = DEFAULT_WEIGHTS,
        history_top_n: int = 30,
    ) -> None:
        self.provider = provider
        self.fx = fx
        self.history = history
        self.weights = weights
        self.history_top_n = history_top_n

    def search(self, query: SearchQuery) -> SearchResult:
        searched_at = datetime.now(UTC)
        raw = self.provider.search(query)
        offers = normalize_offers(raw, query, self.fx, searched_at)
        ranked = score_offers(offers, query, self.weights)
        warnings: list[str] = []
        if any(o.price.usd is None for o in offers):
            warnings.append(
                "Some fares could not be converted to USD; they are shown in native currency."
            )

        persisted = 0
        if self.history is not None and offers and self.provider.name != "replay":
            try:
                top = sorted(
                    (o for o in offers if o.price.usd is not None), key=lambda o: o.price.usd or 0.0
                )[: self.history_top_n]
                persisted = self.history.record(
                    [Observation.from_offer(o, query, searched_at) for o in top]
                )
            except Exception as e:
                log.exception("failed to persist observations")
                warnings.append(f"Price history was not saved: {e}")

        return SearchResult(
            query, self.provider.name, searched_at, offers, ranked, warnings, persisted
        )
