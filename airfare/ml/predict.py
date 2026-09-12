"""Inference helpers used by the UI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from airfare.domain.models import Offer
from airfare.ml.features import feature_frame, feature_row
from airfare.ml.registry import ModelCard, load_model


@dataclass(frozen=True, slots=True)
class PriceDropPredictor:
    model: Any
    card: ModelCard

    @classmethod
    def load(cls, model_dir: Path) -> PriceDropPredictor | None:
        loaded = load_model(model_dir)
        return cls(*loaded) if loaded else None

    def probability(self, offer: Offer, search_date: date | None = None) -> float:
        frame = feature_frame([feature_row(offer, search_date or date.today())])
        return float(self.model.predict_proba(frame)[0, 1])


def price_position(price_usd: float, stats: dict[str, float]) -> str | None:
    """Plain-language position of a fare within the route's observed distribution."""
    if not stats or stats.get("count", 0) < 5:
        return None
    if price_usd <= stats["p25"]:
        return "low"
    if price_usd <= stats["p75"]:
        return "typical"
    return "high"
