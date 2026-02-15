# src/core/scoring.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Optional, Tuple

from core.models import Offer, SearchParams


@dataclass(frozen=True)
class ScoredOffer:
    offer: Offer
    score: float


def _requested_dep(params: SearchParams) -> date:
    return params.departure_date


def _date_offset_days(offer: Offer, params: SearchParams) -> int:
    """
    How far the offer's departure date is from the requested departure date.
    """
    try:
        return abs((offer.departure_date - _requested_dep(params)).days)
    except Exception:
        return 0


def _total_stops(offer: Offer, params: SearchParams) -> int:
    out_stops = int(offer.stops_out or 0)
    ret_stops = int(offer.stops_return or 0)
    if params.trip_structure == "One-way":
        return out_stops
    return out_stops + ret_stops


def score_offer(offer: Offer, params: SearchParams) -> float:
    """
    Heuristic score. Lower is better.

    This is the object-based replacement for engine.py's dict scoring.
    We intentionally remove provider/region bias as part of Phase 3 cleanup.
    """
    price = float(offer.total_price_usd or 1e9)

    stops_penalty = _total_stops(offer, params) * 35.0
    date_penalty = _date_offset_days(offer, params) * 5.0

    return price + stops_penalty + date_penalty


def score_offers(offers: List[Offer], params: SearchParams) -> List[ScoredOffer]:
    scored = [ScoredOffer(o, score_offer(o, params)) for o in offers]
    scored.sort(key=lambda x: x.score)
    return scored


def pick_recommended(scored: List[ScoredOffer]) -> Optional[ScoredOffer]:
    if not scored:
        return None
    return scored[0]


def pick_best_by_price(offers: List[Offer]) -> Optional[Offer]:
    if not offers:
        return None
    return min(offers, key=lambda o: float(o.total_price_usd or 1e9))


def format_offer_label(offer: Offer) -> str:
    """
    Human-readable label for UI.
    Prefer airline name if available.
    """
    airline = offer.airline_name or offer.airline or "Unknown airline"
    return f"{offer.trip_structure} · {airline} · {offer.origin} → {offer.destination}"
