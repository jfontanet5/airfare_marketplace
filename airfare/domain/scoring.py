"""Recommendation scoring.

Score is a USD-equivalent cost: the fare plus explicit penalties for
inconvenience. Lower is better. Every penalty is reported as a human-readable
reason so the UI can explain *why* an option was recommended.
"""

from __future__ import annotations

from dataclasses import dataclass

from airfare.domain.models import Offer, ScoredOffer, SearchQuery


@dataclass(frozen=True, slots=True)
class ScoringWeights:
    stop_penalty_usd: float = 35.0
    date_offset_penalty_usd: float = 5.0
    duration_penalty_usd_per_hour: float = 4.0


DEFAULT_WEIGHTS = ScoringWeights()


def score_offer(
    offer: Offer, query: SearchQuery, weights: ScoringWeights = DEFAULT_WEIGHTS
) -> ScoredOffer:
    if offer.price.usd is None:
        return ScoredOffer(offer, float("inf"), ("Price could not be converted to USD",))

    score = offer.price.usd
    reasons: list[str] = [f"Fare ${offer.price.usd:,.0f}"]

    if offer.total_stops:
        penalty = offer.total_stops * weights.stop_penalty_usd
        score += penalty
        reasons.append(f"+${penalty:,.0f} for {offer.total_stops} stop(s)")
    else:
        reasons.append("Nonstop")

    offset_days = abs((offer.departure_date - query.departure_date).days)
    if offset_days:
        penalty = offset_days * weights.date_offset_penalty_usd
        score += penalty
        reasons.append(f"+${penalty:,.0f} for departing {offset_days} day(s) off requested date")

    if offer.total_duration_minutes:
        hours = offer.total_duration_minutes / 60
        penalty = hours * weights.duration_penalty_usd_per_hour
        score += penalty
        reasons.append(f"+${penalty:,.0f} for {hours:.1f}h total travel time")

    return ScoredOffer(offer, round(score, 2), tuple(reasons))


def score_offers(
    offers: list[Offer], query: SearchQuery, weights: ScoringWeights = DEFAULT_WEIGHTS
) -> list[ScoredOffer]:
    scored = [score_offer(o, query, weights) for o in offers]
    scored.sort(key=lambda s: s.score)
    return scored


def cheapest(offers: list[Offer]) -> Offer | None:
    priced = [o for o in offers if o.price.usd is not None]
    return min(priced, key=lambda o: o.price.usd or 0.0) if priced else None
