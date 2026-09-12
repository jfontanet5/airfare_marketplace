"""The single boundary between provider output and the rest of the system.

Given raw offers from any provider, normalization:

1. converts every price to USD with the FX rate for the search day (or marks
   it un-normalized if FX is unavailable — never silently mislabels currency);
2. assigns the deterministic itinerary signature;
3. applies query constraints the provider could not (max stops, airline filter);
4. deduplicates by signature, keeping the cheapest.

Everything downstream can therefore assume ``offer.signature`` is set and
``offer.price.usd`` is comparable across offers.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime

from airfare.domain.models import Offer, Price, SearchQuery
from airfare.domain.signature import dedup_offers, offer_signature
from airfare.services.fx import FxService, FxUnavailableError

log = logging.getLogger(__name__)


def _normalize_price(price: Price, fx: FxService | None, at: datetime) -> Price:
    if price.currency == "USD":
        return replace(price, usd=price.amount, fx_rate=1.0, fx_as_of=at.date())
    if fx is None:
        return price
    try:
        rate = fx.rate_to_usd(price.currency, at)
    except FxUnavailableError as e:
        log.warning("FX unavailable for %s: %s", price.currency, e)
        return price
    return replace(price, usd=round(price.amount * rate, 2), fx_rate=rate, fx_as_of=at.date())


def _passes_constraints(offer: Offer, q: SearchQuery) -> bool:
    if offer.stops_out > q.max_stops or offer.stops_return > q.max_stops:
        return False
    return not (q.airlines and offer.airline_code not in q.airlines)


def normalize_offers(
    raw: list[Offer], query: SearchQuery, fx: FxService | None, search_ts: datetime
) -> list[Offer]:
    out: list[Offer] = []
    for o in raw:
        if not _passes_constraints(o, query):
            continue
        normalized = replace(o, price=_normalize_price(o.price, fx, search_ts))
        out.append(replace(normalized, signature=offer_signature(normalized)))
    deduped = dedup_offers(out)
    log.info(
        "normalized %d raw -> %d filtered -> %d unique offers (%d without USD)",
        len(raw), len(out), len(deduped), sum(1 for o in deduped if o.price.usd is None),
    )  # fmt: skip
    return deduped
