from dataclasses import replace
from datetime import date, timedelta

import pytest
from airfare.domain.models import Price, SearchQuery
from airfare.domain.scoring import ScoringWeights, cheapest, score_offer, score_offers
from airfare.domain.signature import dedup_offers, offer_signature

from tests.conftest import make_offer


def test_search_query_normalizes_and_validates() -> None:
    q = SearchQuery(" sju", "jfk ", date(2026, 10, 1))
    assert (q.origin, q.destination) == ("SJU", "JFK")
    assert q.trip_type == "one_way"
    with pytest.raises(ValueError):
        SearchQuery("SJU", "SJU", date(2026, 10, 1))
    with pytest.raises(ValueError):
        SearchQuery("SJU", "JFK", date(2026, 10, 5), date(2026, 10, 1))


def test_shifted_keeps_trip_length() -> None:
    q = SearchQuery("SJU", "JFK", date(2026, 10, 1), date(2026, 10, 5), flexible_dates=True)
    s = q.shifted(-2)
    assert s.departure_date == date(2026, 9, 29)
    assert s.return_date == date(2026, 10, 3)
    assert s.trip_length == timedelta(days=4)
    assert not s.flexible_dates


def test_price_uppercases_currency_and_rejects_negative() -> None:
    assert Price(10, "eur").currency == "EUR"
    with pytest.raises(ValueError):
        Price(-1, "USD")


def test_signature_is_stable_and_provider_independent() -> None:
    a = make_offer(price=300, provider="amadeus")
    b = make_offer(price=999, provider="mock")
    assert offer_signature(a) == offer_signature(b)
    assert offer_signature(a) != offer_signature(make_offer(dep_hour=9))
    assert offer_signature(a) != offer_signature(make_offer(carrier="AA"))


def test_dedup_keeps_cheapest_per_signature() -> None:
    expensive = replace(make_offer(price=400), price=Price(400, "USD", usd=400))
    cheap = replace(make_offer(price=300), price=Price(300, "USD", usd=300))
    other = replace(make_offer(price=100, carrier="AA"), price=Price(100, "USD", usd=100))
    out = dedup_offers([expensive, cheap, other])
    assert [o.price.usd for o in out] == [300, 100]


def test_scoring_penalties_and_reasons(query: SearchQuery) -> None:
    w = ScoringWeights(
        stop_penalty_usd=35, date_offset_penalty_usd=5, duration_penalty_usd_per_hour=0
    )
    nonstop = replace(make_offer(price=300), price=Price(300, "USD", usd=300))
    one_stop = replace(make_offer(price=280, stops=1), price=Price(280, "USD", usd=280))
    off_date = replace(
        make_offer(price=250, dep_date=date(2026, 10, 3)), price=Price(250, "USD", usd=250)
    )
    s = score_offers([nonstop, one_stop, off_date], query, w)
    assert [x.score for x in s] == [260.0, 300.0, 315.0]
    assert any("off requested date" in r for r in s[0].reasons)
    assert "Nonstop" in s[1].reasons


def test_unpriced_offer_scores_infinite(query: SearchQuery) -> None:
    s = score_offer(make_offer(currency="EUR"), query)
    assert s.score == float("inf")
    assert cheapest([make_offer(currency="EUR")]) is None
