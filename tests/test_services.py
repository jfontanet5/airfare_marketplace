from datetime import UTC, date, datetime, timedelta

from airfare.domain.models import SearchQuery
from airfare.providers.mock import MockProvider
from airfare.services.fx import FxService
from airfare.services.normalization import normalize_offers
from airfare.services.search import SearchService
from airfare.storage.sqlite import SqlitePriceHistory

from tests.conftest import StaticFx, make_offer

TS = datetime(2026, 9, 12, 15, 0, tzinfo=UTC)


def test_fx_caches_per_day(fx: FxService) -> None:
    src = fx.sources[0]
    assert fx.rate_to_usd("EUR", TS) == 1.10
    assert fx.rate_to_usd("eur", TS) == 1.10
    assert fx.rate_to_usd("USD", TS) == 1.0
    assert src.calls == 1  # type: ignore[attr-defined]
    fx._mem.clear()  # sqlite cache still hit
    assert fx.rate_to_usd("EUR", TS) == 1.10
    assert src.calls == 1  # type: ignore[attr-defined]


def test_fx_falls_through_sources(tmp_path) -> None:  # type: ignore[no-untyped-def]
    class Broken:
        name = "broken"

        def rate_to_usd(self, currency: str, day: date) -> float:
            raise RuntimeError("down")

    fx = FxService(tmp_path / "fx.sqlite", [Broken(), StaticFx({"EUR": 1.2})])
    assert fx.rate_to_usd("EUR", TS) == 1.2


def test_normalization_converts_filters_and_dedups(fx: FxService, query: SearchQuery) -> None:
    raw = [
        make_offer(price=200, currency="EUR"),
        make_offer(price=230, currency="EUR"),  # duplicate itinerary, worse price
        make_offer(price=150, stops=1, carrier="AA"),
        make_offer(price=100, stops=3, carrier="NK"),  # exceeds max_stops=2
    ]
    out = normalize_offers(raw, query, fx, TS)
    assert [o.price.usd for o in out] == [220.0, 150.0]
    assert all(o.signature for o in out)
    assert out[0].price.fx_rate == 1.10 and out[0].price.fx_as_of == TS.date()

    filtered = normalize_offers(
        raw, SearchQuery("SJU", "JFK", date(2026, 10, 1), airlines=frozenset({"AA"})), fx, TS
    )
    assert [o.airline_code for o in filtered] == ["AA"]


def test_normalization_without_fx_marks_unpriced(query: SearchQuery) -> None:
    out = normalize_offers([make_offer(price=200, currency="EUR")], query, None, TS)
    assert out[0].price.usd is None and out[0].price.currency == "EUR"


def test_search_service_end_to_end(tmp_path, fx: FxService) -> None:  # type: ignore[no-untyped-def]
    history = SqlitePriceHistory(tmp_path / "h.sqlite")
    q = SearchQuery(
        "SJU", "JFK", date(2026, 10, 1), date(2026, 10, 5), max_stops=2, flexible_dates=True
    )
    result = SearchService(MockProvider(), fx, history).search(q)
    assert result.offers and result.recommended is not None
    assert len({o.signature for o in result.offers}) == len(result.offers)
    assert {o.departure_date for o in result.offers} == {
        date(2026, 10, 1) + timedelta(days=i) for i in range(-3, 4)
    }
    assert result.observations_persisted > 0
    assert result.cheapest is not None and result.cheapest.price.usd == min(
        o.price.usd for o in result.offers if o.price.usd
    )
    assert history.route_price_stats("SJU", "JFK")["count"] == result.observations_persisted


def test_mock_provider_is_deterministic(query: SearchQuery) -> None:
    a, b = MockProvider().search(query), MockProvider().search(query)
    assert [o.price.amount for o in a] == [o.price.amount for o in b]
    assert any(o.price.currency == "EUR" for o in a)
