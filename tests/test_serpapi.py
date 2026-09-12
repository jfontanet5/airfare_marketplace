import json
from datetime import date

import pytest
import requests
import responses as rsp
from airfare.domain.models import SearchQuery
from airfare.providers.base import (
    ProviderAuthError,
    ProviderRateLimitedError,
    ProviderUnavailableError,
)
from airfare.providers.serpapi.client import SERPAPI_URL, SerpApiClient
from airfare.providers.serpapi.parser import (
    parse_price_insights,
    parse_response,
    parse_return_options,
    split_flight_number,
)
from airfare.providers.serpapi.provider import SerpApiProvider

from tests.conftest import FIXTURES

PAYLOAD = json.loads((FIXTURES / "serpapi_google_flights.json").read_text())
RETURN = json.loads((FIXTURES / "serpapi_return_leg.json").read_text())
RT = date(2026, 10, 5)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("B6 704", ("B6", "704")),
        ("AA1092", ("AA", "1092")),
        ("weird", (None, "weird")),
        (None, (None, None)),
    ],
)
def test_split_flight_number(raw: str | None, expected: tuple[str | None, str | None]) -> None:
    assert split_flight_number(raw) == expected


def test_parse_recorded_response() -> None:
    pairs = parse_response(PAYLOAD, "SJU", "JFK", RT, "USD")
    assert len(pairs) == 2  # the Spirit entry has no price and is dropped
    (jetblue, tok_b6), (american, tok_aa) = pairs
    assert tok_b6 == "TOKEN_B6" and tok_aa == "TOKEN_AA"
    assert jetblue.airline_code == "B6" and jetblue.airline_name == "JetBlue"
    assert (
        jetblue.price.amount == 312
        and jetblue.price.currency == "USD"
        and jetblue.price.usd is None
    )
    assert jetblue.departure_date == date(2026, 10, 1) and jetblue.return_date == RT
    assert jetblue.outbound is not None and jetblue.outbound.duration_minutes == 225
    assert jetblue.inbound is None
    assert american.stops_out == 1 and american.total_duration_minutes == 425
    assert (
        american.outbound is not None and american.outbound.layovers()[0].total_seconds() == 85 * 60
    )
    assert parse_price_insights(PAYLOAD) == {
        "lowest_price": 298,
        "price_level": "typical",
        "typical_price_range": [260, 380],
    }


def test_parse_return_options() -> None:
    opts = parse_return_options(RETURN)
    assert [p for _, p in opts] == [312.0, 340.0]
    assert opts[0][0].segments[0].flight_number == "1155"


def test_client_requires_key() -> None:
    with pytest.raises(ProviderAuthError):
        SerpApiClient("")


@rsp.activate
def test_client_error_mapping() -> None:
    c = SerpApiClient("k")
    c._get_once.retry.wait = lambda *_: 0  # type: ignore[attr-defined]
    rsp.add(rsp.GET, SERPAPI_URL, status=401, json={})
    with pytest.raises(ProviderAuthError):
        c.search({})
    rsp.add(rsp.GET, SERPAPI_URL, status=429, json={})
    with pytest.raises(ProviderRateLimitedError):
        c.search({})
    rsp.add(rsp.GET, SERPAPI_URL, json={"error": "Missing query `departure_id`"})
    with pytest.raises(ProviderUnavailableError):
        c.search({})
    rsp.add(
        rsp.GET, SERPAPI_URL, json={"error": "Google hasn't returned any results for this query."}
    )
    assert c.search({}) == {"best_flights": [], "other_flights": []}
    for _ in range(3):
        rsp.add(rsp.GET, SERPAPI_URL, body=requests.ConnectionError("boom"))
    with pytest.raises(ProviderUnavailableError):
        c.search({})


@rsp.activate
def test_provider_roundtrip_attaches_return_leg_for_top_n() -> None:
    rsp.add(rsp.GET, SERPAPI_URL, json=PAYLOAD)  # outbound search
    rsp.add(rsp.GET, SERPAPI_URL, json=RETURN)  # return leg for the cheapest (American, 298)
    provider = SerpApiProvider(SerpApiClient("k"), return_legs_top_n=1)
    q = SearchQuery("SJU", "JFK", date(2026, 10, 1), RT, max_stops=1)
    offers = provider.search(q)

    assert len(rsp.calls) == 2
    first_params = rsp.calls[0].request.params  # type: ignore[union-attr]
    assert (
        first_params["type"] == "1" and first_params["stops"] == "2" and "api_key" in first_params
    )
    assert rsp.calls[1].request.params["departure_token"] == "TOKEN_AA"  # type: ignore[union-attr]

    enriched = next(o for o in offers if o.airline_code == "AA")
    assert enriched.inbound is not None and enriched.inbound.segments[0].flight_number == "1155"
    assert enriched.price.amount == 312.0  # cheapest pairing's round-trip total
    untouched = next(o for o in offers if o.airline_code == "B6")
    assert untouched.inbound is None and untouched.price.amount == 312


@rsp.activate
def test_provider_one_way_single_request() -> None:
    rsp.add(rsp.GET, SERPAPI_URL, json=PAYLOAD)
    offers = SerpApiProvider(SerpApiClient("k"), return_legs_top_n=5).search(
        SearchQuery("SJU", "JFK", date(2026, 10, 1), airlines=frozenset({"B6"}))
    )
    assert len(rsp.calls) == 1
    assert rsp.calls[0].request.params["type"] == "2"  # type: ignore[union-attr]
    assert rsp.calls[0].request.params["include_airlines"] == "B6"  # type: ignore[union-attr]
    assert all(o.return_date is None for o in offers)
