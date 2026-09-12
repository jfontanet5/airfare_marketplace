import pytest
import requests
import responses as rsp
from airfare.providers.amadeus.client import AmadeusClient
from airfare.providers.base import (
    ProviderAuthError,
    ProviderRateLimitedError,
    ProviderUnavailableError,
)

TOKEN = "https://test.api.amadeus.com/v1/security/oauth2/token"
OFFERS = "https://test.api.amadeus.com/v2/shopping/flight-offers"


def test_missing_credentials() -> None:
    with pytest.raises(ProviderAuthError):
        AmadeusClient("", "")


@rsp.activate
def test_token_cached_and_reused() -> None:
    rsp.add(rsp.POST, TOKEN, json={"access_token": "t1", "expires_in": 1800})
    rsp.add(rsp.GET, OFFERS, json={"data": []})
    rsp.add(rsp.GET, OFFERS, json={"data": []})
    c = AmadeusClient("id", "secret")
    c.get("/v2/shopping/flight-offers", {})
    c.get("/v2/shopping/flight-offers", {})
    assert len([r for r in rsp.calls if r.request.url == TOKEN]) == 1


@rsp.activate
def test_rejected_credentials() -> None:
    rsp.add(rsp.POST, TOKEN, status=401, json={})
    with pytest.raises(ProviderAuthError):
        AmadeusClient("id", "bad").get("/x", {})


@rsp.activate
def test_401_refreshes_token_once() -> None:
    rsp.add(rsp.POST, TOKEN, json={"access_token": "t1", "expires_in": 1800})
    rsp.add(rsp.GET, OFFERS, status=401, json={})
    rsp.add(rsp.POST, TOKEN, json={"access_token": "t2", "expires_in": 1800})
    rsp.add(rsp.GET, OFFERS, json={"data": [1]})
    c = AmadeusClient("id", "secret")
    c._get_once.retry.wait = lambda *_: 0  # type: ignore[attr-defined]
    assert c.get("/v2/shopping/flight-offers", {})["data"] == [1]


@rsp.activate
def test_rate_limit_surfaces_typed_error() -> None:
    rsp.add(rsp.POST, TOKEN, json={"access_token": "t1", "expires_in": 1800})
    for _ in range(4):
        rsp.add(rsp.GET, OFFERS, status=429, json={})
    c = AmadeusClient("id", "secret")
    c._get_once.retry.wait = lambda *_: 0  # type: ignore[attr-defined]
    with pytest.raises(ProviderRateLimitedError):
        c.get("/v2/shopping/flight-offers", {})


@rsp.activate
def test_network_error_becomes_unavailable() -> None:
    rsp.add(rsp.POST, TOKEN, json={"access_token": "t1", "expires_in": 1800})
    for _ in range(4):
        rsp.add(rsp.GET, OFFERS, body=requests.ConnectionError("boom"))
    c = AmadeusClient("id", "secret")
    c._get_once.retry.wait = lambda *_: 0  # type: ignore[attr-defined]
    with pytest.raises(ProviderUnavailableError):
        c.get("/v2/shopping/flight-offers", {})
