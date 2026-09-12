"""Amadeus Self-Service REST client: OAuth2 client-credentials with token caching,
retry with exponential backoff on transient failures, and typed errors."""

from __future__ import annotations

import logging
import time
from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from airfare.providers.base import (
    ProviderAuthError,
    ProviderRateLimitedError,
    ProviderUnavailableError,
)

log = logging.getLogger(__name__)

_BASE_URLS = {"test": "https://test.api.amadeus.com", "production": "https://api.amadeus.com"}


class _TransientError(Exception):
    """Internal marker: retryable failure."""


class AmadeusClient:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        env: str = "test",
        timeout_seconds: float = 20.0,
        session: requests.Session | None = None,
    ) -> None:
        if not client_id or not client_secret:
            raise ProviderAuthError("Amadeus credentials are not configured")
        self.base_url = _BASE_URLS.get(env.lower(), _BASE_URLS["test"])
        self._id, self._secret = client_id, client_secret
        self.timeout = timeout_seconds
        self.http = session or requests.Session()
        self._token: str | None = None
        self._token_expiry = 0.0

    # -- auth -----------------------------------------------------------------

    def _token_valid(self) -> bool:
        return bool(self._token) and time.monotonic() < self._token_expiry - 60

    def _fetch_token(self) -> None:
        try:
            resp = self.http.post(
                f"{self.base_url}/v1/security/oauth2/token",
                data={"grant_type": "client_credentials"},
                auth=(self._id, self._secret),
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise ProviderUnavailableError(f"Amadeus token endpoint unreachable: {e}") from e
        if resp.status_code in (400, 401, 403):
            raise ProviderAuthError("Amadeus rejected the client credentials")
        if resp.status_code != 200:
            raise ProviderUnavailableError(f"Amadeus token request failed ({resp.status_code})")
        payload = resp.json()
        self._token = payload["access_token"]
        self._token_expiry = time.monotonic() + int(payload.get("expires_in", 1800))
        log.debug("amadeus token refreshed, expires_in=%s", payload.get("expires_in"))

    def _auth_header(self) -> dict[str, str]:
        if not self._token_valid():
            self._fetch_token()
        return {"Authorization": f"Bearer {self._token}"}

    # -- requests -------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(_TransientError),
        stop=stop_after_attempt(4),
        wait=wait_exponential_jitter(initial=1, max=10),
        reraise=True,
    )
    def _get_once(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = self.http.get(
                url, params=params, headers=self._auth_header(), timeout=self.timeout
            )
        except requests.RequestException as e:
            raise _TransientError(str(e)) from e

        if resp.status_code == 401:  # token invalidated server-side: refresh and retry
            self._token = None
            raise _TransientError("401")
        if resp.status_code == 429:
            raise _TransientError("429")
        if resp.status_code >= 500:
            raise _TransientError(str(resp.status_code))
        if resp.status_code >= 400:
            detail = _error_detail(resp)
            raise ProviderUnavailableError(f"Amadeus returned {resp.status_code}: {detail}")
        return resp.json()  # type: ignore[no-any-return]

    def get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._get_once(f"{self.base_url}{path}", params)
        except _TransientError as e:
            if str(e) == "429":
                raise ProviderRateLimitedError(
                    "Amadeus rate limit reached; try again shortly"
                ) from e
            raise ProviderUnavailableError(f"Amadeus unavailable after retries: {e}") from e


def _error_detail(resp: requests.Response) -> str:
    try:
        errors = resp.json().get("errors") or []
        return "; ".join(f"{e.get('title')}: {e.get('detail')}" for e in errors) or resp.text[:200]
    except ValueError:
        return resp.text[:200]
