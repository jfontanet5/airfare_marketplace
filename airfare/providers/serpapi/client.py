"""SerpApi client: thin HTTP wrapper with retry/backoff and typed errors."""

from __future__ import annotations

import logging
from typing import Any

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from airfare.providers.base import (
    ProviderAuthError,
    ProviderRateLimitedError,
    ProviderUnavailableError,
)

log = logging.getLogger(__name__)

SERPAPI_URL = "https://serpapi.com/search.json"


class _TransientError(Exception):
    """Internal marker: retryable failure."""


class SerpApiClient:
    def __init__(
        self,
        api_key: str,
        timeout_seconds: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        if not api_key:
            raise ProviderAuthError("SerpApi key is not configured")
        self.api_key = api_key
        self.timeout = timeout_seconds
        self.http = session or requests.Session()

    @retry(
        retry=retry_if_exception_type(_TransientError),
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=8),
        reraise=True,
    )
    def _get_once(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = self.http.get(
                SERPAPI_URL, params={**params, "api_key": self.api_key}, timeout=self.timeout
            )
        except requests.RequestException as e:
            raise _TransientError(str(e)) from e

        if resp.status_code in (401, 403):
            raise ProviderAuthError("SerpApi rejected the API key")
        if resp.status_code == 429:
            raise ProviderRateLimitedError("SerpApi quota or rate limit reached")
        if resp.status_code >= 500:
            raise _TransientError(str(resp.status_code))
        if resp.status_code >= 400:
            raise ProviderUnavailableError(
                f"SerpApi returned {resp.status_code}: {resp.text[:200]}"
            )

        payload: dict[str, Any] = resp.json()
        # SerpApi reports parameter problems as HTTP 200 with an "error" key.
        if "error" in payload:
            msg = str(payload["error"])
            if "hasn't returned any results" in msg.lower():
                return {"best_flights": [], "other_flights": []}
            raise ProviderUnavailableError(f"SerpApi error: {msg}")
        return payload

    def search(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._get_once(params)
        except _TransientError as e:
            raise ProviderUnavailableError(f"SerpApi unavailable after retries: {e}") from e
