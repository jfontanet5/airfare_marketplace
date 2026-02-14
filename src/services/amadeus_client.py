# src/services/amadeus_client.py

import os
import time
from typing import Any, Dict, Optional
import requests


class AmadeusClient:
    """
    Minimal Amadeus REST client with OAuth2 client-credentials token caching.
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        env: Optional[str] = None,
        timeout_seconds: int = 20,
    ):
        self.client_id = (client_id or os.getenv(
            "AMADEUS_CLIENT_ID", "")).strip()
        self.client_secret = (client_secret or os.getenv(
            "AMADEUS_CLIENT_SECRET", "")).strip()
        self.env = (env or os.getenv("AMADEUS_ENV", "test")).strip().lower()
        self.timeout_seconds = timeout_seconds

        if not self.client_id or not self.client_secret:
            raise ValueError(
                "Missing Amadeus credentials. Set AMADEUS_CLIENT_ID and AMADEUS_CLIENT_SECRET."
            )

        # Base URLs for Self-Service APIs
        self.base_url = (
            "https://test.api.amadeus.com"
            if self.env == "test"
            else "https://api.amadeus.com"
        )

        self._access_token: Optional[str] = None
        self._token_expiry_epoch: float = 0.0  # seconds since epoch

    def _token_is_valid(self) -> bool:
        # Refresh 60 seconds early
        return bool(self._access_token) and (time.time() < (self._token_expiry_epoch - 60))

    def _fetch_token(self) -> None:
        url = f"{self.base_url}/v1/security/oauth2/token"
        data = {"grant_type": "client_credentials"}
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        # Use HTTP Basic Auth (most reliable for OAuth client_credentials)
        resp = requests.post(
            url,
            data=data,
            headers=headers,
            auth=(self.client_id, self.client_secret),
            timeout=self.timeout_seconds,
        )

        # Helpful error detail without leaking secrets
        if resp.status_code != 200:
            raise requests.HTTPError(
                f"Amadeus token request failed: {resp.status_code} {resp.text}",
                response=resp,
            )

        payload = resp.json()
        self._access_token = payload["access_token"]
        expires_in = int(payload.get("expires_in", 1800))
        self._token_expiry_epoch = time.time() + expires_in

    def _get_auth_header(self) -> Dict[str, str]:
        if not self._token_is_valid():
            self._fetch_token()
        return {"Authorization": f"Bearer {self._access_token}"}

    def get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = self._get_auth_header()
        resp = requests.get(url, params=params,
                            headers=headers, timeout=self.timeout_seconds)

        # If token expired unexpectedly, refresh once and retry
        if resp.status_code == 401:
            self._fetch_token()
            headers = self._get_auth_header()
            resp = requests.get(url, params=params,
                                headers=headers, timeout=self.timeout_seconds)

        resp.raise_for_status()
        return resp.json()
