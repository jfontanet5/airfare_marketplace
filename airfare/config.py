"""Application settings.

All configuration comes from environment variables (or a ``.env`` file in the
working directory). Every external credential is optional: the app degrades to
offline providers and keyless FX when a key is absent.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict



def _project_root() -> Path:
    """Repo root when running from a source checkout; otherwise the working directory.

    An installed (non-editable) package lives in site-packages, where data/ and models/
    must not go — CI, Docker and any `pip install .` user run from their project dir.
    """
    candidate = Path(__file__).resolve().parent.parent
    return candidate if (candidate / "pyproject.toml").exists() else Path.cwd()


PROJECT_ROOT = _project_root()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # SerpApi (Google Flights engine)
    serpapi_api_key: str = ""
    serpapi_currency: str = "USD"
    serpapi_return_legs_top_n: int = Field(
        default=0,
        description="Round trips: fetch the return leg for the N cheapest outbound options",
    )

    # FX
    twelvedata_api_key: str = ""

    # Paths
    airfare_db_path: Path = PROJECT_ROOT / "data" / "price_history.sqlite"
    airfare_model_dir: Path = PROJECT_ROOT / "models"
    airfare_sample_dir: Path = PROJECT_ROOT / "data" / "sample"
    airfare_airports_cache: Path = PROJECT_ROOT / "data" / "airports_full.csv"

    # Scoring weights (USD-equivalent penalties)
    score_stop_penalty_usd: float = 35.0
    score_date_offset_penalty_usd: float = 5.0
    score_duration_penalty_usd_per_hour: float = 4.0

    # Behaviour
    flexible_window_days: int = 3
    history_top_n: int = 30
    search_cache_ttl_seconds: int = 900

    @property
    def live_configured(self) -> bool:
        return bool(self.serpapi_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
