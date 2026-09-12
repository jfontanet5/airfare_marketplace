"""Cached, process-wide resources for the Streamlit app.

Also makes a fresh deployment self-sufficient: secrets flow into settings,
the history store is seeded from committed CSV partitions, and a demo model is
trained on first start when none is present.
"""

from __future__ import annotations

import logging
import os

import streamlit as st

from airfare.collect.partitions import import_partitions
from airfare.config import Settings, get_settings
from airfare.ml.predict import PriceDropPredictor
from airfare.ml.train import train_synthetic
from airfare.services.airports import Airport, load_airports
from airfare.services.fx import FxService
from airfare.storage.sqlite import SqlitePriceHistory

log = logging.getLogger(__name__)

SECRET_KEYS = ("SERPAPI_API_KEY", "TWELVEDATA_API_KEY", "SERPAPI_RETURN_LEGS_TOP_N")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )


def _secrets_to_env() -> None:
    """Streamlit Cloud provides secrets via st.secrets; settings read the environment."""
    try:
        for key in SECRET_KEYS:
            if key in st.secrets and key not in os.environ:
                os.environ[key] = str(st.secrets[key])
    except Exception:
        return


@st.cache_resource(show_spinner=False)
def settings() -> Settings:
    _secrets_to_env()
    get_settings.cache_clear()
    return get_settings()


@st.cache_resource(show_spinner=False)
def history() -> SqlitePriceHistory:
    store = SqlitePriceHistory(settings().airfare_db_path)
    partitions = settings().airfare_seed_dir
    if partitions.is_dir() and store.routes().empty:
        n = import_partitions(store, partitions)
        log.info("seeded history store with %d observations from %s", n, partitions)
    return store


@st.cache_resource(show_spinner=False)
def fx() -> FxService:
    s = settings()
    return FxService.from_settings(s.airfare_db_path, s.twelvedata_api_key)


@st.cache_resource(show_spinner="Training the demo model (first start only)…")
def predictor() -> PriceDropPredictor | None:
    model_dir = settings().airfare_model_dir
    loaded = PriceDropPredictor.load(model_dir)
    if loaded is None and os.environ.get("AIRFARE_AUTOTRAIN_DEMO", "1") == "1":
        train_synthetic(model_dir, n_rows=4000)
        loaded = PriceDropPredictor.load(model_dir)
    return loaded


@st.cache_data(show_spinner=False, ttl=3600)
def airports(include_full: bool) -> list[Airport]:
    s = settings()
    return load_airports(s.airfare_sample_dir, s.airfare_airports_cache, include_full)
