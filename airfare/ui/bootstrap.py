"""Cached, process-wide resources for the Streamlit app."""

from __future__ import annotations

import logging

import streamlit as st

from airfare.config import Settings, get_settings
from airfare.ml.predict import PriceDropPredictor
from airfare.services.airports import Airport, load_airports
from airfare.services.fx import FxService
from airfare.storage.sqlite import SqlitePriceHistory


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )


@st.cache_resource(show_spinner=False)
def settings() -> Settings:
    return get_settings()


@st.cache_resource(show_spinner=False)
def history() -> SqlitePriceHistory:
    return SqlitePriceHistory(settings().airfare_db_path)


@st.cache_resource(show_spinner=False)
def fx() -> FxService:
    s = settings()
    return FxService.from_settings(s.airfare_db_path, s.twelvedata_api_key)


@st.cache_resource(show_spinner=False)
def predictor() -> PriceDropPredictor | None:
    return PriceDropPredictor.load(settings().airfare_model_dir)


@st.cache_data(show_spinner=False, ttl=3600)
def airports(include_full: bool) -> list[Airport]:
    s = settings()
    return load_airports(s.airfare_sample_dir, s.airfare_airports_cache, include_full)
