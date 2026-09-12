"""Airfare Marketplace — Streamlit entrypoint.

Thin presentation layer: pages call :class:`airfare.services.search.SearchService`
and the history repository; nothing here knows about providers or SQL.
"""

from __future__ import annotations

import streamlit as st

from airfare.ui import bootstrap
from airfare.ui import components as ui
from airfare.ui.nav import PAGES

st.set_page_config(
    page_title="Airfare Marketplace",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get help": None, "Report a bug": None, "About": "Airfare Marketplace"},
)
bootstrap.configure_logging()
ui.inject_css()

with st.sidebar:
    st.markdown("### ✈️ Airfare Marketplace")
    settings = bootstrap.settings()
    st.caption(
        "Live search: " + ("on (SerpApi)" if settings.live_configured else "off — demo & replay")
    )
    predictor = bootstrap.predictor()
    model_state = (
        "none loaded"
        if predictor is None
        else ("synthetic demo" if predictor.card.is_synthetic else "trained on observations")
    )
    st.caption(f"Model: {model_state}")

st.navigation(PAGES).run()
