"""Page registry shared by the entrypoint (navigation) and pages (links)."""

from __future__ import annotations

import streamlit as st

from airfare.ui.views import about, model, search, trends

SEARCH = st.Page(search.render, title="Search", icon="🔎", default=True)
TRENDS = st.Page(trends.render, title="Route trends", icon="📈", url_path="trends")
MODEL = st.Page(model.render, title="Model", icon="🧠", url_path="model")
ABOUT = st.Page(about.render, title="About", icon="ℹ️", url_path="about")

PAGES = [SEARCH, TRENDS, MODEL, ABOUT]
