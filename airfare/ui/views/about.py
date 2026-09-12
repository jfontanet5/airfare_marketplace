"""About: architecture and design decisions, for reviewers."""

from __future__ import annotations

import streamlit as st

from airfare import __version__
from airfare.ui import components as ui

ARCHITECTURE = """
```
SearchQuery ─▶ Provider.search()    raw offers, priced in the provider's currency
           ─▶ normalize_offers()    FX → USD · itinerary signature · constraints · dedup
           ─▶ score_offers()        fare + stop / date / duration penalties, with reasons
           ─▶ history.record()      observations keyed by (signature, search_ts)
           ─▶ SearchResult          offers · ranked · recommended · cheapest · warnings
```
"""

DECISIONS = [
    (
        "Normalize money once, at the boundary",
        "Every provider quotes in its own currency. `Price(amount, currency, usd, fx_rate, fx_as_of)` "
        "is filled in one place; downstream code reads `usd` and handles `None` instead of guessing.",
    ),
    (
        "Identity is a hash of the segment chain",
        "Same flights, same times, same carriers → same signature, regardless of provider or search day. "
        "That is what makes per-itinerary price history (and therefore honest labels) possible.",
    ),
    (
        "Providers are replaceable — and were replaced",
        "The first live source, Amadeus Self-Service, was decommissioned in July 2026. Swapping in "
        "SerpApi's Google Flights engine touched one provider package and the registry; nothing else.",
    ),
    (
        "Storage is a protocol with versioned migrations",
        "SQLite today; the schema migrates in place, and CSV partitions make the history portable "
        "enough for a scheduled GitHub Action to persist it by committing small files.",
    ),
    (
        "The model says where its numbers come from",
        "One feature definition serves training and inference; the model card records data source, "
        "date range, row counts and metrics; the UI labels a synthetic model as a demo.",
    ),
]


def render() -> None:
    ui.page_header(
        "About",
        f"Airfare Marketplace v{__version__} — an airfare intelligence engine built as a portfolio project.",
    )
    st.markdown(
        "Consumer fare sites obscure how a price got to where it is: results are personalised, currencies "
        "are mixed, history is invisible, and “buy now” nudges have no stated basis. This project builds the "
        "pieces needed to answer *is this a good price for this route, right now?* with data the user can inspect."
    )
    st.subheader("One search, end to end")
    st.markdown(ARCHITECTURE)
    st.subheader("Design decisions")
    for title, body in DECISIONS:
        with st.container(border=True):
            st.markdown(
                f"**{title}**  \n<span class='afm-muted'>{body}</span>", unsafe_allow_html=True
            )
    st.subheader("Stack")
    st.markdown(
        "Python 3.12 · pydantic-settings · requests + tenacity · pandas · scikit-learn · SQLite · Streamlit + Altair · "
        "ruff · mypy --strict · pytest (AppTest for the UI) · GitHub Actions · Docker"
    )
    st.link_button("Source on GitHub", "https://github.com/jfontanet5/airfare_marketplace")
