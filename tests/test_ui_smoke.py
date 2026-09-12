"""Drive the Streamlit app headlessly with AppTest."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import streamlit as st
from airfare.config import get_settings
from airfare.domain.models import SearchQuery
from airfare.ml.train import train_synthetic
from airfare.providers.mock import MockProvider
from airfare.services.search import SearchService
from airfare.storage.sqlite import SqlitePriceHistory
from streamlit.testing.v1 import AppTest

APP = Path(__file__).parent.parent / "airfare" / "ui" / "app.py"


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("AIRFARE_DB_PATH", str(tmp_path / "h.sqlite"))
    monkeypatch.setenv("AIRFARE_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("SERPAPI_API_KEY", "")
    monkeypatch.setenv("TWELVEDATA_API_KEY", "")
    _reset_caches()
    yield tmp_path
    _reset_caches()


def _reset_caches() -> None:
    """Settings and Streamlit resource caches are process-wide; each test gets a fresh DB."""
    get_settings.cache_clear()
    st.cache_resource.clear()
    st.cache_data.clear()


def _page(name: str) -> AppTest:
    """AppTest.from_function runs the function body as a bare script, so import inside."""
    script = f"from airfare.ui.views import {name}\n{name}.render()\n"
    return AppTest.from_string(script, default_timeout=60)


def _seed_history(db: Path) -> None:
    history = SqlitePriceHistory(db)
    q = SearchQuery("SJU", "JFK", date(2026, 11, 1), date(2026, 11, 8))
    SearchService(MockProvider(), None, history).search(q)


def test_search_page_end_to_end(env: Path) -> None:
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert any("Know when a fare" in md.value for md in at.markdown)

    labels = at.selectbox[0].options
    at.selectbox[0].select(next(o for o in labels if o.startswith("SJU")))
    at.selectbox[1].select(next(o for o in labels if o.startswith("JFK")))
    at.button[0].click().run()

    assert not at.exception, at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert int(metrics["Options"]) >= 3
    assert metrics["Cheapest"].startswith("$")
    assert any("Recommended" in md.value for md in at.markdown)
    assert any(t.label.startswith("Results (") for t in at.tabs)


def test_trends_page_empty_and_seeded(env: Path) -> None:
    at = _page("trends").run()
    assert not at.exception and any("No history yet" in i.value for i in at.info)

    _seed_history(env / "h.sqlite")
    st.cache_data.clear()
    at = _page("trends").run()
    assert not at.exception, at.exception
    assert at.selectbox[0].options[0].startswith("SJU → JFK")
    assert {m.label for m in at.metric} >= {"Observations", "Lowest seen", "Median"}
    # switch to a specific departure date -> per-itinerary table renders
    at.selectbox[1].select(at.selectbox[1].options[1]).run()
    assert not at.exception, at.exception
    assert any("Per-itinerary movement" in h.value for h in at.subheader)


def test_model_page_without_and_with_model(env: Path) -> None:
    at = _page("model").run()
    assert not at.exception and any("No model loaded" in w.value for w in at.warning)

    train_synthetic(env / "models", n_rows=1200)
    st.cache_resource.clear()
    at = _page("model").run()
    assert not at.exception, at.exception
    assert {m.label for m in at.metric} >= {"ROC-AUC", "Brier score"}
    assert any("synthetic generating process" in i.value for i in at.info)


def test_about_page(env: Path) -> None:
    at = _page("about").run()
    assert not at.exception
    assert any("Design decisions" in h.value for h in at.subheader)
