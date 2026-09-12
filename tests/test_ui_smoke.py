"""Drive the Streamlit app headlessly with AppTest: form -> search -> render."""

from __future__ import annotations

from pathlib import Path

import pytest
from airfare.config import get_settings
from streamlit.testing.v1 import AppTest

APP = Path(__file__).parent.parent / "airfare" / "ui" / "app.py"


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setenv("AIRFARE_DB_PATH", str(tmp_path / "h.sqlite"))
    monkeypatch.setenv("AIRFARE_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("AMADEUS_CLIENT_ID", "")
    monkeypatch.setenv("AMADEUS_CLIENT_SECRET", "")
    monkeypatch.setenv("TWELVEDATA_API_KEY", "")
    get_settings.cache_clear()
    return AppTest.from_file(str(APP), default_timeout=60)


def test_renders_and_searches_offline(app: AppTest) -> None:
    at = app.run()
    assert not at.exception
    assert at.title[0].value.endswith("Airfare Marketplace")

    labels = at.selectbox[0].options
    sju = next(o for o in labels if o.startswith("SJU"))
    jfk = next(o for o in labels if o.startswith("JFK"))
    at.selectbox[0].select(sju)
    at.selectbox[1].select(jfk)
    at.button[0].click().run()

    assert not at.exception, at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert int(metrics["Options"]) >= 3
    assert metrics["Cheapest"].startswith("$")
    assert any("Recommended" in md.value for md in at.markdown)
