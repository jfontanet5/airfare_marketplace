from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from airfare.collect.cli import main
from airfare.collect.partitions import export_day, import_partitions
from airfare.collect.watchlist import Route, load_watchlist, parse_routes, plan_queries
from airfare.config import get_settings
from airfare.domain.models import Price, SearchQuery
from airfare.storage.repository import Observation
from airfare.storage.sqlite import SqlitePriceHistory

from tests.conftest import make_offer


def test_route_parsing() -> None:
    assert parse_routes("sju-jfk, MIA->SJU") == [Route("SJU", "JFK"), Route("MIA", "SJU")]
    with pytest.raises(ValueError):
        Route.parse("SJUJFK")


def test_watchlist_file_and_plan(tmp_path: Path) -> None:
    f = tmp_path / "w.txt"
    f.write_text("# comment\nSJU-JFK\nMIA-JFK  # inline\n\n")
    routes = load_watchlist(f)
    assert [str(r) for r in routes] == ["SJU-JFK", "MIA-JFK"]
    qs = plan_queries(routes, [14, 30], date(2026, 9, 12), trip_length_days=7)
    assert len(qs) == 4
    assert qs[0].departure_date == date(2026, 9, 26) and qs[0].return_date == date(2026, 10, 3)
    assert plan_queries(routes, [14], date(2026, 9, 12), None)[0].return_date is None


def test_cli_dry_run_and_budget(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "--routes", "SJU-JFK", "--horizons", "14,30", "--dry-run"]) == 0
    assert (
        main(["run", "--routes", "SJU-JFK,MIA-JFK", "--horizons", "14,30", "--max-requests", "3"])
        == 2
    )


def test_cli_run_with_mock_and_export_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "h.sqlite"
    monkeypatch.setenv("AIRFARE_DB_PATH", str(db))
    monkeypatch.setenv("SERPAPI_API_KEY", "")
    get_settings.cache_clear()
    parts = tmp_path / "parts"
    code = main(
        [
            "run",
            "--routes",
            "SJU-JFK",
            "--horizons",
            "20",
            "--provider",
            "mock",
            "--export",
            "--partitions",
            str(parts),
        ]
    )
    assert code == 0
    files = list(parts.glob("*.csv"))
    assert len(files) == 1 and files[0].stem == datetime.now(UTC).date().isoformat()

    fresh = SqlitePriceHistory(tmp_path / "fresh.sqlite")
    n = import_partitions(fresh, parts)
    assert n == len(pd.read_csv(files[0]))
    assert import_partitions(fresh, parts) == 0  # idempotent
    assert fresh.route_price_stats("SJU", "JFK")["count"] == n
    get_settings.cache_clear()


def test_export_day_empty(tmp_path: Path) -> None:
    history = SqlitePriceHistory(tmp_path / "h.sqlite")
    assert export_day(history, date(2020, 1, 1), tmp_path / "out") is None
    assert not (tmp_path / "out").exists()


def test_export_uses_utc_day(tmp_path: Path) -> None:

    history = SqlitePriceHistory(tmp_path / "h.sqlite")
    q = SearchQuery("SJU", "JFK", date(2026, 10, 1))
    ts = datetime(2026, 9, 12, 23, 30, tzinfo=UTC)
    o = replace(make_offer(), price=Price(300, "USD", usd=300.0, fx_rate=1.0), signature="s1")
    history.record([Observation.from_offer(o, q, ts)])
    assert export_day(history, date(2026, 9, 12), tmp_path / "out") is not None
    assert export_day(history, date(2026, 9, 12) + timedelta(days=1), tmp_path / "out") is None
