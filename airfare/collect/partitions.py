"""CSV partitions: a portable, diff-friendly copy of the observation store.

One file per collection day under ``data/observations/``. They let a scheduled
job (e.g. GitHub Actions) persist results by committing small files, and let a
fresh clone rebuild the SQLite store with ``airfare-collect import``.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from airfare.storage.repository import Observation
from airfare.storage.sqlite import SqlitePriceHistory

log = logging.getLogger(__name__)

COLUMNS = [
    "search_ts", "provider", "origin", "destination", "departure_date", "return_date", "passengers",
    "airline_code", "airline_name", "flight_number", "dep_at", "arr_at", "stops_out", "stops_return",
    "duration_minutes", "price_amount", "currency", "fx_rate", "price_usd", "signature",
]  # fmt: skip


def export_day(history: SqlitePriceHistory, day: date, out_dir: Path) -> Path | None:
    """Write every observation whose search_ts falls on ``day`` (UTC) to <out_dir>/<day>.csv."""
    with history.connect() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM observations WHERE substr(search_ts, 1, 10) = ? "
            "ORDER BY search_ts, signature",
            conn,
            params=(day.isoformat(),),
        )
    if df.empty:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{day.isoformat()}.csv"
    df[COLUMNS].to_csv(target, index=False)
    log.info("exported %d observations -> %s", len(df), target)
    return target


def _to_obs(r: dict[Any, Any]) -> Observation:
    def opt_dt(v: Any) -> datetime | None:
        return datetime.fromisoformat(str(v)) if isinstance(v, str) and v else None

    def opt_float(v: Any) -> float | None:
        return None if v is None or (isinstance(v, float) and pd.isna(v)) else float(v)

    def opt_str(v: Any) -> str | None:
        return None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)

    duration = opt_float(r.get("duration_minutes"))
    return Observation(
        search_ts=datetime.fromisoformat(str(r["search_ts"])),
        provider=str(r["provider"]),
        origin=str(r["origin"]),
        destination=str(r["destination"]),
        departure_date=date.fromisoformat(str(r["departure_date"])),
        return_date=date.fromisoformat(str(r["return_date"]))
        if opt_str(r.get("return_date"))
        else None,
        passengers=int(r["passengers"]),
        airline_code=opt_str(r.get("airline_code")) or "",
        airline_name=opt_str(r.get("airline_name")),
        flight_number=opt_str(r.get("flight_number")),
        dep_at=opt_dt(r.get("dep_at")),
        arr_at=opt_dt(r.get("arr_at")),
        stops_out=int(r["stops_out"]),
        stops_return=int(r["stops_return"]),
        duration_minutes=int(duration) if duration is not None else None,
        price_amount=float(r["price_amount"]),
        currency=str(r["currency"]),
        fx_rate=opt_float(r.get("fx_rate")),
        price_usd=opt_float(r.get("price_usd")),
        signature=str(r["signature"]),
    )


def import_partitions(history: SqlitePriceHistory, in_dir: Path) -> int:
    """Load every CSV partition into the store; duplicates are ignored, so this is idempotent."""
    total = 0
    for path in sorted(in_dir.glob("*.csv")):
        df = pd.read_csv(path, dtype={"flight_number": str, "airline_code": str})
        rows = [_to_obs(r) for r in df.to_dict("records")]
        n = history.record(rows)
        total += n
        log.info("%s: %d new of %d", path.name, n, len(rows))
    return total
