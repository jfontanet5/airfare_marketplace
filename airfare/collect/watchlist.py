"""Watch-list parsing: which routes and horizons the collector snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from airfare.domain.models import SearchQuery


@dataclass(frozen=True, slots=True)
class Route:
    origin: str
    destination: str

    @classmethod
    def parse(cls, text: str) -> Route:
        parts = text.strip().upper().replace("->", "-").split("-")
        if len(parts) != 2 or any(len(p) != 3 for p in parts):
            raise ValueError(f"route must look like SJU-JFK, got {text!r}")
        return cls(parts[0], parts[1])

    def __str__(self) -> str:
        return f"{self.origin}-{self.destination}"


def parse_routes(spec: str) -> list[Route]:
    return [Route.parse(r) for r in spec.split(",") if r.strip()]


def load_watchlist(path: Path) -> list[Route]:
    """One route per line; '#' starts a comment."""
    routes: list[Route] = []
    for line in path.read_text().splitlines():
        text = line.split("#", 1)[0].strip()
        if text:
            routes.append(Route.parse(text))
    return routes


def plan_queries(
    routes: list[Route],
    horizons_days: list[int],
    today: date,
    trip_length_days: int | None,
    max_stops: int = 2,
) -> list[SearchQuery]:
    """Cartesian product of routes x horizons, one query each (one provider request each)."""
    out: list[SearchQuery] = []
    for r in routes:
        for h in horizons_days:
            dep = today + timedelta(days=h)
            ret = dep + timedelta(days=trip_length_days) if trip_length_days else None
            out.append(SearchQuery(r.origin, r.destination, dep, ret, max_stops=max_stops))
    return out
