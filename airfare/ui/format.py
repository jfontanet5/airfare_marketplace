"""Small pure formatting helpers for the UI."""

from __future__ import annotations

from datetime import datetime, timedelta

from airfare.domain.models import Itinerary, Offer


def usd(value: float | None) -> str:
    return "—" if value is None else f"${value:,.0f}"


def money(offer: Offer) -> str:
    p = offer.price
    if p.usd is None:
        return f"{p.amount:,.0f} {p.currency}"
    return usd(p.usd)


def clock(dt: datetime | None) -> str:
    return dt.strftime("%-I:%M %p") if dt else "—"


def hours(minutes: int | None) -> str:
    if minutes is None:
        return "—"
    h, m = divmod(minutes, 60)
    return f"{h}h {m:02d}m"


def delta(td: timedelta) -> str:
    return hours(int(td.total_seconds() // 60))


def stops_label(n: int) -> str:
    return "Nonstop" if n == 0 else f"{n} stop{'s' if n > 1 else ''}"


def route_line(it: Itinerary) -> str:
    """'SJU → MIA → JFK'."""
    if not it.segments:
        return "—"
    return " → ".join([it.segments[0].origin, *(s.destination for s in it.segments)])
