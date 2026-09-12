import json
from datetime import date

import pytest
from airfare.providers.amadeus.parser import parse_iso_duration_minutes, parse_response

from tests.conftest import FIXTURES


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("PT6H30M", 390),
        ("PT45M", 45),
        ("PT2H", 120),
        ("P1DT2H", 1560),
        ("garbage", None),
        (None, None),
    ],
)
def test_iso_duration(raw: str | None, expected: int | None) -> None:
    assert parse_iso_duration_minutes(raw) == expected


def test_parse_recorded_response() -> None:
    payload = json.loads((FIXTURES / "amadeus_flight_offers.json").read_text())
    offers = parse_response(payload, "SJU", "JFK")
    assert len(offers) == 2
    o = offers[0]
    assert o.price.amount == 251.30
    assert o.price.currency == "EUR"
    assert o.price.usd is None  # not normalized yet
    assert o.airline_code == "B6"
    assert o.airline_name == "JETBLUE AIRWAYS"
    assert o.departure_date == date(2026, 10, 1)
    assert o.return_date == date(2026, 10, 5)
    assert o.stops_out == 0 and o.stops_return == 1
    assert o.outbound is not None and o.outbound.duration_minutes == 225
    assert o.inbound is not None and len(o.inbound.layovers()) == 1
    assert offers[1].airline_code == "AA"
