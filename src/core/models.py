# src/core/models.py

from dataclasses import dataclass
from datetime import date
from typing import Optional, List, Dict, Any


@dataclass(frozen=True)
class SearchParams:
    origin: str
    destination: str
    trip_structure: str
    departure_date: date
    return_date: Optional[date]
    optimization_mode: str
    passengers: int
    max_stops: str
    airlines: List[str]
    multicity: bool
    flexible_dates: bool
    use_real_data: bool


@dataclass
class Offer:
    provider: str
    origin: str
    destination: str
    trip_structure: str
    departure_date: date
    return_date: Optional[date]
    airline: str
    stops_out: int
    stops_return: int
    total_price_usd: float
    currency: str = "USD"
    score: float = 0.0
    raw: Optional[Dict[str, Any]] = None
