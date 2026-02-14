# src/services/offer_bridge.py

from typing import List, Dict, Any
from core.models import Offer


def offers_to_legacy_dicts(offers_objects: List[Offer]) -> List[Dict[str, Any]]:
    offers: List[Dict[str, Any]] = []

    for o in offers_objects:
        raw = dict(o.raw or {})

        # Start from raw (for legacy keys like region), then override with normalized dataclass fields
        d: Dict[str, Any] = {}
        d.update(raw)
        d.update(o.__dict__)

        # Remove big / noisy fields so the UI doesn't dump huge nested JSON
        d.pop("raw", None)
        # common large Amadeus field (if present)
        d.pop("travelerPricings", None)
        d.pop("price", None)              # nested object
        d.pop("itineraries", None)        # nested object
        d.pop("validatingAirlineCodes", None)

        # Ensure region exists for legacy scoring
        d.setdefault("region", "US")

        offers.append(d)

    return offers
