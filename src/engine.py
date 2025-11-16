from datetime import date, timedelta
from typing import List, Dict, Any, Optional


def generate_dummy_offers(search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Simulate multi-region, multi-provider offers.

    Respects:
      - trip_structure (Roundtrip vs One-way)
      - optimization_mode (Optimal vs Roundtrip only)
      - max_stops (Nonstop / up to 1 / up to 2+)
      - flexible_dates (±3 days around selected date)
    """
    origin = search_params["origin"]
    destination = search_params["destination"]
    trip_structure = search_params["trip_structure"]
    optimization_mode = search_params["optimization_mode"]
    max_stops_label = search_params["max_stops"]
    flexible_dates = search_params["flexible_dates"]
    departure_date = search_params["departure_date"]          # datetime.date
    # datetime.date or None
    return_date = search_params["return_date"]

    # Map label -> numeric limit
    if "Nonstop" in max_stops_label:
        max_allowed_stops = 0
    elif "1 stop" in max_stops_label:
        max_allowed_stops = 1
    else:
        # "Up to 2+ stops" – treat as 2 for this dummy engine
        max_allowed_stops = 2

    base_price = 300  # just a starting point

    # Determine which day offsets we consider
    if flexible_dates:
        # ±3 days window
        day_offsets = range(-3, 4)  # -3, -2, -1, 0, 1, 2, 3
    else:
        day_offsets = [0]

    offers: List[Dict[str, Any]] = []

    for offset in day_offsets:
        dep_variant = departure_date + timedelta(days=offset)
        ret_variant = return_date + \
            timedelta(days=offset) if (
                return_date and trip_structure == "Roundtrip") else None

        # Simple price adjustment: assume some days are cheaper
        # (purely illustrative; real logic would come from data)
        price_adjustment = offset * 5  # +5 per day away from chosen date
        day_base_price = base_price + price_adjustment

        # Simulated offers seen from different regions/providers for this date
        daily_offers = [
            {
                "option_id": f"US-A-{offset:+d}",
                "region": "US",
                "provider": "ProviderA",
                "description": f"{trip_structure} · Single carrier · US region",
                "total_price_usd": day_base_price,
                "stops_out": 0,
                "stops_return": 0 if trip_structure == "Roundtrip" else None,
                "trip_structure": trip_structure,
                "departure_date": dep_variant.isoformat(),
                "return_date": ret_variant.isoformat() if ret_variant else None,
                "date_offset_days": offset,
            },
            {
                "option_id": f"US-B-{offset:+d}",
                "region": "US",
                "provider": "ProviderB",
                "description": f"{trip_structure} · 1 stop · US region",
                "total_price_usd": day_base_price - 15,
                "stops_out": 1,
                "stops_return": 0 if trip_structure == "Roundtrip" else None,
                "trip_structure": trip_structure,
                "departure_date": dep_variant.isoformat(),
                "return_date": ret_variant.isoformat() if ret_variant else None,
                "date_offset_days": offset,
            },
            {
                "option_id": f"EU-A-{offset:+d}",
                "region": "EU",
                "provider": "ProviderA",
                "description": f"{trip_structure} · Mixed carriers · EU region",
                "total_price_usd": day_base_price - 35,
                "stops_out": 1,
                "stops_return": 1 if trip_structure == "Roundtrip" else None,
                "trip_structure": trip_structure,
                "departure_date": dep_variant.isoformat(),
                "return_date": ret_variant.isoformat() if ret_variant else None,
                "date_offset_days": offset,
            },
            {
                "option_id": f"LATAM-A-{offset:+d}",
                "region": "LATAM",
                "provider": "ProviderC",
                "description": f"{trip_structure} · Aggressive pricing · LATAM region",
                "total_price_usd": day_base_price - 45,
                "stops_out": 2,
                "stops_return": 2 if trip_structure == "Roundtrip" else None,
                "trip_structure": trip_structure,
                "departure_date": dep_variant.isoformat(),
                "return_date": ret_variant.isoformat() if ret_variant else None,
                "date_offset_days": offset,
            },
        ]

        offers.extend(daily_offers)

    # --- Apply optimization_mode behavior ---

    if optimization_mode == "Traditional" and trip_structure == "Roundtrip":
        offers = [o for o in offers if o["trip_structure"] == "Roundtrip"]

    # --- Apply max_stops filter ---

    def passes_stops_filter(offer: Dict[str, Any]) -> bool:
        out_stops = offer.get("stops_out") or 0
        ret_stops = offer.get("stops_return")
        if trip_structure == "One-way":
            # Only outbound segment matters
            return out_stops <= max_allowed_stops
        else:
            # Both segments should respect the max stops
            ret_stops_val = ret_stops or 0
            return out_stops <= max_allowed_stops and ret_stops_val <= max_allowed_stops

    offers = [o for o in offers if passes_stops_filter(o)]

    return offers


def score_offer(offer: Dict[str, Any], search_params: Dict[str, Any]) -> float:
    """
    Compute a score for an offer that balances price, stops, date offset,
    and region. Lower score = better.

    This is a rule-based heuristic that we can later replace or augment
    with an ML model.
    """
    price = float(offer.get("total_price_usd", 1e9))

    # Stops penalty
    stops_out = offer.get("stops_out") or 0
    stops_ret = offer.get("stops_return") or 0

    if search_params["trip_structure"] == "One-way":
        total_stops = stops_out
    else:
        total_stops = stops_out + stops_ret

    stops_penalty = total_stops * 35  # penalty per stop

    # Date offset penalty (how far from requested departure)
    requested_dep: date = search_params["departure_date"]

    # Dummy engine includes 'date_offset_days'
    if "date_offset_days" in offer:
        offset_days = abs(int(offer["date_offset_days"]))
    else:
        # CSV engine: compute from departure_date in offer
        dep_str = offer.get("departure_date")
        offset_days = 0
        if dep_str:
            try:
                dep_date = date.fromisoformat(str(dep_str))
                offset_days = abs((dep_date - requested_dep).days)
            except Exception:
                offset_days = 0

    date_penalty = offset_days * 5  # small penalty per day away

    # Region penalty (soft preference for US region for US-origin users)
    region = str(offer.get("region", "")).upper()
    region_penalty = 0
    if region not in ("US", ""):
        region_penalty = 15  # small bump for non-US region

    # Combine: base price plus penalties
    score = price + stops_penalty + date_penalty + region_penalty
    return score


def pick_best_offers(offers: List[Dict[str, Any]]):
    """
    From a list of offers, return:
      - best_global: cheapest overall
      - best_us: cheapest with region == 'US' (if any)
    """
    if not offers:
        return None, None

    best_global = min(offers, key=lambda o: o["total_price_usd"])
    us_offers = [o for o in offers if o["region"] == "US"]
    best_us = min(
        us_offers, key=lambda o: o["total_price_usd"]) if us_offers else None

    return best_global, best_us


def pick_recommended_offer(offers: List[Dict[str, Any]], search_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Pick the single recommended offer based on the heuristic score.
    """
    if not offers:
        return None

    for o in offers:
        o["score"] = score_offer(o, search_params)

    # Lower score is better
    return min(offers, key=lambda o: o["score"])


def format_offer_label(offer: Dict[str, Any]) -> str:
    """
    Build a human-readable label for an itinerary, whether it comes from
    dummy data or from a real CSV (which may not have 'description').
    """
    # If there is already a description, use it
    desc = offer.get("description")
    region = offer.get("region", "N/A")

    if desc:
        return f"{desc} · Region: {region}"

    # Otherwise build something from available fields
    parts = []

    trip_structure = offer.get("trip_structure")
    if trip_structure:
        parts.append(trip_structure)

    airline = offer.get("airline")
    if airline:
        parts.append(str(airline))

    origin = offer.get("origin")
    destination = offer.get("destination")
    if origin and destination:
        parts.append(f"{origin} → {destination}")

    if not parts:
        parts.append("Itinerary")

    return f"{' · '.join(parts)} · Region: {region}"
