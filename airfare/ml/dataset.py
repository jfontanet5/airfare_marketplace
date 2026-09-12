"""Build a labeled training set from stored observations.

One row per (itinerary signature, search day). The label answers: did the
price for *this same itinerary* fall by at least DROP_THRESHOLD_PCT within the
next N_DAYS_WINDOW days? Rows whose future window has no observation yet are
right-censored and dropped — they cannot be labeled honestly.

Only information available on the search day enters the features, and the
train/validation split is by search day, so nothing from the future leaks.
"""

from __future__ import annotations

import pandas as pd

from airfare.ml.features import DROP_THRESHOLD_PCT, FEATURES, LABEL, N_DAYS_WINDOW


def daily_prices(observations: pd.DataFrame) -> pd.DataFrame:
    """One row per (signature, search_day): the day's lowest USD price for that itinerary."""
    df = observations.dropna(subset=["price_usd"]).copy()
    if df.empty:
        return df
    df["search_ts"] = pd.to_datetime(df["search_ts"], utc=True, format="ISO8601")
    df["search_day"] = df["search_ts"].dt.floor("D")
    df = df.sort_values(["signature", "search_day", "price_usd"])
    return df.drop_duplicates(["signature", "search_day"], keep="first").reset_index(drop=True)


def _future_min(group: pd.DataFrame) -> pd.Series:
    """For each row, the minimum price observed for the same signature within (day, day+N]."""
    g = group.sort_values("search_day")
    days = g["search_day"].to_numpy()
    prices = g["price_usd"].to_numpy()
    out = []
    for day in days:
        horizon = day + pd.Timedelta(days=N_DAYS_WINDOW)
        mask = (days > day) & (days <= horizon)
        out.append(prices[mask].min() if mask.any() else float("nan"))
    return pd.Series(out, index=g.index)


def build_dataset(observations: pd.DataFrame) -> pd.DataFrame:
    """Columns: FEATURES + search_date + future_min_price_usd + LABEL. Censored rows removed."""
    df = daily_prices(observations)
    if df.empty:
        return pd.DataFrame(columns=[*FEATURES, "search_date", "future_min_price_usd", LABEL])

    df["future_min_price_usd"] = df.groupby("signature", group_keys=False)[
        ["search_day", "price_usd"]
    ].apply(_future_min)
    df = df.dropna(subset=["future_min_price_usd"])

    dep = pd.to_datetime(df["departure_date"])
    df["search_date"] = df["search_day"].dt.date
    df["days_until_departure"] = (dep.dt.tz_localize("UTC") - df["search_day"]).dt.days
    df["current_price_usd"] = df["price_usd"]
    df["total_stops"] = df["stops_out"].fillna(0).astype(int) + df["stops_return"].fillna(0).astype(
        int
    )
    df["departure_dow"] = dep.dt.weekday
    df["search_dow"] = df["search_day"].dt.weekday
    df["airline_code"] = df["airline_code"].fillna("UNK").replace("", "UNK")
    df[LABEL] = (
        df["future_min_price_usd"] <= df["current_price_usd"] * (1 - DROP_THRESHOLD_PCT)
    ).astype(int)

    return df[[*FEATURES, "search_date", "future_min_price_usd", LABEL]].reset_index(drop=True)
