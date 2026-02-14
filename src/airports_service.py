# src/airports_service.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import pandas as pd
import requests
import streamlit as st

# OurAirports "airports.csv" download
OURAIRPORTS_AIRPORTS_CSV_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"


@dataclass(frozen=True)
class AirportsConfig:
    cache_path: Path
    refresh_hours: int = 24


def _project_root() -> Path:
    # src/.. -> project root
    return Path(__file__).resolve().parent.parent


def _default_config() -> AirportsConfig:
    root = _project_root()
    return AirportsConfig(cache_path=root / "data" / "airports_dynamic.csv", refresh_hours=24)


def _download_csv(url: str) -> bytes:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.content


def _is_cache_fresh(path: Path, refresh_hours: int) -> bool:
    if not path.exists():
        return False
    age_seconds = (pd.Timestamp.utcnow(
    ) - pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")).total_seconds()
    return age_seconds < refresh_hours * 3600


def _load_and_prepare_airports(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # OurAirports schema: iata_code, name, municipality, iso_country, type, scheduled_service, etc.
    # Filter: valid IATA + scheduled service if present
    df = df[df["iata_code"].notna()].copy()
    df["iata_code"] = df["iata_code"].astype(str).str.upper().str.strip()
    df = df[df["iata_code"].str.len() == 3].copy()

    # Prefer scheduled_service == "yes" when available (keeps commercial airports)
    if "scheduled_service" in df.columns:
        df["scheduled_service"] = df["scheduled_service"].astype(
            str).str.lower().str.strip()
        df = df[df["scheduled_service"].isin(["yes"])].copy()

    # Basic fields
    name = df["name"].fillna("").astype(str)
    city = df.get("municipality", pd.Series(
        [""] * len(df))).fillna("").astype(str)
    country = df.get("iso_country", pd.Series(
        [""] * len(df))).fillna("").astype(str)

    # Build label for type-to-search
    df["label"] = df["iata_code"] + " — " + \
        city + ", " + country + " (" + name + ")"

    # Keep only what we need + stable ordering
    df = df[["iata_code", "label"]].drop_duplicates().sort_values(
        ["iata_code", "label"]).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def get_airports_df(force_refresh: bool = False) -> pd.DataFrame:
    """
    Returns a dataframe with columns:
      - iata_code
      - label (human-friendly)
    Uses local cached CSV and refreshes periodically.
    """
    cfg = _default_config()
    cfg.cache_path.parent.mkdir(parents=True, exist_ok=True)

    if force_refresh or not _is_cache_fresh(cfg.cache_path, cfg.refresh_hours):
        content = _download_csv(OURAIRPORTS_AIRPORTS_CSV_URL)
        cfg.cache_path.write_bytes(content)

    return _load_and_prepare_airports(cfg.cache_path)


def iata_from_label(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    # label format: "SJU — City, CC (Airport Name)"
    return label.split(" — ", 1)[0].strip().upper()
