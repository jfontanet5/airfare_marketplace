"""Airport lookup for the search form.

A small curated list of commercial airports ships with the repo so offline
mode never touches the network. When live mode is on, the full OurAirports
dataset is downloaded once a day into a local cache and merged in.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

OURAIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"


@dataclass(frozen=True, slots=True)
class Airport:
    iata: str
    name: str
    city: str
    country: str

    @property
    def label(self) -> str:
        return f"{self.iata} — {self.city}, {self.country} ({self.name})"

    def matches(self, needle: str) -> bool:
        n = needle.lower()
        return n in self.iata.lower() or n in self.city.lower() or n in self.name.lower()


def _from_frame(df: pd.DataFrame) -> list[Airport]:
    df = df[df["iata_code"].notna()].copy()
    df["iata_code"] = df["iata_code"].astype(str).str.upper().str.strip()
    df = df[df["iata_code"].str.len() == 3]
    if "scheduled_service" in df.columns:
        df = df[df["scheduled_service"].astype(str).str.lower() == "yes"]
    out = [
        Airport(
            iata=str(r["iata_code"]),
            name=str(r.get("name", "") or ""),
            city=str(r.get("municipality", "") or ""),
            country=str(r.get("iso_country", "") or ""),
        )
        for r in df.to_dict("records")
    ]
    return sorted({a.iata: a for a in out}.values(), key=lambda a: a.iata)


def load_bundled(sample_dir: Path) -> list[Airport]:
    return _from_frame(pd.read_csv(sample_dir / "airports.csv"))


def load_full(cache_path: Path, refresh_hours: int = 24, timeout: float = 20.0) -> list[Airport]:
    stale = (
        not cache_path.exists() or (time.time() - cache_path.stat().st_mtime) > refresh_hours * 3600
    )
    if stale:
        log.info("downloading airports dataset to %s", cache_path)
        resp = requests.get(OURAIRPORTS_URL, timeout=timeout)
        resp.raise_for_status()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(resp.content)
    return _from_frame(pd.read_csv(cache_path, low_memory=False))


def load_airports(sample_dir: Path, cache_path: Path, include_full: bool) -> list[Airport]:
    bundled = load_bundled(sample_dir)
    if not include_full:
        return bundled
    try:
        full = load_full(cache_path)
    except Exception as e:
        log.warning("full airport list unavailable (%s); using bundled list", e)
        return bundled
    merged = {a.iata: a for a in full}
    merged.update({a.iata: a for a in bundled})
    return sorted(merged.values(), key=lambda a: a.iata)
