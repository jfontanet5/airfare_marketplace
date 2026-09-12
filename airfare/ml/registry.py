"""Model registry: a directory per model with the pickle and a metadata card."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

DEFAULT_MODEL_NAME = "price_drop"


@dataclass(slots=True)
class ModelCard:
    name: str
    trained_at: str
    data_source: str  # "synthetic" | "observations"
    n_train: int
    n_valid: int
    features: list[str]
    label: str
    metrics: dict[str, float]
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_synthetic(self) -> bool:
        return self.data_source == "synthetic"


def save_model(model: Any, card: ModelCard, model_dir: Path) -> Path:
    target = model_dir / card.name
    target.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target / "model.pkl")
    (target / "card.json").write_text(json.dumps(asdict(card), indent=2, default=str))
    return target


def load_model(model_dir: Path, name: str = DEFAULT_MODEL_NAME) -> tuple[Any, ModelCard] | None:
    target = model_dir / name
    if not (target / "model.pkl").exists() or not (target / "card.json").exists():
        return None
    card = ModelCard(**json.loads((target / "card.json").read_text()))
    return joblib.load(target / "model.pkl"), card


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
