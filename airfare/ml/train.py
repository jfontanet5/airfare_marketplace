"""Train the price-drop classifier.

Time-based split (train on earlier search dates, validate on later ones),
probability calibration, and a model card with honest metrics. Two data
sources share the pipeline: the documented synthetic process (demo) and real
observations collected by ``airfare-collect`` (see ``airfare.ml.dataset``).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from airfare.config import get_settings
from airfare.ml.dataset import build_dataset
from airfare.ml.features import CATEGORICAL, FEATURES, LABEL, NUMERIC
from airfare.ml.registry import DEFAULT_MODEL_NAME, ModelCard, now_iso, save_model
from airfare.ml.synthetic import make_synthetic_training_data
from airfare.storage.sqlite import SqlitePriceHistory

log = logging.getLogger(__name__)


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ]
    )
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=42
    )
    return Pipeline([("pre", pre), ("clf", CalibratedClassifierCV(clf, method="isotonic", cv=3))])


def time_split(df: pd.DataFrame, valid_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("search_date")
    cut = df["search_date"].iloc[int(len(df) * (1 - valid_fraction))]
    return df[df["search_date"] < cut], df[df["search_date"] >= cut]


def evaluate(model: Any, valid: pd.DataFrame) -> dict[str, float]:
    proba = model.predict_proba(valid[FEATURES])[:, 1]
    y = valid[LABEL].to_numpy()
    return {
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "pr_auc": round(float(average_precision_score(y, proba)), 4),
        "brier": round(float(brier_score_loss(y, proba)), 4),
        "positive_rate": round(float(np.mean(y)), 4),
    }


MIN_OBSERVATION_ROWS = 300


def _fit_and_save(df: pd.DataFrame, model_dir: Path, data_source: str, notes: str) -> ModelCard:
    train, valid = time_split(df)
    if train[LABEL].nunique() < 2 or valid[LABEL].nunique() < 2:
        raise ValueError("both classes must be present in train and validation splits")
    model = build_pipeline().fit(train[FEATURES], train[LABEL])
    metrics = evaluate(model, valid)
    card = ModelCard(
        name=DEFAULT_MODEL_NAME,
        trained_at=now_iso(),
        data_source=data_source,
        n_train=len(train),
        n_valid=len(valid),
        features=FEATURES,
        label=LABEL,
        metrics=metrics,
        notes=notes,
        extra={
            "search_date_min": str(df["search_date"].min()),
            "search_date_max": str(df["search_date"].max()),
        },
    )
    path = save_model(model, card, model_dir)
    log.info("saved %s -> %s metrics=%s", card.name, path, metrics)
    return card


def train_synthetic(model_dir: Path, n_rows: int = 5000) -> ModelCard:
    return _fit_and_save(
        make_synthetic_training_data(n_rows),
        model_dir,
        "synthetic",
        "Demo model trained on a documented synthetic generating process "
        "(airfare.ml.synthetic). Metrics reflect that process, not real markets.",
    )


def train_from_observations(model_dir: Path, history: SqlitePriceHistory) -> ModelCard:
    with history.connect() as conn:
        observations = pd.read_sql_query("SELECT * FROM observations", conn)
    df = build_dataset(observations)
    if len(df) < MIN_OBSERVATION_ROWS:
        raise ValueError(
            f"only {len(df)} labeled rows (need {MIN_OBSERVATION_ROWS}); "
            "keep the collector running and retry"
        )
    return _fit_and_save(
        df,
        model_dir,
        "observations",
        f"Trained on {len(df)} labeled (signature, search-day) rows from collected observations; "
        f"label = ≥5% drop within {7} days for the same itinerary.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the price-drop model")
    parser.add_argument("--source", choices=["synthetic", "observations"], default="synthetic")
    parser.add_argument("--rows", type=int, default=5000, help="synthetic rows")
    parser.add_argument("--model-dir", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    model_dir = args.model_dir or settings.airfare_model_dir
    if args.source == "observations":
        card = train_from_observations(model_dir, SqlitePriceHistory(settings.airfare_db_path))
    else:
        card = train_synthetic(model_dir, args.rows)
    print(f"trained {card.name} ({card.data_source}); validation metrics: {card.metrics}")


if __name__ == "__main__":
    main()
