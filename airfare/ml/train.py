"""Train the price-drop classifier.

Time-based split (train on earlier search dates, validate on later ones),
probability calibration, and a model card with honest metrics. Currently the
only data source is synthetic; the collector + dataset builder will plug in
real observations here without changing the pipeline.
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
from airfare.ml.features import CATEGORICAL, FEATURES, LABEL, NUMERIC
from airfare.ml.registry import DEFAULT_MODEL_NAME, ModelCard, now_iso, save_model
from airfare.ml.synthetic import make_synthetic_training_data

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


def train_synthetic(model_dir: Path, n_rows: int = 5000) -> ModelCard:
    df = make_synthetic_training_data(n_rows)
    train, valid = time_split(df)
    model = build_pipeline().fit(train[FEATURES], train[LABEL])
    metrics = evaluate(model, valid)
    card = ModelCard(
        name=DEFAULT_MODEL_NAME,
        trained_at=now_iso(),
        data_source="synthetic",
        n_train=len(train),
        n_valid=len(valid),
        features=FEATURES,
        label=LABEL,
        metrics=metrics,
        notes=(
            "Demo model trained on a documented synthetic generating process "
            "(airfare.ml.synthetic). Metrics reflect that process, not real markets."
        ),
    )
    path = save_model(model, card, model_dir)
    log.info("saved %s -> %s metrics=%s", card.name, path, metrics)
    return card


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the price-drop model")
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--model-dir", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    card = train_synthetic(args.model_dir or get_settings().airfare_model_dir, args.rows)
    print(f"trained {card.name} ({card.data_source}); validation metrics: {card.metrics}")


if __name__ == "__main__":
    main()
