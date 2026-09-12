from dataclasses import replace
from datetime import date
from pathlib import Path

from airfare.domain.models import Price
from airfare.ml.features import FEATURES, feature_row
from airfare.ml.predict import PriceDropPredictor, price_position
from airfare.ml.synthetic import make_synthetic_training_data
from airfare.ml.train import train_synthetic

from tests.conftest import make_offer


def test_feature_row_uses_codes_and_usd() -> None:
    o = replace(make_offer(price=200, currency="EUR"), price=Price(200, "EUR", usd=220.0))
    row = feature_row(o, date(2026, 9, 1))
    assert row["airline_code"] == "B6" and row["current_price_usd"] == 220.0
    assert row["days_until_departure"] == 30
    assert set(row) == set(FEATURES)


def test_synthetic_data_has_signal() -> None:
    df = make_synthetic_training_data(2000, today=date(2026, 9, 1))
    assert 0.2 < df["price_drops"].mean() < 0.8
    assert df.groupby(df["days_until_departure"] > 60)["price_drops"].mean().is_monotonic_increasing


def test_train_and_predict_roundtrip(tmp_path: Path) -> None:
    card = train_synthetic(tmp_path, n_rows=1500)
    assert card.is_synthetic and card.metrics["roc_auc"] > 0.6
    predictor = PriceDropPredictor.load(tmp_path)
    assert predictor is not None
    o = replace(make_offer(price=400), price=Price(400, "USD", usd=400.0))
    p = predictor.probability(o, date(2026, 9, 1))
    assert 0.0 <= p <= 1.0


def test_price_position() -> None:
    stats = {"count": 20, "p25": 200.0, "p75": 300.0}
    assert price_position(150, stats) == "low"
    assert price_position(250, stats) == "typical"
    assert price_position(350, stats) == "high"
    assert price_position(250, {"count": 2}) is None
