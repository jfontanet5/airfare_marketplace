"""Model page: what the price-drop signal is, how it was trained, and how much real data exists."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from airfare.ml.dataset import build_dataset
from airfare.ml.features import DROP_THRESHOLD_PCT, FEATURES, N_DAYS_WINDOW
from airfare.ml.train import MIN_OBSERVATION_ROWS
from airfare.ui import bootstrap
from airfare.ui import components as ui


@st.cache_data(show_spinner=False, ttl=300)
def _readiness() -> tuple[int, int]:
    history = bootstrap.history()
    with history.connect() as conn:
        obs = pd.read_sql_query("SELECT * FROM observations", conn)
    labeled = build_dataset(obs) if not obs.empty else pd.DataFrame()
    return len(obs), len(labeled)


def render() -> None:
    ui.page_header(
        "Price-drop model",
        "A calibrated probability, with its training data and metrics on the table.",
    )
    predictor = bootstrap.predictor()

    st.subheader("What it predicts")
    st.markdown(
        f"For a fare seen today, the probability that **the same itinerary** (same flights, same "
        f"departure) will be at least **{DROP_THRESHOLD_PCT:.0%} cheaper** at some point in the next "
        f"**{N_DAYS_WINDOW} days**. Labels come from observing each itinerary repeatedly; rows without a "
        f"future observation are dropped rather than guessed. Training uses a time-based split (earlier "
        f"search days train, later ones validate) and isotonic calibration, so 30% means roughly 3 in 10."
    )

    st.subheader("Current model")
    if predictor is None:
        st.warning(
            "No model loaded. Run `make train` for the synthetic demo or `make train-real` once enough history exists."
        )
    else:
        card = predictor.card
        badge = (
            ui.chip("synthetic demo", "demo")
            if card.is_synthetic
            else ui.chip("trained on observations", "live")
        )
        st.markdown(
            f"**{card.name}** &nbsp; {badge} &nbsp; <span class='afm-muted'>trained {card.trained_at}</span>",
            unsafe_allow_html=True,
        )
        m = card.metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC-AUC", f"{m.get('roc_auc', float('nan')):.3f}")
        c2.metric("PR-AUC", f"{m.get('pr_auc', float('nan')):.3f}")
        c3.metric(
            "Brier score",
            f"{m.get('brier', float('nan')):.3f}",
            help="Mean squared error of the probabilities; lower is better, 0.25 is a coin flip",
        )
        c4.metric("Validation rows", f"{card.n_valid:,}", help=f"{card.n_train:,} training rows")
        if card.is_synthetic:
            st.info(
                "These metrics describe the documented synthetic generating process in "
                "`airfare/ml/synthetic.py`, not real markets. The badge disappears once the model is "
                "retrained on collected observations."
            )
        else:
            st.caption(
                f"Observation window: {card.extra.get('search_date_min')} → {card.extra.get('search_date_max')}"
            )
        with st.expander("Features and notes"):
            st.markdown("  \n".join(f"· `{f}`" for f in card.features or FEATURES))
            st.caption(card.notes)

    st.subheader("Real-data readiness")
    n_obs, n_labeled = _readiness()
    progress = min(1.0, n_labeled / MIN_OBSERVATION_ROWS) if MIN_OBSERVATION_ROWS else 0.0
    st.progress(
        progress,
        text=f"{n_labeled:,} labeled rows of {MIN_OBSERVATION_ROWS} needed · {n_obs:,} raw observations",
    )
    if n_labeled >= MIN_OBSERVATION_ROWS:
        st.success("Enough labeled history to train: run `make train-real`.")
    else:
        st.caption(
            f"A row becomes labeled only after the same itinerary is observed again within {N_DAYS_WINDOW} days, "
            "so the count lags collection by about a week. The daily collector adds roughly 90 observations per run."
        )
