"""Tests for the shared experiment code (data, labels, split and the Student B models).

They run on the committed Kraken snapshot, so they need no network access:
    uv run pytest
"""

import joblib
from mykrml_25740287.models.validation import chronological_split_with_gap
import numpy as np
import pandas as pd
import pytest

from at3_crypto.config import MODELS_DIR
from at3_crypto.dataset import SNAPSHOT_PATH, load_snapshot
from at3_crypto.features import TARGET, build_panel, split_rows
from at3_crypto.modeling.predict import forecast_latest

pytestmark = pytest.mark.skipif(not SNAPSHOT_PATH.exists(), reason="data snapshot not available")
TOKENS = {"BTC", "ETH", "XRP", "SOL", "DOGE", "XLM", "TRX", "XMR"}


@pytest.fixture(scope="module")
def panel():
    return build_panel(load_snapshot())


def test_snapshot_has_complete_daily_history():
    raw = load_snapshot()
    assert set(raw["token"]) == TOKENS
    assert not raw.duplicated(["token", "date"]).any()
    for _, g in raw.groupby("token"):
        assert len(g) == 720
        assert (g["date"].diff().dropna() == pd.Timedelta(days=1)).all()


def test_ohlc_invariants():
    raw = load_snapshot()
    assert (raw["high"] >= raw[["open", "close", "low"]].max(axis=1)).all()
    assert (raw["low"] <= raw[["open", "close"]].min(axis=1)).all()
    assert (raw[["open", "high", "low", "close"]] > 0).all().all()


def test_label_is_mri_seven_days_later(panel):
    for _, g in panel.groupby("token"):
        g = g.set_index("date")
        expected = g["MRI"].reindex(g.index + pd.Timedelta(days=7)).to_numpy()
        assert np.allclose(g[TARGET].to_numpy(), expected, equal_nan=True)


def test_labelled_rows_and_split_match_the_notebook(panel):
    labelled, latest = split_rows(panel)
    assert len(labelled) == 5544
    assert labelled["MRI"].between(0, 100).all()
    assert len(latest) == 8
    train, val, test = chronological_split_with_gap(labelled["date"], val_ratio=0.15, test_ratio=0.15, gap_days=7)
    assert (train.sum(), val.sum(), test.sum()) == (3824, 776, 832)
    # 7-day purge gaps: no validation/test date within 7 days of the previous set
    assert (labelled.loc[val, "date"].min() - labelled.loc[train, "date"].max()).days > 7
    assert (labelled.loc[test, "date"].min() - labelled.loc[val, "date"].max()).days > 7


def test_saved_bundle_forecasts_every_token(panel):
    bundle = joblib.load(MODELS_DIR / "student_b" / "mri_lightgbm.joblib")
    table = forecast_latest(bundle, panel)
    assert set(table["token"]) == TOKENS
    assert table["forecast_t_plus_7"].between(0, 100).all()
