"""Student C (Trend Category) checks: class boundaries, the t+7 label, the
flat-price projection and the purged date folds. Run: uv run pytest tests/test_student_c.py"""

import numpy as np
import pandas as pd
import pytest

from at3_crypto.modeling import student_c_tc as tc
from at3_crypto.targets import TC_LABELS, classify_tc, trend_score


@pytest.mark.parametrize(("score", "cls"), [(-0.1001, 0), (-0.10, 1), (-0.0201, 1), (-0.02, 2), (0.0, 2),
                                            (0.02, 2), (0.0201, 3), (0.10, 3), (0.1001, 4)])
def test_tc_boundaries(score, cls):
    assert classify_tc(score) == cls


def test_labels_and_nan():
    assert TC_LABELS == ["Strong Downtrend", "Downtrend", "Sideways", "Uptrend", "Strong Uptrend"]
    assert classify_tc(float("nan")) is None
    assert trend_score({"MA5": 110.0, "MA20": 100.0, "Return20": 0.1}) == pytest.approx(0.1)


@pytest.fixture(scope="module")
def frame():
    return tc.build_modelling_frame(tc.load_ohlcv())


def test_label_is_tc_seven_days_later_per_token(frame):
    for _, g in frame.groupby("token"):
        g = g.sort_values("date").reset_index(drop=True)
        known = g["target_TC"].notna()
        np.testing.assert_array_equal(g.loc[known, "target_TC"].values, g["TC"].shift(-7)[known].values)
        assert g["target_TC"].tail(7).isna().all()


def test_projection_uses_only_past_closes(frame):
    g = frame[frame["token"] == "ETH"].sort_values("date").reset_index(drop=True)
    c = g["close"]
    manual = 0.7 * (c / ((c.rolling(13).sum() + 7 * c) / 20) - 1) + 0.3 * (c / c.shift(13) - 1)
    np.testing.assert_allclose(g["TrendScoreProj7"], manual, rtol=1e-10, equal_nan=True)
    # changing a FUTURE close must not change today's projection
    g2 = g.copy(); g2.loc[len(g2) - 1, "close"] *= 2
    assert tc.projected_trend_score(g2).iloc[-2] == pytest.approx(g["TrendScoreProj7"].iloc[-2])


def test_purged_folds_are_ordered_and_gapped(frame):
    data = tc.chronological_split(frame.dropna(subset=[*tc.FEATURE_COLUMNS, "target_TC"]), 7)
    for tr, va in tc.purged_date_folds(data.train["date"]):
        gap = (data.train["date"].iloc[va].min() - data.train["date"].iloc[tr].max()).days
        assert gap > 7 and data.train["token"].iloc[va].nunique() == 8
    assert (data.validation["date"].min() - data.train["date"].max()).days > 7
    assert (data.test["date"].min() - data.validation["date"].max()).days > 7


def test_augmentation_never_masks_own_features(frame):
    data = tc.chronological_split(frame.dropna(subset=[*tc.FEATURE_COLUMNS, "target_TC"]), 7)
    x, y = data.xy("train")
    xa, ya = tc.augment_missing_features(x, y)
    assert len(xa) == 2 * len(x) and xa.iloc[: len(x)].equals(x)
    assert xa["TrendScoreProj7"].notna().all()
