"""Tests for Student D's TOR features and target.

CONTRACT.md section 5 requires the classification boundary values to be unit
tested, because the bands mix inclusive and exclusive bounds. The feature tests
guard the exact JSON keys other services consume: renaming `VolumeSMA20` or
`RelativeVolume` silently breaks Student A's model, and nothing else in the
pipeline would fail.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from at3_crypto.dataset import SNAPSHOT_PATH
from at3_crypto.engineered_features import (
    FEATURE_OWNERS,
    LONG_WINDOW,
    OWNED_BY_STUDENT_D,
    engineer_features,
)
from at3_crypto.kraken import SUPPORTED_TOKENS, KrakenClient
from at3_crypto.modeling.student_d_tor import (
    PEER_FEATURE_GROUPS,
    _score,
    augment_missing_features,
    load_dataset,
    load_ohlcv,
    predict_proba_all_labels,
    purged_date_folds,
)
from at3_crypto.targets import (
    FORECAST_HORIZON,
    TOR_LABELS,
    build_tor_target,
    classify_tor,
    opportunity_score,
)


def make_ohlcv(token: str = "BTC", days: int = 200, seed: int = 7) -> pd.DataFrame:
    """Deterministic synthetic OHLCV, so the suite runs offline."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.03, days)))
    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=days, freq="D", tz="UTC"),
        "token": token,
        "open": close * (1 + rng.normal(0, 0.005, days)),
        "high": close * (1 + abs(rng.normal(0, 0.012, days))),
        "low": close * (1 - abs(rng.normal(0, 0.012, days))),
        "close": close,
        "volume": abs(rng.normal(1_000, 250, days)) + 10,
    })


@pytest.fixture
def engineered() -> pd.DataFrame:
    return engineer_features(make_ohlcv())


# --- Contract section 1: tokens -----------------------------------------

def test_supported_tokens_match_the_contract():
    assert set(SUPPORTED_TOKENS) == {
        "BTC", "ETH", "XRP", "SOL", "DOGE", "XLM", "TRX", "XMR"
    }


# --- Contract section 4: feature names and formulas ---------------------

def test_student_d_owns_the_contract_feature_names():
    """The exact JSON keys Student A consumes from POST /features."""
    assert OWNED_BY_STUDENT_D == ["VolumeSMA20", "RelativeVolume"]
    assert FEATURE_OWNERS["VolumeSMA20"] == "student_d"
    assert FEATURE_OWNERS["RelativeVolume"] == "student_d"


def test_owned_feature_formulas(engineered):
    row = 150
    window = engineered["volume"].iloc[row - LONG_WINDOW + 1 : row + 1]

    assert engineered["VolumeSMA20"].iloc[row] == pytest.approx(window.mean())
    assert engineered["RelativeVolume"].iloc[row] == pytest.approx(
        engineered["volume"].iloc[row] / window.mean()
    )


def test_peer_feature_formulas_match_the_contract(engineered):
    """Local fallbacks must equal the owners' definitions exactly."""
    row = 150
    close = engineered["close"]

    assert engineered["Return20"].iloc[row] == pytest.approx(
        (close.iloc[row] - close.iloc[row - 20]) / close.iloc[row - 20]
    )
    assert engineered["MA5"].iloc[row] == pytest.approx(close.iloc[row - 4 : row + 1].mean())
    assert engineered["MA20"].iloc[row] == pytest.approx(close.iloc[row - 19 : row + 1].mean())
    # ddof=1, per the contract's stated assumption.
    assert engineered["Volatility20"].iloc[row] == pytest.approx(
        engineered["DailyReturn"].iloc[row - 19 : row + 1].std(ddof=1)
    )


def test_rolling_windows_never_span_two_tokens():
    pooled = pd.concat([make_ohlcv("BTC", seed=1), make_ohlcv("ETH", seed=2)])
    result = engineer_features(pooled)

    for token in ("BTC", "ETH"):
        subset = result[result["token"] == token].reset_index(drop=True)
        assert subset["MA20"].iloc[: LONG_WINDOW - 1].isna().all()
        assert subset["MA20"].iloc[LONG_WINDOW - 1 :].notna().all()

    alone = engineer_features(make_ohlcv("BTC", seed=1))
    together = result[result["token"] == "BTC"].reset_index(drop=True)
    pd.testing.assert_series_equal(alone["VolumeSMA20"], together["VolumeSMA20"])


def test_relative_volume_is_nan_not_inf_on_zero_volume():
    """An infinity would survive dropna() and reach the model."""
    frame = make_ohlcv(days=40)
    frame.loc[:, "volume"] = 0.0
    result = engineer_features(frame)

    assert result["RelativeVolume"].isna().all()
    assert not np.isinf(result["RelativeVolume"].dropna()).any()


# --- Contract section 3: the incomplete candle --------------------------

def test_in_progress_candle_is_dropped():
    """Kraken serves today's part-day candle; its volume is not a full day."""
    today = pd.Timestamp.now(tz="UTC").normalize()
    candles = [
        [int((today - pd.Timedelta(offset, unit="D")).timestamp()),
         "100", "101", "99", "100", "100", "1000", 10]
        for offset in range(4, -1, -1)
    ]

    class FakeResponse:
        status_code = 200
        def raise_for_status(self): ...
        def json(self): return {"error": [], "result": {"XXBTZUSD": candles, "last": 0}}

    class FakeSession:
        def get(self, *args, **kwargs): return FakeResponse()

    client = KrakenClient(session=FakeSession())

    assert client.fetch_ohlcv("BTC")["date"].max() < today
    assert client.fetch_ohlcv("BTC", drop_incomplete=False)["date"].max() == today


# --- Contract section 5: Opportunity Score and boundaries ---------------

def test_opportunity_score_matches_the_contract():
    features = {
        "Return20": 0.10,       # M = 0.75
        "MA5": 110.0, "MA20": 100.0,   # T = 1.10
        "RelativeVolume": 1.0,  # V = 0.5
        "Volatility20": 0.05,   # R = 0.5
    }
    expected = 0.40 * 0.75 + 0.30 * 1.10 + 0.20 * 0.5 - 0.10 * 0.5
    assert opportunity_score(features) == pytest.approx(expected)


def test_opportunity_score_clips_v_and_r_but_not_m_and_t():
    saturated = opportunity_score(
        {"Return20": 0.0, "MA5": 1.0, "MA20": 1.0, "RelativeVolume": 50.0, "Volatility20": 5.0}
    )
    at_cap = opportunity_score(
        {"Return20": 0.0, "MA5": 1.0, "MA20": 1.0, "RelativeVolume": 2.0, "Volatility20": 0.10}
    )
    assert saturated == pytest.approx(at_cap)

    # M is deliberately unclipped, so OS can exceed 1.
    assert opportunity_score(
        {"Return20": 5.0, "MA5": 1.0, "MA20": 1.0, "RelativeVolume": 1.0, "Volatility20": 0.0}
    ) > 1.0


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (-1.0, "Strong Sell"), (0.0, "Strong Sell"), (0.1999999, "Strong Sell"),
        (0.20, "Sell"), (0.30, "Sell"), (0.3999999, "Sell"),
        (0.40, "Hold"), (0.50, "Hold"), (0.5999999, "Hold"),
        (0.60, "Buy"), (0.70, "Buy"), (0.7999999, "Buy"),
        (0.80, "Strong Buy"), (1.50, "Strong Buy"),
    ],
)
def test_classification_boundaries(score, expected):
    """Every boundary in CONTRACT.md section 5, inclusive side included."""
    assert TOR_LABELS[classify_tor(score)] == expected


def test_classify_tor_propagates_nan():
    result = classify_tor(pd.Series([0.5, np.nan, 0.9]))
    assert result.iloc[0] == 2
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == 4


# --- Contract section 5: the t+7 shift ----------------------------------

def test_target_looks_exactly_seven_days_ahead():
    pooled = pd.concat([make_ohlcv("BTC", seed=1), make_ohlcv("ETH", seed=2)])
    data = build_tor_target(engineer_features(pooled), horizon=7)

    for token in ("BTC", "ETH"):
        subset = data[data["token"] == token].reset_index(drop=True)
        pd.testing.assert_series_equal(
            subset["target_TOR"], subset["TOR"].shift(-7), check_names=False
        )
        assert subset["target_TOR"].tail(7).isna().all()


def test_target_is_not_recoverable_from_todays_features():
    data = build_tor_target(engineer_features(make_ohlcv()), horizon=7).dropna(
        subset=["TOR", "target_TOR"]
    )
    assert (data["TOR"] == data["target_TOR"]).mean() < 0.95


# --- Contract section 8: the cross-validation scheme ---------------------

@pytest.fixture(scope="module")
def dataset():
    """The real split, built from the committed snapshot (no network)."""
    if not SNAPSHOT_PATH.exists():
        pytest.skip("data snapshot not available")
    data, _ = load_dataset()
    return data


def test_chronological_split_purges_the_forecast_horizon(dataset):
    assert dataset.train["date"].max() < dataset.validation["date"].min()
    assert dataset.validation["date"].max() < dataset.test["date"].min()

    to_validation = (dataset.validation["date"].min() - dataset.train["date"].max()).days
    to_test = (dataset.test["date"].min() - dataset.validation["date"].max()).days
    assert to_validation > FORECAST_HORIZON
    assert to_test > FORECAST_HORIZON


def test_cv_folds_are_ordered_in_time_with_a_seven_day_purge(dataset):
    """The contract's gap must purge seven DAYS, not seven rows.

    `TimeSeriesSplit` cuts on row positions. The modelling frame is a panel of
    eight tokens per date, so handing it rows would split it by token - every
    fold's validation set would span its training set's own dates, and `gap=7`
    would purge less than one calendar day.
    """
    dates = dataset.train["date"]
    folds = list(purged_date_folds(dates))
    assert len(folds) == 5

    for fold_train, fold_valid in folds:
        train_dates, valid_dates = dates.iloc[fold_train], dates.iloc[fold_valid]
        assert valid_dates.min() > train_dates.max()
        assert (valid_dates.min() - train_dates.max()).days > FORECAST_HORIZON
        # No date may appear on both sides of a fold.
        assert not set(train_dates) & set(valid_dates)


def test_cv_folds_keep_every_token_on_both_sides(dataset):
    """A fold must measure forecasting, not generalisation to an unseen coin."""
    tokens = dataset.train["token"]
    for fold_train, fold_valid in purged_date_folds(dataset.train["date"]):
        assert set(tokens.iloc[fold_train]) == set(SUPPORTED_TOKENS)
        assert set(tokens.iloc[fold_valid]) == set(SUPPORTED_TOKENS)


def test_row_order_cannot_affect_the_folds(dataset):
    """Shuffling the frame must not change which dates land in which fold."""
    shuffled = dataset.train.sample(frac=1.0, random_state=0)
    original = [
        (sorted(set(dataset.train["date"].iloc[a])), sorted(set(dataset.train["date"].iloc[b])))
        for a, b in purged_date_folds(dataset.train["date"])
    ]
    reordered = [
        (sorted(set(shuffled["date"].iloc[a])), sorted(set(shuffled["date"].iloc[b])))
        for a, b in purged_date_folds(shuffled["date"])
    ]
    assert original == reordered


# --- Contract section 7: missing-feature augmentation -------------------

def test_augmentation_doubles_the_rows_and_spares_the_owned_feature(dataset):
    x, y = dataset.xy("train")
    x_aug, y_aug = augment_missing_features(x, y)

    assert len(x_aug) == 2 * len(x)
    assert len(y_aug) == 2 * len(y)
    # The original rows are untouched; only the appended copy carries masks.
    pd.testing.assert_frame_equal(x_aug.iloc[: len(x)], x.reset_index(drop=True))

    copy = x_aug.iloc[len(x) :]
    # Student D owns RelativeVolume, so it is never unavailable at inference.
    assert copy["RelativeVolume"].notna().all()
    # Every peer group must actually be exercised.
    for columns in PEER_FEATURE_GROUPS.values():
        assert copy[columns].isna().any().any()


def test_augmentation_is_reproducible(dataset):
    x, y = dataset.xy("train")
    first, _ = augment_missing_features(x, y, seed=7)
    second, _ = augment_missing_features(x, y, seed=7)
    pd.testing.assert_frame_equal(first, second)


# --- Contract section 8: comparable metrics -----------------------------

def test_scores_average_over_all_five_labels():
    """Averaging only the observed labels would make the figures incomparable."""
    perfect_on_one_class = _score([0, 0, 0], [0, 0, 0])
    assert perfect_on_one_class["accuracy"] == 1.0
    # One of five labels right, four absent and scored 0 -> 1/5.
    assert perfect_on_one_class["macro_f1"] == pytest.approx(0.2)


def test_probabilities_are_widened_to_every_label():
    """log-loss over five labels needs five columns, even if a class is unseen."""
    x = pd.DataFrame({"f": np.arange(40, dtype=float)})
    y = pd.Series([2] * 20 + [3] * 20)          # only two of the five ratings
    model = lgb.LGBMClassifier(n_estimators=5, verbose=-1).fit(x, y)

    probabilities = predict_proba_all_labels(model, x)
    assert probabilities.shape == (40, len(TOR_LABELS))
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    # The ratings the model never saw carry essentially no mass.
    assert probabilities[:, [0, 1, 4]].max() < 1e-9


# --- Reproducibility ----------------------------------------------------

def test_snapshot_is_the_default_source():
    """Live fetches move Kraken's 720-candle window and break reproducibility."""
    if not SNAPSHOT_PATH.exists():
        pytest.skip("data snapshot not available")
    raw = load_ohlcv()
    assert list(raw.columns) == ["date", "token", "open", "high", "low", "close", "volume"]
    assert str(raw["date"].dt.tz) == "UTC"
    assert set(raw["token"]) == set(SUPPORTED_TOKENS)
