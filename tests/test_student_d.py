"""Tests for Student D's TOR features and target.

CONTRACT.md section 5 requires the classification boundary values to be unit
tested, because the bands mix inclusive and exclusive bounds. The feature tests
guard the exact JSON keys other services consume: renaming `VolumeSMA20` or
`RelativeVolume` silently breaks Student A's model, and nothing else in the
pipeline would fail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from at3_crypto.engineered_features import (
    FEATURE_OWNERS,
    LONG_WINDOW,
    OWNED_BY_STUDENT_D,
    engineer_features,
)
from at3_crypto.kraken import SUPPORTED_TOKENS, KrakenClient
from at3_crypto.targets import (
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
