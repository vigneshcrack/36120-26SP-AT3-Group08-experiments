"""Target construction, per CONTRACT.md section 5.

Implements Student D's Trading Opportunity Rating (TOR). The contract requires
`np.select` rather than `pd.cut` for the mixed inclusive/exclusive boundary
rules, and that the boundaries be unit-tested (see tests/test_student_d.py).

For every token: compute the index from the features at each date, then
`target_t = index.shift(-7)` within that token. The final 7 rows per token have
no target and are dropped from training.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

FORECAST_HORIZON: Final = 7

# Ordered worst -> best. The exact strings are fixed by CONTRACT.md section 5
# and are what the API returns and Streamlit renders.
TOR_LABELS: Final[list[str]] = [
    "Strong Sell",
    "Sell",
    "Hold",
    "Buy",
    "Strong Buy",
]
TOR_LABEL_BY_INDEX: Final[dict[int, str]] = dict(enumerate(TOR_LABELS))

# Features the TOR target is built from, and therefore the model's inputs.
TOR_INPUT_FEATURES: Final[list[str]] = [
    "Return20",
    "MA5",
    "MA20",
    "RelativeVolume",
    "Volatility20",
]


def opportunity_score(features: pd.DataFrame | dict) -> pd.Series | float:
    """Opportunity Score, CONTRACT.md section 5 (Student D).

        M  = (Return20 + 0.20) / 0.40      momentum
        T  = MA5 / MA20                    trend
        V  = min(RelativeVolume, 2) / 2    volume
        R  = min(Volatility20 / 0.10, 1)   volatility penalty

        OS = 0.40*M + 0.30*T + 0.20*V - 0.10*R

    The contract clips V and R but leaves M and T unclipped, so OS is not
    bounded to [0, 1]. Implemented as written.

    Accepts a DataFrame (returns a Series) or a mapping of scalars (returns a
    float), so one function serves training and single-row inference.
    """
    scalar_input = not isinstance(features, pd.DataFrame)
    frame = pd.DataFrame([features]) if scalar_input else features

    momentum = (frame["Return20"] + 0.20) / 0.40
    trend = frame["MA5"] / frame["MA20"]
    volume = frame["RelativeVolume"].clip(upper=2) / 2
    volatility_penalty = (frame["Volatility20"] / 0.10).clip(upper=1)

    score = 0.40 * momentum + 0.30 * trend + 0.20 * volume - 0.10 * volatility_penalty
    return float(score.iloc[0]) if scalar_input else score


def classify_tor(score: pd.Series | float) -> pd.Series | int:
    """Map an Opportunity Score to its TOR class index (0-4).

    Uses `np.select` as the contract requires: the bands mix inclusive and
    exclusive bounds (`0.20 <= OS < 0.40`), which `pd.cut` cannot express
    directly without care about `right=`.
    """
    scalar_input = not isinstance(score, pd.Series)
    series = pd.Series([score], dtype="float64") if scalar_input else score
    values = series.to_numpy(dtype="float64")

    conditions = [
        values < 0.20,
        (values >= 0.20) & (values < 0.40),
        (values >= 0.40) & (values < 0.60),
        (values >= 0.60) & (values < 0.80),
        values >= 0.80,
    ]
    classes = pd.Series(
        np.select(conditions, [0, 1, 2, 3, 4], default=np.nan).astype("float64"),
        index=series.index,
    )
    classes[series.isna()] = np.nan

    return int(classes.iloc[0]) if scalar_input else classes


def build_tor_target(features: pd.DataFrame, horizon: int = FORECAST_HORIZON) -> pd.DataFrame:
    """Append today's OS/TOR and the forward-looking TOR target.

    Adds:
        OS          Opportunity Score on date t
        TOR         TOR class on date t (for charts and the persistence
                    baseline, never a model feature)
        target_OS   Opportunity Score on date t+horizon
        target_TOR  TOR class on date t+horizon  <- the label

    The shift is applied per token, so no label crosses a token boundary.
    """
    frame = features.sort_values(["token", "date"]).copy()
    frame["OS"] = opportunity_score(frame)
    frame["TOR"] = classify_tor(frame["OS"])

    grouped = frame.groupby("token", sort=False)
    frame["target_OS"] = grouped["OS"].shift(-horizon)
    frame["target_TOR"] = grouped["TOR"].shift(-horizon)

    return frame.reset_index(drop=True)


# Student C - Trend Category (TC), CONTRACT.md section 5
# ==========================================================================

# Ordered most bearish -> most bullish. Exact strings fixed by CONTRACT.md 5.
TC_LABELS: Final[list[str]] = [
    "Strong Downtrend",
    "Downtrend",
    "Sideways",
    "Uptrend",
    "Strong Uptrend",
]
TC_LABEL_BY_INDEX: Final[dict[int, str]] = dict(enumerate(TC_LABELS))

# Features the brief mandates for the TC model.
TC_REQUIRED_FEATURES: Final[list[str]] = ["MA5", "MA20", "Return20"]


def trend_score(features: pd.DataFrame | dict) -> pd.Series | float:
    """TrendScore = 0.7 * ((MA5 / MA20) - 1) + 0.3 * Return20  (Student C).

    Accepts a DataFrame (returns a Series) or a mapping of scalars (returns a
    float), so one function serves training and single-row inference.
    """
    scalar_input = not isinstance(features, pd.DataFrame)
    frame = pd.DataFrame([features]) if scalar_input else features
    score = 0.7 * ((frame["MA5"] / frame["MA20"]) - 1.0) + 0.3 * frame["Return20"]
    return float(score.iloc[0]) if scalar_input else score


def classify_tc(score: pd.Series | float) -> pd.Series | int:
    """Map a TrendScore to its TC class index (0-4) with np.select.

        TrendScore <  -0.10            Strong Downtrend
        -0.10 <= TrendScore < -0.02    Downtrend
        -0.02 <= TrendScore <= 0.02    Sideways      (both ends inclusive)
         0.02 <  TrendScore <= 0.10    Uptrend
        TrendScore >  0.10             Strong Uptrend

    Sideways is closed on both sides, so the bands are not a plain half-open
    `pd.cut`; `np.select` states each boundary exactly as the brief does.
    NaN in -> NaN out (or None for a scalar).
    """
    scalar_input = not isinstance(score, pd.Series)
    series = pd.Series([score], dtype="float64") if scalar_input else score
    v = series.to_numpy(dtype="float64")

    conditions = [
        v < -0.10,
        (v >= -0.10) & (v < -0.02),
        (v >= -0.02) & (v <= 0.02),
        (v > 0.02) & (v <= 0.10),
        v > 0.10,
    ]
    classes = pd.Series(
        np.select(conditions, [0, 1, 2, 3, 4], default=np.nan).astype("float64"),
        index=series.index,
    )
    classes[series.isna()] = np.nan
    if scalar_input:
        value = classes.iloc[0]
        return None if pd.isna(value) else int(value)
    return classes


def build_tc_target(features: pd.DataFrame, horizon: int = FORECAST_HORIZON) -> pd.DataFrame:
    """Append today's TrendScore/TC and the forward-looking TC target.

    Adds:
        TrendScore         TrendScore on date t (a model feature: it is a
                           deterministic function of Return20, MA5 and MA20)
        TC                 TC class on date t (persistence baseline, charts)
        target_TrendScore  TrendScore on date t+horizon
        target_TC          TC class on date t+horizon   <- the label

    The shift is applied per token, so no label crosses a token boundary.
    """
    frame = features.sort_values(["token", "date"]).copy()
    frame["TrendScore"] = trend_score(frame)
    frame["TC"] = classify_tc(frame["TrendScore"])

    grouped = frame.groupby("token", sort=False)
    frame["target_TrendScore"] = grouped["TrendScore"].shift(-horizon)
    frame["target_TC"] = grouped["TC"].shift(-horizon)
    return frame.reset_index(drop=True)
