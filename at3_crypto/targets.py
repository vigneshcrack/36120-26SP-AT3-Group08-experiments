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
