"""Engineered features, per CONTRACT.md section 4.

One implementation shared by all four services so training and serving cannot
drift apart. Names below are the EXACT JSON keys agreed in the contract - a
service consuming a peer's `/features` matches on them, so renaming one here
breaks another student's model.

Ownership (contract section 2):
    Student A  DailyReturn, Return5, Return20, Momentum
    Student B  MA5, MA20, TrendStrength
    Student C  Volatility20, MaxPrice20, MaxDrawdown, GapRisk
    Student D  VolumeSMA20, RelativeVolume

All windows use `min_periods = window`, so the leading rows are NaN until
enough history exists, and every rolling computation is grouped by token so a
window never spans two assets.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

SHORT_WINDOW: Final = 5
LONG_WINDOW: Final = 20

# Feature -> owning student, exactly as agreed in CONTRACT.md section 2.
FEATURE_OWNERS: Final[dict[str, str]] = {
    "DailyReturn": "student_a",
    "Return5": "student_a",
    "Return20": "student_a",
    "Momentum": "student_a",
    "MA5": "student_b",
    "MA20": "student_b",
    "TrendStrength": "student_b",
    "Volatility20": "student_c",
    "MaxPrice20": "student_c",
    "MaxDrawdown": "student_c",
    "GapRisk": "student_c",
    "VolumeSMA20": "student_d",
    "RelativeVolume": "student_d",
}

OWNED_BY_STUDENT_D: Final[list[str]] = ["VolumeSMA20", "RelativeVolume"]

# Formulas served on Student D's POST /features, so consumers can see the
# definition without reading the source.
OWNED_FEATURE_DEFINITIONS: Final[dict[str, str]] = {
    "VolumeSMA20": "VolumeSMA20 = mean(Volume_t ... Volume_t-19)",
    "RelativeVolume": "RelativeVolume = Volume_t / VolumeSMA20",
}

REQUIRED_OHLCV_COLUMNS: Final = {
    "date", "token", "open", "high", "low", "close", "volume",
}


def engineer_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Add every CONTRACT.md section 4 feature to a daily OHLCV frame.

    Expects the columns produced by `KrakenClient.fetch_ohlcv`. Returns a copy
    sorted by (token, date) with the engineered columns appended. Leading rows
    keep NaN rather than being dropped, so the caller decides.
    """
    missing = REQUIRED_OHLCV_COLUMNS - set(ohlcv.columns)
    if missing:
        raise ValueError(f"OHLCV frame is missing columns: {sorted(missing)}")

    frame = ohlcv.sort_values(["token", "date"]).copy()
    grouped = frame.groupby("token", sort=False)

    # --- Student A ---------------------------------------------------------
    frame["DailyReturn"] = grouped["close"].pct_change(1)
    frame["Return5"] = grouped["close"].pct_change(SHORT_WINDOW)
    frame["Return20"] = grouped["close"].pct_change(LONG_WINDOW)
    frame["Momentum"] = frame["close"] - grouped["close"].shift(LONG_WINDOW)

    # --- Student B ---------------------------------------------------------
    frame["MA5"] = _rolling(frame, "close", SHORT_WINDOW, "mean")
    frame["MA20"] = _rolling(frame, "close", LONG_WINDOW, "mean")
    frame["TrendStrength"] = (frame["MA5"] / frame["MA20"]) - 1.0

    # --- Student C ---------------------------------------------------------
    # Sample standard deviation (ddof=1), not annualised - contract section 4.
    frame["Volatility20"] = _rolling(frame, "DailyReturn", LONG_WINDOW, "std")
    frame["MaxPrice20"] = _rolling(frame, "high", LONG_WINDOW, "max")
    frame["MaxDrawdown"] = (frame["MaxPrice20"] - frame["close"]) / frame["MaxPrice20"]
    previous_close = grouped["close"].shift(1)
    overnight_gap = (frame["open"] - previous_close).abs() / previous_close
    frame["GapRisk"] = (
        overnight_gap.groupby(frame["token"], sort=False)
        .rolling(LONG_WINDOW, min_periods=LONG_WINDOW)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # --- Student D (owned here) -------------------------------------------
    frame["VolumeSMA20"] = _rolling(frame, "volume", LONG_WINDOW, "mean")
    # Guard the division: thinly traded tokens can report a zero 20-day average
    # volume, and an infinity would survive dropna() and reach the model.
    frame["RelativeVolume"] = np.where(
        frame["VolumeSMA20"] > 0, frame["volume"] / frame["VolumeSMA20"], np.nan
    )

    return frame.reset_index(drop=True)


def _rolling(frame: pd.DataFrame, column: str, window: int, how: str) -> pd.Series:
    """Per-token rolling aggregation requiring a full window."""
    rolled = (
        frame.groupby("token", sort=False)[column]
        .rolling(window, min_periods=window)
        .agg(how)
    )
    return rolled.reset_index(level=0, drop=True)
