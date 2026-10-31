"""Student C - Trend Category (TC) training pipeline (XGBoost Classifier).

Follows CONTRACT.md section 8 and mirrors Student D's pipeline so the two
classification services are trained, scored and reported the same way:

  - chronological split across all tokens, ~70/15/15, with a 7-day purge gap
    before validation and before test (never random splitting);
  - Optuna (TPE, seed 42) with TimeSeriesSplit(n_splits=5, gap=7) applied to
    the DISTINCT DATES of the training period (README section 7, decision D12);
  - baselines: majority class, stratified random, persistence (today's trend
    class as the 7-day forecast) and an untuned XGBoost;
  - metrics: macro-F1 over all five labels (primary), balanced accuracy,
    per-class recall, log-loss and calibration (ECE);
  - missing-feature augmentation (randomly masking peer features) so the NaN
    path is learned, plus a degradation table with each peer down;
  - the test set is scored once; the deployed model is refitted on
    train + validation with the selected hyperparameters;
  - seed 42 everywhere; artefacts written under models/student_c/ only.

Data comes from the committed snapshot (data/raw/kraken_daily_ohlcv.csv) by
default, so every number is reproducible; --refresh re-fetches from Kraken.

Run:  python -m at3_crypto.modeling.student_c_tc --trials 50
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
import json
import logging
import os
from pathlib import Path
from typing import Any, Final
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

from at3_crypto.engineered_features import engineer_features
from at3_crypto.kraken import SUPPORTED_TOKENS, KrakenClient
from at3_crypto.targets import FORECAST_HORIZON, TC_LABELS, build_tc_target, classify_tc

logger = logging.getLogger(__name__)

RANDOM_SEED: Final = 42
MODEL_VERSION: Final = "1.0.0"

TRAIN_FRACTION: Final = 0.70
VALIDATION_FRACTION: Final = 0.15
PURGE_DAYS: Final = FORECAST_HORIZON
CV_SPLITS: Final = 5
MASK_PROBABILITY: Final = 0.35
MAX_BOOSTING_ROUNDS: Final = 700

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
SNAPSHOT_PATH: Final = PROJECT_ROOT / "data" / "raw" / "kraken_daily_ohlcv.csv"
MODEL_DIR: Final = PROJECT_ROOT / "models" / "student_c"
FIGURE_DIR: Final = PROJECT_ROOT / "reports" / "figures" / "student_c"
PROCESSED_DIR: Final = PROJECT_ROOT / "data" / "processed" / "student_c"

# --------------------------------------------------------------------------
# Feature sets
# --------------------------------------------------------------------------
# The brief mandates MA5, MA20 and Return20. Two scale-free derivations of
# them are added because the model is pooled across eight tokens whose prices
# span five orders of magnitude (DOGE ~0.2 USD, BTC ~100k USD): raw MA levels
# act as a token identifier, not as a trend signal.
#   MA5_MA20_Ratio  = MA5 / MA20                (Student B's TrendStrength + 1)
#   TrendScore      = today's TrendScore         (the target's own formula at t)
# Student C's own features describe the *shape* of the recent path - how
# noisy (Volatility20), how far below its recent peak (MaxDrawdown) and how
# jumpy (GapRisk) - which is information the trend formula does not contain.
# Return5 (Student A) captures the most recent week, i.e. the part of the
# window that will still be inside MA5 at t+7.
# TrendScoreProj7 (Student C, engineered here) is the TrendScore that day t+7
# WOULD have if the price stayed flat at Close_t for the next 7 days. 13 of
# the 20 closes inside MA20(t+7), and the base price of Return20(t+7), are
# already known on day t, so this projection carries most of the target's
# structure while using only data up to t. It is computed from Kraken closes
# by this service itself, so it survives when Student A or B is down.
REQUIRED_FEATURES: Final[list[str]] = ["MA5", "MA20", "Return20"]
# Scale-free versions of the two required price levels (MA / Close): the same
# information the brief asks for, comparable across tokens.
SCALE_FREE_MA: Final[list[str]] = ["MA5_to_Close", "MA20_to_Close"]
DERIVED_FEATURES: Final[list[str]] = ["MA5_MA20_Ratio", "TrendScore"]
OWN_FEATURES: Final[list[str]] = ["Volatility20", "MaxDrawdown", "GapRisk"]
EXTRA_PEER_FEATURES: Final[list[str]] = ["Return5"]
PROJECTION_FEATURES: Final[list[str]] = ["TrendScoreProj7"]

# Candidate feature sets, compared in the notebook (section D) by CV and by
# validation macro-F1 with one fixed screening configuration.
FEATURE_SETS: Final[dict[str, list[str]]] = {
    "A_required_raw": ["Return20", "MA5", "MA20"],
    "B_raw_plus_all": ["Return20", "MA5", "MA20", *DERIVED_FEATURES, *OWN_FEATURES,
                       *EXTRA_PEER_FEATURES, *PROJECTION_FEATURES],
    "C_scale_free_core": ["Return20", *DERIVED_FEATURES, *EXTRA_PEER_FEATURES,
                          *PROJECTION_FEATURES],
    "D_scale_free_plus_vol": ["Return20", *SCALE_FREE_MA, *DERIVED_FEATURES,
                              *EXTRA_PEER_FEATURES, *PROJECTION_FEATURES, "Volatility20"],
    "E_scale_free_plus_own": ["Return20", *SCALE_FREE_MA, *DERIVED_FEATURES,
                              *EXTRA_PEER_FEATURES, *PROJECTION_FEATURES, *OWN_FEATURES],
}
DEFAULT_FEATURE_SET: Final = "C_scale_free_core"
FEATURE_COLUMNS: Final[list[str]] = list(FEATURE_SETS[DEFAULT_FEATURE_SET])

# Peer-owned inputs (CONTRACT.md section 2), for masking and degradation.
# TrendScore needs both A (Return20) and B (MA5/MA20), so it belongs to both.
PEER_FEATURE_GROUPS: Final[dict[str, list[str]]] = {
    "student_a": ["Return20", "Return5", "TrendScore"],
    "student_b": ["MA5", "MA20", "MA5_to_Close", "MA20_to_Close", "MA5_MA20_Ratio", "TrendScore"],
}

FEATURE_SOURCES: Final[dict[str, str]] = {
    "MA5": "Student B /features",
    "MA20": "Student B /features",
    "Return20": "Student A /features",
    "Return5": "Student A /features",
    "MA5_MA20_Ratio": "derived: Student B MA5 / MA20",
    "MA5_to_Close": "derived: Student B MA5 / Close_t (scale-free MA5)",
    "MA20_to_Close": "derived: Student B MA20 / Close_t (scale-free MA20)",
    "TrendScore": "derived: 0.7*(MA5/MA20 - 1) + 0.3*Return20 (needs A and B)",
    "Volatility20": "owned by Student C",
    "MaxDrawdown": "owned by Student C",
    "GapRisk": "owned by Student C",
    "TrendScoreProj7": "owned by Student C: flat-price projection of TrendScore at t+7 "
    "from Kraken closes",
}

SEARCH_SPACE: Final[dict[str, str]] = {
    "learning_rate": "log-uniform [0.005, 0.30]",
    "max_depth": "int [2, 8]",
    "min_child_weight": "log-uniform [0.5, 50]",
    "subsample": "uniform [0.5, 1.0]",
    "colsample_bytree": "uniform [0.5, 1.0]",
    "gamma": "uniform [0.0, 5.0] (min loss reduction to split)",
    "reg_alpha": "log-uniform [1e-8, 10.0] (L1)",
    "reg_lambda": "log-uniform [1e-3, 20.0] (L2)",
    "n_estimators": "int [100, 700] (tuned; no early stopping, which would select "
    "the round count on the fold being scored)",
    "class_balance": "NOT searched by TPE: two separate studies are run, one with "
    "'none' and one with 'balanced' (sklearn compute_sample_weight('balanced') as "
    "sample_weight); the winner is chosen on the held-out VALIDATION split",
    "cv": "TimeSeriesSplit(n_splits=5, gap=7) over the DISTINCT DATES of the "
    "training period only, so the gap purges 7 days and not 7 rows",
    "objective": "maximise mean macro-F1 across the 5 CV folds, all 5 labels",
}

# Fixed XGBoost settings shared by every fit. `hist` + a fixed seed and thread
# count make repeated runs reproducible.
BASE_PARAMS: Final[dict[str, Any]] = {
    "objective": "multi:softprob",
    "num_class": len(TC_LABELS),
    "tree_method": "hist",
    "eval_metric": "mlogloss",
    "random_state": RANDOM_SEED,
    "n_jobs": 4,
    "verbosity": 0,
}


@dataclass
class Dataset:
    """A chronologically split, model-ready dataset."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))

    def xy(self, split: str, columns: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
        frame = getattr(self, split)
        return frame[columns or self.feature_columns], frame["target_TC"].astype(int)

    def summary(self) -> dict[str, Any]:
        return {
            name: {
                "rows": len(frame),
                "start": str(frame["date"].min().date()),
                "end": str(frame["date"].max().date()),
                "class_balance": {
                    TC_LABELS[int(k)]: round(float(v), 4)
                    for k, v in frame["target_TC"].value_counts(normalize=True).sort_index().items()
                },
            }
            for name, frame in (
                ("train", self.train),
                ("validation", self.validation),
                ("test", self.test),
            )
        }


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def load_ohlcv(tokens: list[str] | None = None, refresh: bool = False) -> pd.DataFrame:
    """Daily OHLCV for `tokens`, from the committed snapshot or live from Kraken.

    Kraken serves only the latest 720 daily candles, so a live fetch moves the
    window forward and every number would change with the date; the snapshot is
    therefore the default.
    """
    tokens = tokens or list(SUPPORTED_TOKENS)
    if refresh:
        raw = KrakenClient().fetch_many(tokens)
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(SNAPSHOT_PATH, index=False)
        logger.info("Refreshed the snapshot at %s", SNAPSHOT_PATH)
    else:
        raw = pd.read_csv(SNAPSHOT_PATH)
        logger.info("Loaded the committed snapshot: %d rows", len(raw))

    raw = raw.loc[raw["token"].isin(tokens)].copy()
    raw["date"] = pd.to_datetime(raw["date"], utc=True)
    columns = ["date", "token", "open", "high", "low", "close", "volume"]
    return raw.loc[:, columns].sort_values(["token", "date"]).reset_index(drop=True)


def projected_trend_score(frame: pd.DataFrame, horizon: int = FORECAST_HORIZON) -> pd.Series:
    """TrendScore at t+horizon assuming Close stays flat at Close_t (Student C).

    With C_{t+1} = ... = C_{t+7} = C_t:
        MA5(t+7)      = C_t
        MA20(t+7)     = (sum(C_{t-12} .. C_t) + 7 * C_t) / 20
        Return20(t+7) = C_t / C_{t-13} - 1
    Only closes up to day t are used, so there is no look-ahead. Requires 14
    days of history per token; earlier rows are NaN.
    """
    known = 20 - horizon  # closes inside MA20(t+7) that are already observed
    grouped = frame.groupby("token", sort=False)["close"]
    close = frame["close"]
    known_sum = grouped.transform(lambda s: s.rolling(known, min_periods=known).sum())
    ma20_projected = (known_sum + horizon * close) / 20
    return20_projected = close / grouped.shift(known) - 1
    return 0.7 * (close / ma20_projected - 1) + 0.3 * return20_projected


def add_model_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add the scale-free derived inputs (TrendScore is added by build_tc_target)."""
    frame = features.sort_values(["token", "date"]).reset_index(drop=True)
    frame["MA5_MA20_Ratio"] = frame["MA5"] / frame["MA20"]
    frame["MA5_to_Close"] = frame["MA5"] / frame["close"]
    frame["MA20_to_Close"] = frame["MA20"] / frame["close"]
    frame["TrendScoreProj7"] = projected_trend_score(frame)
    return frame


def build_modelling_frame(raw: pd.DataFrame, horizon: int = FORECAST_HORIZON) -> pd.DataFrame:
    """OHLCV -> engineered features -> TrendScore/TC today and at t+horizon."""
    features = add_model_features(engineer_features(raw))
    return build_tc_target(features, horizon=horizon)


def load_dataset(
    tokens: list[str] | None = None,
    horizon: int = FORECAST_HORIZON,
    refresh: bool = False,
) -> tuple[Dataset, pd.DataFrame]:
    """Load OHLCV, engineer features, build the target, drop warm-up rows and split."""
    full = build_modelling_frame(load_ohlcv(tokens, refresh=refresh), horizon)
    all_inputs = sorted({c for cols in FEATURE_SETS.values() for c in cols})
    modelling = full.dropna(subset=[*all_inputs, "target_TC"]).reset_index(drop=True)
    logger.info(
        "Modelling frame: %d rows, %s .. %s",
        len(modelling),
        modelling["date"].min().date(),
        modelling["date"].max().date(),
    )
    return chronological_split(modelling, horizon), modelling


def chronological_split(frame: pd.DataFrame, horizon: int) -> Dataset:
    """Split on shared dates, purging `horizon` days at each seam."""
    dates = np.sort(frame["date"].unique())
    train_end = pd.Timestamp(dates[int(len(dates) * TRAIN_FRACTION)])
    validation_end = pd.Timestamp(dates[int(len(dates) * (TRAIN_FRACTION + VALIDATION_FRACTION))])
    purge = pd.Timedelta(horizon, unit="D")

    train = frame[frame["date"] <= train_end]
    validation = frame[(frame["date"] > train_end + purge) & (frame["date"] <= validation_end)]
    test = frame[frame["date"] > validation_end + purge]
    logger.info(
        "Split at %s / %s (purge %d days): %d / %d / %d rows",
        train_end.date(),
        validation_end.date(),
        horizon,
        len(train),
        len(validation),
        len(test),
    )
    return Dataset(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def purged_date_folds(
    dates: pd.Series, n_splits: int = CV_SPLITS, gap: int = PURGE_DAYS
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """TimeSeriesSplit applied to DISTINCT DATES, mapped back to row positions.

    The frame is a panel of eight tokens per date. Splitting rows directly
    would cut the panel by token and `gap=7` would purge 7 rows instead of
    7 days (group decision D12). Splitting the dates fixes both.
    """
    unique = np.sort(dates.unique())
    position = np.searchsorted(unique, dates.to_numpy())
    for train_dates, valid_dates in TimeSeriesSplit(n_splits=n_splits, gap=gap).split(unique):
        yield (
            np.flatnonzero(np.isin(position, train_dates)),
            np.flatnonzero(np.isin(position, valid_dates)),
        )


# --------------------------------------------------------------------------
# Missing-feature augmentation (CONTRACT.md section 7)
# --------------------------------------------------------------------------


def augment_missing_features(
    x: pd.DataFrame, y: pd.Series, seed: int = RANDOM_SEED
) -> tuple[pd.DataFrame, pd.Series]:
    """Append one copy of the rows with a random peer's features masked.

    At inference, Return20/Return5 come from Student A and MA5/MA20 from
    Student B. If either service is asleep or failing those inputs arrive as
    NaN. XGBoost handles NaN natively by learning a default branch direction,
    but only if NaN appears in training - otherwise the default direction is
    arbitrary. In the appended copy each row has, with probability 0.35, one
    randomly chosen peer's columns blanked. Student C's own features are never
    masked because this service always computes them itself.
    """
    rng = np.random.default_rng(seed)
    augmented = x.copy()
    peers = list(PEER_FEATURE_GROUPS)
    chosen = rng.integers(0, len(peers), size=len(augmented))
    masked = rng.random(len(augmented)) < MASK_PROBABILITY

    for index, peer in enumerate(peers):
        columns = [c for c in PEER_FEATURE_GROUPS[peer] if c in augmented.columns]
        rows = masked & (chosen == index)
        if rows.any() and columns:
            augmented.loc[rows, columns] = np.nan

    return (
        pd.concat([x, augmented], ignore_index=True),
        pd.concat([y, y], ignore_index=True),
    )


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

ALL_LABELS: Final[list[int]] = list(range(len(TC_LABELS)))


def score(y_true, y_pred) -> dict[str, float]:
    """Scores over the FIXED five-label set, so every figure is comparable."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        balanced = float(balanced_accuracy_score(y_true, y_pred))
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(balanced, 4),
        "macro_f1": round(
            float(f1_score(y_true, y_pred, labels=ALL_LABELS, average="macro", zero_division=0)), 4
        ),
        "weighted_f1": round(
            float(f1_score(y_true, y_pred, labels=ALL_LABELS, average="weighted", zero_division=0)),
            4,
        ),
        # Ordinal view: how often the forecast is within one category of the truth.
        # A Downtrend called Sideways is a much smaller error than one called
        # Strong Uptrend, and plain accuracy cannot tell those apart.
        "within_one_class": round(
            float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= 1)), 4
        ),
    }


def expected_calibration_error(y_true, probabilities, n_bins: int = 10) -> float:
    """ECE: the gap between stated confidence and observed accuracy.

    The app shows users a confidence score, so this is a product concern.
    """
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == np.asarray(y_true)).astype(float)
    error = 0.0
    for low, high in pairwise(np.linspace(0.0, 1.0, n_bins + 1)):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.any():
            error += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(error)


def predict_proba_all_labels(model: xgb.XGBClassifier, x: pd.DataFrame) -> np.ndarray:
    """Probabilities as a (rows x 5) matrix, floored and renormalised."""
    probabilities = np.clip(model.predict_proba(x), 1e-15, None)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------
# Model fitting
# --------------------------------------------------------------------------


def fit(params: dict[str, Any], x: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
    """Fit XGBoost with `params` on top of BASE_PARAMS.

    `class_balance` is not an XGBoost argument: "balanced" is translated into
    per-row sample weights (inverse class frequency), XGBoost's equivalent of
    LightGBM's class_weight="balanced".
    """
    params = dict(params)
    balance = params.pop("class_balance", "none")
    weights = compute_sample_weight("balanced", y) if balance == "balanced" else None
    model = xgb.XGBClassifier(**BASE_PARAMS, **params)
    model.fit(x, y, sample_weight=weights)
    return model


def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    """TPE search space (class_balance is fixed per study, see SEARCH_SPACE)."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.30, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 50.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, MAX_BOOSTING_ROUNDS),
    }


def cross_validate(
    params: dict[str, Any],
    data: Dataset,
    columns: list[str] | None = None,
    augment: bool = True,
) -> list[float]:
    """Mean-able list of macro-F1 scores over the purged date folds."""
    x_train, y_train = data.xy("train", columns)
    scores: list[float] = []
    for number, (fold_train, fold_valid) in enumerate(purged_date_folds(data.train["date"])):
        x_fold, y_fold = x_train.iloc[fold_train], y_train.iloc[fold_train]
        if augment:
            # Inside the fold, on training rows only; seed depends on the fold
            # (not the trial) so every candidate sees the same masks.
            x_fold, y_fold = augment_missing_features(x_fold, y_fold, seed=RANDOM_SEED + number)
        model = fit(params, x_fold, y_fold)
        predictions = model.predict(x_train.iloc[fold_valid])
        scores.append(
            float(
                f1_score(
                    y_train.iloc[fold_valid],
                    predictions,
                    labels=ALL_LABELS,
                    average="macro",
                    zero_division=0,
                )
            )
        )
    return scores


# A small, fixed configuration used to compare feature sets before tuning, so
# the feature-set choice is not entangled with the hyperparameter search.
SCREENING_PARAMS: Final[dict[str, Any]] = {
    "learning_rate": 0.03,
    "max_depth": 3,
    "n_estimators": 250,
    "min_child_weight": 10.0,
    "reg_lambda": 5.0,
    "class_balance": "none",
}


def compare_feature_sets(data: Dataset) -> dict[str, dict[str, float]]:
    """CV and validation macro-F1 of each candidate feature set (screening config)."""
    results = {}
    for name, columns in FEATURE_SETS.items():
        fold_scores = cross_validate(SCREENING_PARAMS, data, columns, augment=False)
        model = fit(SCREENING_PARAMS, *data.xy("train", columns))
        x_val, y_val = data.xy("validation", columns)
        results[name] = {
            "n_features": len(columns),
            "cv_macro_f1": round(float(np.mean(fold_scores)), 4),
            "cv_std": round(float(np.std(fold_scores)), 4),
            "validation_macro_f1": score(y_val, model.predict(x_val))["macro_f1"],
        }
        logger.info("Feature set %-24s %s", name, results[name])
    return results


def cv_rule_baselines(data: Dataset) -> dict[str, float]:
    """Persistence and flat-price projection scored on the SAME CV folds as
    Optuna, so the tuned CV score has a like-for-like reference."""
    y_train = data.train["target_TC"].astype(int)
    rules = {
        "persistence": data.train["TC"].astype(int),
        "flat_price_projection": classify_tc(data.train["TrendScoreProj7"]).astype(int),
    }
    out = {}
    for name, predicted in rules.items():
        fold_scores = [
            f1_score(
                y_train.iloc[v], predicted.iloc[v], labels=ALL_LABELS, average="macro", zero_division=0
            )
            for _, v in purged_date_folds(data.train["date"])
        ]
        out[name] = round(float(np.mean(fold_scores)), 4)
    return out


def optimise(
    data: Dataset, n_trials: int = 50, class_balance: str = "none", use_wandb: bool = False
) -> optuna.Study:
    """Optuna TPE study maximising mean CV macro-F1 over the purged date folds."""
    run = _start_wandb(n_trials, class_balance) if use_wandb else None

    def objective(trial: optuna.Trial) -> float:
        params = {**suggest_params(trial), "class_balance": class_balance}
        fold_scores = cross_validate(params, data)
        value = float(np.mean(fold_scores))
        trial.set_user_attr("fold_macro_f1", [round(s, 4) for s in fold_scores])
        trial.set_user_attr("fold_std", round(float(np.std(fold_scores)), 4))
        if run is not None:
            run.log({"trial": trial.number, "cv_macro_f1": value, **params})
        return value

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        study_name=f"student_c_tc_xgboost_{class_balance}",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    if run is not None:
        run.summary["best_cv_macro_f1"] = study.best_value
        run.finish()
    study.set_user_attr("class_balance", class_balance)
    logger.info(
        "[%s] best CV macro-F1 %.4f (trial %d)", class_balance, study.best_value,
        study.best_trial.number,
    )
    return study


def select_study(data: Dataset, studies: list[optuna.Study]) -> tuple[optuna.Study, dict]:
    """Pick the study whose best configuration scores highest on VALIDATION.

    Each candidate is fitted on the training split (with augmentation, as in
    CV) and scored on the validation split, which neither study has seen. The
    test split plays no part in the choice.
    """
    x_train, y_train = data.xy("train")
    x_val, y_val = data.xy("validation")
    x_aug, y_aug = augment_missing_features(x_train, y_train)
    report = {}
    for study in studies:
        params = selected_params(study)
        model = fit(params, x_aug, y_aug)
        report[study.user_attrs["class_balance"]] = {
            "best_cv_macro_f1": round(float(study.best_value), 4),
            **{f"validation_{k}": v for k, v in score(y_val, model.predict(x_val)).items()},
        }
    best = max(studies, key=lambda st: report[st.user_attrs["class_balance"]][
        "validation_macro_f1"])
    logger.info("Model selection on validation: %s -> %s", report,
                best.user_attrs["class_balance"])
    return best, report


def _start_wandb(n_trials: int, class_balance: str = "none"):
    """Start a W&B run if possible; offline without a key; never break training."""
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed - skipping experiment logging")
        return None
    if not os.getenv("WANDB_API_KEY"):
        os.environ.setdefault("WANDB_MODE", "offline")
        logger.warning("WANDB_API_KEY not set - logging offline to ./wandb")
    try:
        return wandb.init(
            project=os.getenv("WANDB_PROJECT", "36120-26SP-AT3-Group08"),
            group="student_c",
            name=f"student_c_tc-xgboost-{class_balance}-optuna-{n_trials}",
            config={"seed": RANDOM_SEED, "n_trials": n_trials, "search_space": SEARCH_SPACE},
            reinit="finish_previous",
        )
    except Exception as exc:  # noqa: BLE001 - logging must never break training
        logger.warning("Could not start a wandb run (%s) - continuing without it", exc)
        return None


# --------------------------------------------------------------------------
# Baselines, final fit and evaluation
# --------------------------------------------------------------------------


def score_baselines(data: Dataset) -> dict[str, dict[str, float]]:
    """Majority, stratified random, persistence and untuned XGBoost on test."""
    x_train, y_train = data.xy("train")
    x_test, y_test = data.xy("test")
    results: dict[str, dict[str, float]] = {}
    for name, strategy in (("majority_class", "most_frequent"), ("stratified_random", "stratified")):
        dummy = DummyClassifier(strategy=strategy, random_state=RANDOM_SEED).fit(x_train, y_train)
        results[name] = score(y_test, dummy.predict(x_test))
    # Persistence: today's trend class is the forecast for t+7.
    results["persistence"] = score(y_test, data.test["TC"].astype(int))
    # Rule-based projection: the class of the flat-price projected TrendScore.
    results["flat_price_projection"] = score(
        y_test, classify_tc(data.test["TrendScoreProj7"]).astype(int)
    )
    results["xgboost_defaults"] = score(y_test, fit({}, x_train, y_train).predict(x_test))
    return results


def selected_params(study: optuna.Study) -> dict[str, Any]:
    """The configuration actually fitted and deployed: study.best_params plus
    the study's fixed class_balance setting."""
    return {**study.best_params, "class_balance": study.user_attrs.get("class_balance", "none")}


def fit_final_model(data: Dataset, params: dict[str, Any]) -> xgb.XGBClassifier:
    """Refit on train + validation (NOT test), with augmentation.

    Refitting on train + validation only means the test figures describe the
    artefact that is actually served.
    """
    combined = pd.concat([data.train, data.validation], ignore_index=True)
    x, y = combined[data.feature_columns], combined["target_TC"].astype(int)
    x_aug, y_aug = augment_missing_features(x, y)
    logger.info("Refitting on %d rows (%d after augmentation)", len(x), len(x_aug))
    return fit(params, x_aug, y_aug)


def degradation_table(model: xgb.XGBClassifier, data: Dataset) -> dict[str, dict[str, float]]:
    """Test performance with each peer API down (its features set to NaN)."""
    x_test, y_test = data.xy("test")
    results = {"all_peers_available": score(y_test, model.predict(x_test))}
    for peer, columns in PEER_FEATURE_GROUPS.items():
        degraded = x_test.copy()
        degraded[[c for c in columns if c in degraded.columns]] = np.nan
        results[f"{peer}_down"] = score(y_test, model.predict(degraded))
    everything = x_test.copy()
    for columns in PEER_FEATURE_GROUPS.values():
        everything[[c for c in columns if c in everything.columns]] = np.nan
    results["all_peers_down"] = score(y_test, model.predict(everything))
    return results


def augmentation_ablation(data: Dataset, params: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Same hyperparameters fitted with and without augmentation, scored with
    all peers up and with Student A down - the measured cost/benefit."""
    combined = pd.concat([data.train, data.validation], ignore_index=True)
    x, y = combined[data.feature_columns], combined["target_TC"].astype(int)
    x_test, y_test = data.xy("test")
    a_down = x_test.copy()
    a_down[[c for c in PEER_FEATURE_GROUPS["student_a"] if c in a_down.columns]] = np.nan
    b_down = x_test.copy()
    b_down[[c for c in PEER_FEATURE_GROUPS["student_b"] if c in b_down.columns]] = np.nan

    results = {}
    for name, (fx, fy) in {
        "trained_clean": (x, y),
        "trained_augmented": augment_missing_features(x, y),
    }.items():
        model = fit(params, fx, fy)
        results[name] = {
            "macro_f1_all_peers_up": score(y_test, model.predict(x_test))["macro_f1"],
            "macro_f1_student_a_down": score(y_test, model.predict(a_down))["macro_f1"],
            "macro_f1_student_b_down": score(y_test, model.predict(b_down))["macro_f1"],
        }
    return results


def feature_importance(model: xgb.XGBClassifier, features: list[str]) -> dict[str, float]:
    """Normalised gain importance, largest first."""
    gains = model.get_booster().get_score(importance_type="gain")
    raw = {name: float(gains.get(name, 0.0)) for name in features}
    total = sum(raw.values()) or 1.0
    return {k: round(v / total, 4) for k, v in sorted(raw.items(), key=lambda kv: -kv[1])}


def evaluate(model: xgb.XGBClassifier, data: Dataset) -> dict[str, Any]:
    """Evaluate once on the held-out test split."""
    x_test, y_test = data.xy("test")
    predictions = model.predict(x_test)
    probabilities = predict_proba_all_labels(model, x_test)
    report = classification_report(
        y_test,
        predictions,
        labels=ALL_LABELS,
        target_names=list(TC_LABELS),
        output_dict=True,
        zero_division=0,
    )
    frame = data.test.assign(_pred=predictions)
    return {
        **score(y_test, predictions),
        "log_loss": round(float(log_loss(y_test, probabilities, labels=ALL_LABELS)), 4),
        "ece": round(expected_calibration_error(y_test, probabilities), 4),
        "mean_confidence": round(float(probabilities.max(axis=1).mean()), 4),
        "per_class_recall": {lbl: round(float(report[lbl]["recall"]), 4) for lbl in TC_LABELS},
        "per_class_precision": {
            lbl: round(float(report[lbl]["precision"]), 4) for lbl in TC_LABELS
        },
        "labels_in_test": [TC_LABELS[c] for c in sorted(set(y_test))],
        "confusion_matrix": {
            "labels": list(TC_LABELS),
            "matrix": confusion_matrix(y_test, predictions, labels=ALL_LABELS).tolist(),
        },
        "per_token": {
            str(token): {**score(g["target_TC"].astype(int), g["_pred"]), "rows": len(g)}
            for token, g in frame.groupby("token", sort=True)
        },
        "degradation": degradation_table(model, data),
    }


# --------------------------------------------------------------------------
# Artefacts
# --------------------------------------------------------------------------


def build_metrics(
    model: xgb.XGBClassifier,
    data: Dataset,
    study: optuna.Study,
    metrics: dict[str, Any],
    tokens: list[str],
) -> dict[str, Any]:
    """metrics.json in the shape GET /model-metadata serves (CONTRACT.md 6.5)."""
    training_dates = pd.concat([data.train, data.validation])["date"]
    return {
        "model_version": MODEL_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "target": "Trend Category",
        "target_description": (
            "Five-class trend label (Strong Downtrend .. Strong Uptrend) from "
            "TrendScore = 0.7*((MA5/MA20) - 1) + 0.3*Return20, measured 7 days "
            "after the prediction date"
        ),
        "algorithm": "XGBoost Classifier",
        "forecast_horizon": FORECAST_HORIZON,
        "features": list(data.feature_columns),
        "feature_sources": {c: FEATURE_SOURCES[c] for c in data.feature_columns},
        "feature_importance": feature_importance(model, data.feature_columns),
        "classification_labels": list(TC_LABELS),
        "classification_rules": {
            "Strong Downtrend": "TrendScore < -0.10",
            "Downtrend": "-0.10 <= TrendScore < -0.02",
            "Sideways": "-0.02 <= TrendScore <= 0.02",
            "Uptrend": "0.02 < TrendScore <= 0.10",
            "Strong Uptrend": "TrendScore > 0.10",
        },
        "supported_tokens": list(tokens),
        "metrics": {
            "test_macro_f1": metrics["macro_f1"],
            "test_balanced_accuracy": metrics["balanced_accuracy"],
            "test_accuracy": metrics["accuracy"],
            "test_within_one_class": metrics["within_one_class"],
            "test_log_loss": metrics["log_loss"],
            "test_ece": metrics["ece"],
        },
        "metrics_note": (
            "macro-F1 averages over all five TC labels, including any absent from "
            "the test window, so the model, baselines and per-token figures are comparable"
        ),
        "per_class_recall": metrics["per_class_recall"],
        "per_class_precision": metrics["per_class_precision"],
        "labels_in_test": metrics["labels_in_test"],
        "per_token": metrics["per_token"],
        "confusion_matrix": metrics["confusion_matrix"],
        "baselines": metrics.get("baselines", {}),
        "feature_set_comparison": metrics.get("feature_set_comparison", {}),
        "model_selection": metrics.get("model_selection", {}),
        "augmentation_ablation": metrics.get("augmentation_ablation", {}),
        "degradation": metrics["degradation"],
        "training_period": {
            "start": str(training_dates.min().date()),
            "end": str(training_dates.max().date()),
        },
        "split": data.summary(),
        "best_params": selected_params(study),
        "best_cv_macro_f1": round(float(study.best_value), 4),
        "cv_strategy": (
            "TimeSeriesSplit(n_splits=5, gap=7) over the DISTINCT DATES of the "
            "training period, so folds are ordered in time and the gap purges 7 days"
        ),
        "search_space": SEARCH_SPACE,
        "n_trials": len(study.trials),
        "random_seed": RANDOM_SEED,
        "data_source": (
            "Kraken public REST API /0/public/OHLC (interval=1440), via the "
            "committed snapshot data/raw/kraken_daily_ohlcv.csv"
        ),
        "library_versions": {
            "xgboost": xgb.__version__,
            "optuna": optuna.__version__,
            "pandas": pd.__version__,
        },
    }


def save_splits(data: Dataset) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for split, suffix in (("train", "train"), ("validation", "val"), ("test", "test")):
        x, y = data.xy(split)
        x.to_csv(PROCESSED_DIR / f"X_{suffix}.csv", index=False)
        y.to_frame("target_TC").to_csv(PROCESSED_DIR / f"y_{suffix}.csv", index=False)


def save_artefacts(model, data, study, metrics, tokens) -> dict[str, Any]:
    """Write models/student_c/tc_xgboost.joblib and metrics.json; return the payload."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_metrics(model, data, study, metrics, tokens)
    bundle = {
        "model": model,
        "feature_columns": list(data.feature_columns),
        "labels": dict(enumerate(TC_LABELS)),
        **payload,
    }
    joblib.dump(bundle, MODEL_DIR / "tc_xgboost.joblib", compress=3)
    # Also the native XGBoost format, which loads across xgboost versions.
    model.save_model(MODEL_DIR / "tc_xgboost.json")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(payload, indent=2, default=str))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--tokens", nargs="*", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    tokens = args.tokens or list(SUPPORTED_TOKENS)

    data, _ = load_dataset(tokens=tokens, refresh=args.refresh)
    feature_sets = compare_feature_sets(data)
    feature_sets["_cv_rule_baselines"] = cv_rule_baselines(data)
    baselines = score_baselines(data)
    studies = [
        optimise(data, n_trials=args.trials, class_balance=b, use_wandb=args.wandb)
        for b in ("none", "balanced")
    ]
    study, selection = select_study(data, studies)
    params = selected_params(study)
    model = fit_final_model(data, params)
    metrics = evaluate(model, data)
    metrics["baselines"] = baselines
    metrics["feature_set_comparison"] = feature_sets
    metrics["model_selection"] = selection
    metrics["augmentation_ablation"] = augmentation_ablation(data, params)
    logger.info(
        "TEST macro-F1 %.4f | balanced acc %.4f | ECE %.4f",
        metrics["macro_f1"],
        metrics["balanced_accuracy"],
        metrics["ece"],
    )
    save_splits(data)
    save_artefacts(model, data, study, metrics, tokens)
    logger.info("Wrote artefacts to %s", MODEL_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
