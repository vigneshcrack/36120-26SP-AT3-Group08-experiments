"""Student D - Trading Opportunity Rating (TOR) training pipeline.

Follows CONTRACT.md section 8:

  - chronological split across all tokens, 70/15/15, with a 7-day purge gap
    before validation and before test (never random splitting);
  - Optuna (TPE, seed 42) with TimeSeriesSplit(n_splits=5, gap=7) on the
    training period only; search space and best parameters documented;
  - baselines include majority class AND persistence (today's class as the
    7-day forecast);
  - classification metrics: macro-F1 (primary), balanced accuracy, per-class
    recall, log-loss and calibration (ECE);
  - missing-feature augmentation (randomly masking peer features) so the NaN
    path is learned, plus a degradation table reporting performance with each
    peer down;
  - the test set is evaluated once; the deployed model is refitted with the
    selected hyperparameters;
  - seed 42 everywhere; artefacts written under models/student_d/ only.

The contract's `TimeSeriesSplit(n_splits=5, gap=7)` is applied to the DISTINCT
DATES of the training period, not to its rows - see `purged_date_folds`. The
modelling frame is a panel of eight tokens sharing every date, so splitting rows
would cut it by token instead of by time.

Data comes from the committed snapshot (`data/raw/kraken_daily_ohlcv.csv`) by
default, so every number is reproducible; `--refresh` fetches live from Kraken
and rewrites the snapshot.

Run:  python -m at3_crypto.modeling.student_d_tor --trials 40
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
import lightgbm as lgb
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

from at3_crypto.dataset import SNAPSHOT_PATH, load_snapshot
from at3_crypto.engineered_features import engineer_features
from at3_crypto.kraken import SUPPORTED_TOKENS, KrakenClient
from at3_crypto.targets import FORECAST_HORIZON, TOR_LABELS, build_tor_target

logger = logging.getLogger(__name__)

RANDOM_SEED: Final = 42  # CONTRACT.md section 8
MODEL_VERSION: Final = "1.0.0"

TRAIN_FRACTION: Final = 0.70
VALIDATION_FRACTION: Final = 0.15
PURGE_DAYS: Final = FORECAST_HORIZON

# Model inputs: the five features the brief mandates for TOR, plus the
# scale-free MA5/MA20 ratio. The ratio is the contract's own intermediate
# variable T; it is included because MA5 and MA20 are absolute price levels and
# the model is trained across eight tokens whose prices span five orders of
# magnitude, where the raw levels act as a token identifier rather than a trend.
FEATURE_COLUMNS: Final[list[str]] = [
    "Return20",
    "RelativeVolume",
    "Volatility20",
    "MA5",
    "MA20",
    "MA5_MA20_Ratio",
    "OS",
]

# Peer-owned inputs, for the masking augmentation and the degradation table.
PEER_FEATURE_GROUPS: Final[dict[str, list[str]]] = {
    "student_a": ["Return20", "OS"],
    "student_b": ["MA5", "MA20", "MA5_MA20_Ratio", "OS"],
    "student_c": ["Volatility20", "OS"],
}

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
MODEL_DIR: Final = PROJECT_ROOT / "models" / "student_d"
FIGURE_DIR: Final = PROJECT_ROOT / "reports" / "figures" / "student_d"
PROCESSED_DIR: Final = PROJECT_ROOT / "data" / "processed" / "student_d"

MAX_BOOSTING_ROUNDS: Final = 700
MASK_PROBABILITY: Final = 0.35  # share of rows masked in the augmented copy
CV_SPLITS: Final = 5

SEARCH_SPACE: Final[dict[str, str]] = {
    "learning_rate": "log-uniform [0.005, 0.30]",
    "max_depth": "int [3, 12]",
    "num_leaves": "int [8, min(255, 2**max_depth)] (a wider range is inert: "
    "LightGBM cannot grow more leaves than the depth allows)",
    "min_child_samples": "int [5, 120]",
    "subsample": "uniform [0.5, 1.0] (bagging every subsample_freq iterations)",
    "subsample_freq": "int [1, 7]",
    "colsample_bytree": "uniform [0.5, 1.0]",
    "reg_alpha": "log-uniform [1e-8, 10.0] (L1)",
    "reg_lambda": "log-uniform [1e-8, 10.0] (L2)",
    "min_split_gain": "uniform [0.0, 0.5]",
    "class_weight": "categorical [None, 'balanced']",
    "n_estimators": "int [150, 700] (tuned; no early stopping, which would "
    "select the round count on the fold being scored)",
    "cv": "TimeSeriesSplit(n_splits=5, gap=7) over the DISTINCT DATES of the "
    "training period only, so the gap purges 7 days and not 7 rows",
    "objective": "maximise mean macro-F1 across the 5 CV folds, all 5 labels",
}


@dataclass
class Dataset:
    """A chronologically split, model-ready dataset."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))

    def xy(self, split: str) -> tuple[pd.DataFrame, pd.Series]:
        frame = getattr(self, split)
        return frame[self.feature_columns], frame["target_TOR"].astype(int)

    def summary(self) -> dict[str, Any]:
        return {
            name: {
                "rows": len(frame),
                "start": str(frame["date"].min().date()),
                "end": str(frame["date"].max().date()),
                "class_balance": {
                    TOR_LABELS[int(k)]: round(v, 4)
                    for k, v in frame["target_TOR"]
                    .value_counts(normalize=True)
                    .sort_index()
                    .items()
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
    """Return daily OHLCV for `tokens`, from the snapshot or live from Kraken.

    Kraken serves only the latest 720 daily candles, so a live fetch moves the
    window forward and every number in the notebook would change with the date.
    The default therefore reads the committed snapshot, exactly as Student B's
    pipeline does, and `refresh=True` re-fetches and rewrites it.

    Both paths return the same seven columns with a UTC-aware `date`, so the
    features are identical whichever source is used.
    """
    tokens = tokens or list(SUPPORTED_TOKENS)

    if refresh:
        raw = KrakenClient().fetch_many(tokens)
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(SNAPSHOT_PATH, index=False)
        logger.info("Refreshed the snapshot at %s", SNAPSHOT_PATH)
    else:
        if not SNAPSHOT_PATH.exists():
            raise FileNotFoundError(
                f"No snapshot at {SNAPSHOT_PATH}. Run with refresh=True "
                "(or `python -m at3_crypto.dataset`) to download one."
            )
        raw = load_snapshot()
        logger.info("Loaded the committed snapshot: %d rows", len(raw))

    raw = raw.loc[raw["token"].isin(tokens)].copy()
    raw["date"] = pd.to_datetime(raw["date"], utc=True)
    columns = ["date", "token", "open", "high", "low", "close", "volume"]
    return raw.loc[:, columns].sort_values(["token", "date"]).reset_index(drop=True)


def load_dataset(
    tokens: list[str] | None = None,
    horizon: int = FORECAST_HORIZON,
    refresh: bool = False,
) -> tuple[Dataset, pd.DataFrame]:
    """Load OHLCV, engineer features, build the target and split."""
    raw = load_ohlcv(tokens, refresh=refresh)

    features = engineer_features(raw)
    features["MA5_MA20_Ratio"] = features["MA5"] / features["MA20"]

    # build_tor_target adds OS (today's Opportunity Score), which is one of the
    # model features, so the target must be built before rows are dropped.
    modelling = build_tor_target(features, horizon=horizon)
    modelling = modelling.dropna(subset=[*FEATURE_COLUMNS, "target_TOR"])
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
    """CONTRACT.md section 8's TimeSeriesSplit, applied to DATES not rows.

    The modelling frame is a panel: all eight tokens share every date, and the
    rows arrive grouped by token. Handing those rows straight to
    `TimeSeriesSplit` therefore cuts the panel BY TOKEN - every fold's
    validation set spans the full date range of its training set, so the folds
    measure cross-token generalisation and not forecasting, and `gap=7` purges
    seven ROWS (less than one calendar day of an eight-token panel) instead of
    seven days.

    Splitting the distinct dates and mapping them back to row positions gives
    the contract's intended semantics: folds are ordered in time and the gap
    purges `gap` calendar days, which is exactly the label horizon. Row order
    inside the frame stops mattering at all.

    Yields (train_positions, validation_positions) as positional indices into
    `dates`.
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
    """Append copies of the training rows with one peer's features masked.

    At inference, peer features arrive from independently deployed services that
    can be asleep or failing, in which case the contract says they become NaN.
    Training only on complete rows would leave the NaN path unlearned: LightGBM
    would still run, but its default split direction for missing values would be
    whatever the complete data happened to imply. Masking teaches it explicitly.

    ONE extra copy of the training rows is appended (not one per peer): in that
    copy each row independently has a randomly chosen peer's features blanked
    with probability `MASK_PROBABILITY`. That covers every peer-down pattern
    while only doubling the training rows instead of quadrupling them, so the
    fit stays fast enough for teaching staff to re-run the notebook.
    """
    rng = np.random.default_rng(seed)
    augmented = x.copy()

    peers = list(PEER_FEATURE_GROUPS)
    chosen = rng.integers(0, len(peers), size=len(augmented))
    masked = rng.random(len(augmented)) < MASK_PROBABILITY

    for index, peer in enumerate(peers):
        columns = [c for c in PEER_FEATURE_GROUPS[peer] if c in augmented.columns]
        rows = masked & (chosen == index)
        if rows.any():
            augmented.loc[rows, columns] = np.nan
            logger.debug("Augmented %d rows with %s masked", int(rows.sum()), peer)

    return (
        pd.concat([x, augmented], ignore_index=True),
        pd.concat([y, y], ignore_index=True),
    )


# --------------------------------------------------------------------------
# Baselines (CONTRACT.md section 8: majority AND persistence)
# --------------------------------------------------------------------------


def score_baselines(data: Dataset) -> dict[str, dict[str, float]]:
    """Score the required baselines on the test split."""
    x_train, y_train = data.xy("train")
    x_test, y_test = data.xy("test")
    results: dict[str, dict[str, float]] = {}

    for name, strategy in (
        ("majority_class", "most_frequent"),
        ("stratified_random", "stratified"),
    ):
        dummy = DummyClassifier(strategy=strategy, random_state=RANDOM_SEED)
        dummy.fit(x_train, y_train)
        results[name] = _score(y_test, dummy.predict(x_test))

    # Persistence: today's TOR class is the forecast for t+7. The baseline to
    # beat, because ~half of ratings do not change over a week.
    persistence = data.test["TOR"].astype(int)
    results["persistence"] = _score(y_test, persistence)

    default = lgb.LGBMClassifier(**_BASE_PARAMS)
    default.fit(x_train, y_train)
    results["lightgbm_defaults"] = _score(y_test, default.predict(x_test))
    return results


def _score(y_true, y_pred) -> dict[str, float]:
    """Score against the FIXED five-label set, so every figure is comparable.

    Averaging over only the labels that happen to appear would make macro-F1
    mean a different thing for each model and for each token - `Strong Sell` is
    ~2% of the data and is absent from some windows entirely, so a four-class
    average and a five-class average are not the same number. Scoring all five
    labels penalises every model and baseline identically, which is what makes
    the comparison in the notebook legitimate.
    """
    labels = list(range(len(TOR_LABELS)))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        balanced = float(balanced_accuracy_score(y_true, y_pred))
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(balanced, 4),
        "macro_f1": round(
            float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4
        ),
        "weighted_f1": round(
            float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)), 4
        ),
    }


# --------------------------------------------------------------------------
# Hyperparameter optimisation
# --------------------------------------------------------------------------


def _suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    # max_depth is drawn first so num_leaves can be bounded by it: LightGBM
    # grows leaf-wise but still honours max_depth, so a num_leaves above
    # 2**max_depth is silently inert and wastes the trial budget on
    # configurations that are duplicates of a smaller one.
    max_depth = trial.suggest_int("max_depth", 3, 12)
    return {
        "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.30, log=True),
        "max_depth": max_depth,
        "num_leaves": trial.suggest_int("num_leaves", 8, min(255, 2**max_depth)),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 120),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "n_estimators": trial.suggest_int("n_estimators", 150, MAX_BOOSTING_ROUNDS),
    }


# Fixed LightGBM settings shared by every fit, so the only thing that varies
# between trials is the searched hyperparameters. `deterministic` and
# `force_row_wise` make repeated runs bit-identical, which is what lets the
# README promise a reproducible result.
_BASE_PARAMS: Final[dict[str, Any]] = {
    "objective": "multiclass",
    "num_class": len(TOR_LABELS),
    "random_state": RANDOM_SEED,
    "deterministic": True,
    "force_row_wise": True,
    "n_jobs": -1,
    "verbose": -1,
}


def _fit(params: dict[str, Any], x: pd.DataFrame, y: pd.Series) -> lgb.LGBMClassifier:
    """Fit LightGBM with `params` on top of the fixed base settings.

    There is deliberately no early stopping. The only held-out data available
    inside a CV fold is the fold being scored, so stopping on it would choose
    the round count by looking at the very rows the trial is judged on and
    report an optimistic score. `n_estimators` is searched by Optuna instead.
    """
    model = lgb.LGBMClassifier(**_BASE_PARAMS, **params)
    model.fit(x, y)
    return model


def optimise(data: Dataset, n_trials: int = 40, use_wandb: bool = False) -> optuna.Study:
    """Optuna study using TimeSeriesSplit(5, gap=7) over the training DATES.

    See `purged_date_folds`: the split is taken on distinct dates, so the folds
    are ordered in time and the 7-day gap purges the label horizon.
    """
    x_train, y_train = data.xy("train")
    folds = list(purged_date_folds(data.train["date"]))
    run = _start_wandb(n_trials) if use_wandb else None

    for number, (fold_train, fold_valid) in enumerate(folds):
        logger.info(
            "CV fold %d: train %s..%s (%d rows) -> valid %s..%s (%d rows)",
            number,
            data.train["date"].iloc[fold_train].min().date(),
            data.train["date"].iloc[fold_train].max().date(),
            len(fold_train),
            data.train["date"].iloc[fold_valid].min().date(),
            data.train["date"].iloc[fold_valid].max().date(),
            len(fold_valid),
        )

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        fold_scores: list[float] = []

        for number, (fold_train, fold_valid) in enumerate(folds):
            # Augmentation is applied inside the fold, on training rows only,
            # so the masked copies can never leak into a validation fold. The
            # seed depends on the FOLD, not the trial: a per-trial seed would
            # redraw the masks for every candidate and make the objective noisy
            # in a way that has nothing to do with the hyperparameters.
            x_fold, y_fold = augment_missing_features(
                x_train.iloc[fold_train], y_train.iloc[fold_train], seed=RANDOM_SEED + number
            )
            model = _fit(params, x_fold, y_fold)
            predictions = model.predict(x_train.iloc[fold_valid])
            fold_scores.append(
                f1_score(
                    y_train.iloc[fold_valid],
                    predictions,
                    labels=list(range(len(TOR_LABELS))),
                    average="macro",
                    zero_division=0,
                )
            )

        score = float(np.mean(fold_scores))
        trial.set_user_attr("fold_macro_f1", [round(s, 4) for s in fold_scores])
        trial.set_user_attr("fold_std", round(float(np.std(fold_scores)), 4))
        if run is not None:
            run.log({"trial": trial.number, "cv_macro_f1": score, **_loggable(params)})
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        study_name="student_d_tor_lightgbm",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    if run is not None:
        run.summary["best_cv_macro_f1"] = study.best_value
        run.finish()

    logger.info(
        "Best CV macro-F1 %.4f (trial %d, fold sd %.4f)",
        study.best_value,
        study.best_trial.number,
        study.best_trial.user_attrs.get("fold_std", float("nan")),
    )
    return study


def _loggable(params: dict[str, Any]) -> dict[str, Any]:
    """Render params for W&B, keeping `class_weight=None` visible as a choice."""
    return {k: ("none" if v is None else v) for k, v in params.items()}


def _start_wandb(n_trials: int):
    """Start a Weights & Biases run if it is usable, else carry on without it.

    CONTRACT.md section 8 asks for trials to be logged to W&B. Without
    credentials the offline mode still produces a local run directory that can
    be synced later, so a missing API key degrades rather than failing training.
    """
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
            project="36120-at3-group08",
            group="student_d",
            name=f"tor-lightgbm-optuna-{n_trials}",
            config={"seed": RANDOM_SEED, "n_trials": n_trials, "search_space": SEARCH_SPACE},
            reinit=True,
        )
    except Exception as exc:  # noqa: BLE001 - logging must never break training
        logger.warning("Could not start a wandb run (%s) - continuing without it", exc)
        return None


# --------------------------------------------------------------------------
# Final fit and evaluation
# --------------------------------------------------------------------------


def selected_params(study: optuna.Study) -> dict[str, Any]:
    """The configuration that is actually fitted and deployed.

    This is `study.best_params` unchanged, which matters: `metrics.json`
    reports the same dictionary to GET /model-metadata, so what the service
    advertises is exactly what the artefact was trained with.
    """
    return dict(study.best_params)


def fit_final_model(data: Dataset, study: optuna.Study) -> lgb.LGBMClassifier:
    """Refit the selected configuration on train + validation, with augmentation.

    CONTRACT.md section 8 says the deployed model is refitted with the selected
    hyperparameters. It is refitted on train + validation and NOT on the test
    split, deliberately: the figures reported in the notebook then describe the
    artefact that is actually served, rather than a different model that has
    also seen the test period.
    """
    combined = pd.concat([data.train, data.validation], ignore_index=True)
    x = combined[data.feature_columns]
    y = combined["target_TOR"].astype(int)
    x_aug, y_aug = augment_missing_features(x, y)

    params = selected_params(study)
    logger.info(
        "Refitting on %d rows (%d after augmentation) with %d boosting rounds",
        len(x),
        len(x_aug),
        params["n_estimators"],
    )
    return _fit(params, x_aug, y_aug)


def expected_calibration_error(y_true, probabilities, n_bins: int = 10) -> float:
    """Expected Calibration Error - CONTRACT.md section 8 asks for calibration.

    The app shows a confidence score to users, so the gap between stated
    confidence and observed accuracy is a product-level concern, not only a
    statistical one.
    """
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == np.asarray(y_true)).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for low, high in pairwise(edges):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.sum() == 0:
            continue
        error += (in_bin.mean()) * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(error)


def predict_proba_all_labels(model: lgb.LGBMClassifier, x: pd.DataFrame) -> np.ndarray:
    """Probabilities as a (rows x 5) matrix covering every TOR label.

    `predict_proba` only returns columns for the classes the model was fitted
    on. If a rating is absent from the training window the matrix is narrower
    than the label set and `log_loss(..., labels=range(5))` raises on the shape
    mismatch - which is exactly what happens when the pipeline is run on a
    single token. Widening it here keeps the metric defined in that case.
    """
    narrow = model.predict_proba(x)
    full = np.zeros((len(narrow), len(TOR_LABELS)), dtype=float)
    full[:, model.classes_.astype(int)] = narrow
    # A class the model never saw gets probability 0, which would make log-loss
    # infinite the first time it is the truth. Floor it, then renormalise.
    full = np.clip(full, 1e-15, None)
    return full / full.sum(axis=1, keepdims=True)


def evaluate(model: lgb.LGBMClassifier, data: Dataset) -> dict[str, Any]:
    """Evaluate once on the held-out test split."""
    x_test, y_test = data.xy("test")
    predictions = model.predict(x_test)
    probabilities = predict_proba_all_labels(model, x_test)
    all_labels = list(range(len(TOR_LABELS)))

    report = classification_report(
        y_test,
        predictions,
        labels=all_labels,
        target_names=list(TOR_LABELS),
        output_dict=True,
        zero_division=0,
    )
    return {
        **_score(y_test, predictions),
        "log_loss": round(float(log_loss(y_test, probabilities, labels=all_labels)), 4),
        "ece": round(expected_calibration_error(y_test, probabilities), 4),
        "mean_confidence": round(float(probabilities.max(axis=1).mean()), 4),
        # Every rating is listed, including any absent from the test window, so
        # the gap is visible instead of silently dropped.
        "per_class_recall": {
            label: round(float(report[label]["recall"]), 4) for label in TOR_LABELS
        },
        "labels_in_test": [TOR_LABELS[c] for c in sorted(set(y_test))],
        "per_class": report,
        "confusion_matrix": {
            "labels": list(TOR_LABELS),
            "matrix": confusion_matrix(y_test, predictions, labels=all_labels).tolist(),
        },
        "per_token": _score_per_token(data.test, predictions),
        "degradation": degradation_table(model, data),
    }


def _score_per_token(test: pd.DataFrame, predictions) -> dict[str, dict[str, float]]:
    """Per-token scores, each over the same five labels so they can be ranked."""
    frame = test.assign(_predicted=predictions)
    return {
        str(token): {
            **_score(group["target_TOR"].astype(int), group["_predicted"]),
            "rows": len(group),
        }
        for token, group in frame.groupby("token", sort=True)
    }


def degradation_table(model: lgb.LGBMClassifier, data: Dataset) -> dict[str, dict[str, float]]:
    """Performance with each peer API down - CONTRACT.md section 7.

    Each peer's owned features are set to NaN across the whole test set and the
    model is re-scored, so the report can state what a user actually gets when
    Student A, B or C is unavailable.
    """
    x_test, y_test = data.xy("test")
    results = {"all_peers_available": _score(y_test, model.predict(x_test))}

    for peer, columns in PEER_FEATURE_GROUPS.items():
        present = [c for c in columns if c in x_test.columns]
        degraded = x_test.copy()
        degraded[present] = np.nan
        results[f"{peer}_down"] = _score(y_test, model.predict(degraded))

    everything = x_test.copy()
    for columns in PEER_FEATURE_GROUPS.values():
        everything[[c for c in columns if c in everything.columns]] = np.nan
    results["all_peers_down"] = _score(y_test, model.predict(everything))
    return results


def augmentation_ablation(data: Dataset, study: optuna.Study) -> dict[str, dict[str, float]]:
    """Quantify what the contract's missing-feature augmentation costs and buys.

    Augmentation is mandated by CONTRACT.md section 7, but it is not free: half
    the training rows carry blanked peer features, which slightly weakens the
    model when every peer is healthy. This refits the same hyperparameters with
    and without augmentation and scores both in the healthy case and with
    Student A (the most widely consumed peer) down, so the trade is reported as
    a measurement rather than an assertion.
    """
    combined = pd.concat([data.train, data.validation], ignore_index=True)
    x, y = combined[data.feature_columns], combined["target_TOR"].astype(int)
    x_test, y_test = data.xy("test")

    params = selected_params(study)

    degraded = x_test.copy()
    degraded[[c for c in PEER_FEATURE_GROUPS["student_a"] if c in degraded.columns]] = np.nan

    results: dict[str, dict[str, float]] = {}
    for name, (fit_x, fit_y) in {
        "trained_clean": (x, y),
        "trained_augmented": augment_missing_features(x, y),
    }.items():
        model = _fit(params, fit_x, fit_y)
        results[name] = {
            "macro_f1_all_peers_up": _score(y_test, model.predict(x_test))["macro_f1"],
            "macro_f1_student_a_down": _score(y_test, model.predict(degraded))["macro_f1"],
        }
    return results


def feature_importance(model: lgb.LGBMClassifier, features: list[str]) -> dict[str, float]:
    gains = model.booster_.feature_importance(importance_type="gain")
    total = float(gains.sum()) or 1.0
    return {
        name: round(float(gain) / total, 4)
        for name, gain in sorted(zip(features, gains), key=lambda kv: -kv[1])
    }


# --------------------------------------------------------------------------
# Artefacts
# --------------------------------------------------------------------------


def build_metrics(
    model: lgb.LGBMClassifier,
    data: Dataset,
    study: optuna.Study,
    metrics: dict[str, Any],
    tokens: list[str],
) -> dict[str, Any]:
    """metrics.json in the shape GET /model-metadata reads (CONTRACT.md 6.5)."""
    training_dates = pd.concat([data.train, data.validation])["date"]
    return {
        "model_version": MODEL_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "target": "Trading Opportunity Rating",
        "target_description": (
            "Five-class rating of the trading opportunity (Strong Sell .. Strong Buy) "
            "derived from the Opportunity Score OS = 0.40*M + 0.30*T + 0.20*V - 0.10*R, "
            "measured 7 days after the prediction date"
        ),
        "algorithm": "LightGBM Classifier",
        "forecast_horizon": FORECAST_HORIZON,
        "features": list(data.feature_columns),
        # Every entry in `features` must appear here, so GET /model-metadata can
        # never advertise an input with no stated provenance.
        "feature_sources": {
            "Return20": "Student A /features",
            "MA5": "Student B /features",
            "MA20": "Student B /features",
            "MA5_MA20_Ratio": "derived: Student B MA5 / MA20",
            "Volatility20": "Student C /features",
            "RelativeVolume": "owned by Student D",
            "OS": (
                "derived: today's Opportunity Score from Return20, MA5/MA20, "
                "RelativeVolume and Volatility20 - needs all three peers, so it "
                "is treated as a peer-owned input and masked when any peer is down"
            ),
        },
        "feature_importance": feature_importance(model, data.feature_columns),
        "classification_labels": list(TOR_LABELS),
        "supported_tokens": list(tokens),
        "metrics": {
            "test_macro_f1": metrics["macro_f1"],
            "test_balanced_accuracy": metrics["balanced_accuracy"],
            "test_accuracy": metrics["accuracy"],
            "test_log_loss": metrics["log_loss"],
            "test_ece": metrics["ece"],
        },
        "metrics_note": (
            "macro-F1 and weighted-F1 average over all five TOR labels, "
            "including any absent from the test window (F1 = 0 for those), so "
            "the model, the baselines and the per-token figures are comparable"
        ),
        "per_class_recall": metrics["per_class_recall"],
        "labels_in_test": metrics["labels_in_test"],
        "per_token": metrics["per_token"],
        "confusion_matrix": metrics["confusion_matrix"],
        "baselines": metrics.get("baselines", {}),
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
            "training period, so folds are ordered in time and the gap purges "
            "7 days; splitting the panel's rows would cut it by token instead"
        ),
        "search_space": SEARCH_SPACE,
        "n_trials": len(study.trials),
        "random_seed": RANDOM_SEED,
        "data_source": (
            "Kraken public REST API /0/public/OHLC (interval=1440), via the "
            "committed snapshot data/raw/kraken_daily_ohlcv.csv"
        ),
        "library_versions": {
            "lightgbm": lgb.__version__,
            "optuna": optuna.__version__,
            "pandas": pd.__version__,
        },
    }


def save_splits(data: Dataset) -> None:
    """Write the modelling matrices the notebook's section H also saves."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        suffix = {"train": "train", "validation": "val", "test": "test"}[split]
        x, y = data.xy(split)
        x.to_csv(PROCESSED_DIR / f"X_{suffix}.csv", index=False)
        y.to_frame("target_TOR").to_csv(PROCESSED_DIR / f"y_{suffix}.csv", index=False)
    logger.info("Wrote the six split files to %s", PROCESSED_DIR)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--tokens", nargs="*", default=None)
    parser.add_argument("--wandb", action="store_true", help="log trials to Weights & Biases")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-fetch from Kraken and rewrite the snapshot (default: use the snapshot)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    tokens = args.tokens or list(SUPPORTED_TOKENS)

    data, _ = load_dataset(tokens=tokens, refresh=args.refresh)
    baselines = score_baselines(data)
    for name, scores in baselines.items():
        logger.info("Baseline %-20s macro-F1 %.4f", name, scores["macro_f1"])

    study = optimise(data, n_trials=args.trials, use_wandb=args.wandb)
    model = fit_final_model(data, study)
    metrics = evaluate(model, data)
    metrics["baselines"] = baselines
    metrics["augmentation_ablation"] = augmentation_ablation(data, study)
    logger.info(
        "TEST macro-F1 %.4f | balanced acc %.4f | ECE %.4f",
        metrics["macro_f1"],
        metrics["balanced_accuracy"],
        metrics["ece"],
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_splits(data)

    payload = build_metrics(model, data, study, metrics, tokens)
    bundle = {
        "model": model,
        "feature_columns": list(data.feature_columns),
        "labels": dict(enumerate(TOR_LABELS)),
        **payload,
    }
    joblib.dump(bundle, MODEL_DIR / "tor_lightgbm.joblib", compress=3)
    (MODEL_DIR / "metrics.json").write_text(json.dumps(payload, indent=2, default=str))

    logger.info("Wrote %s", MODEL_DIR / "tor_lightgbm.joblib")
    logger.info("Wrote %s", MODEL_DIR / "metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
