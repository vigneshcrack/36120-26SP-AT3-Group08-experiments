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

Run:  python -m at3_crypto.modeling.student_d_tor --trials 60
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

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

from at3_crypto.engineered_features import FEATURE_OWNERS, engineer_features
from at3_crypto.kraken import SUPPORTED_TOKENS, KrakenClient
from at3_crypto.targets import FORECAST_HORIZON, TOR_LABELS, build_tor_target

logger = logging.getLogger(__name__)

RANDOM_SEED: Final = 42          # CONTRACT.md section 8
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
    "Return20", "RelativeVolume", "Volatility20", "MA5", "MA20", "MA5_MA20_Ratio",
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
MASK_PROBABILITY: Final = 0.35   # share of rows masked in the augmented copy
EARLY_STOPPING_ROUNDS: Final = 50

SEARCH_SPACE: Final[dict[str, str]] = {
    "learning_rate": "log-uniform [0.005, 0.30]",
    "num_leaves": "int [15, 255]",
    "max_depth": "int [3, 12]",
    "min_child_samples": "int [5, 120]",
    "subsample": "uniform [0.5, 1.0] (bagging every subsample_freq iterations)",
    "subsample_freq": "int [1, 7]",
    "colsample_bytree": "uniform [0.5, 1.0]",
    "reg_alpha": "log-uniform [1e-8, 10.0] (L1)",
    "reg_lambda": "log-uniform [1e-8, 10.0] (L2)",
    "min_split_gain": "uniform [0.0, 0.5]",
    "class_weight": "categorical [None, 'balanced']",
    "n_estimators": "int [150, 700]",
    "cv": "TimeSeriesSplit(n_splits=5, gap=7) on the training period only",
    "objective": "maximise mean macro-F1 across the 5 CV folds",
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
                "rows": int(len(frame)),
                "start": str(frame["date"].min().date()),
                "end": str(frame["date"].max().date()),
                "class_balance": {
                    TOR_LABELS[int(k)]: round(v, 4)
                    for k, v in frame["target_TOR"].value_counts(normalize=True).sort_index().items()
                },
            }
            for name, frame in (
                ("train", self.train), ("validation", self.validation), ("test", self.test)
            )
        }


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_dataset(
    tokens: list[str] | None = None, horizon: int = FORECAST_HORIZON
) -> tuple[Dataset, pd.DataFrame]:
    """Fetch from Kraken, engineer features, build the target and split."""
    tokens = tokens or list(SUPPORTED_TOKENS)
    raw = KrakenClient().fetch_many(tokens)

    features = engineer_features(raw)
    features["MA5_MA20_Ratio"] = features["MA5"] / features["MA20"]

    # build_tor_target adds OS (today's Opportunity Score), which is one of the
    # model features, so the target must be built before rows are dropped.
    modelling = build_tor_target(features, horizon=horizon)
    modelling = modelling.dropna(subset=[*FEATURE_COLUMNS, "target_TOR"])
    logger.info(
        "Modelling frame: %d rows, %s .. %s",
        len(modelling), modelling["date"].min().date(), modelling["date"].max().date(),
    )
    return _chronological_split(modelling, horizon), modelling


def _chronological_split(frame: pd.DataFrame, horizon: int) -> Dataset:
    """Split on shared dates, purging `horizon` days at each seam."""
    dates = np.sort(frame["date"].unique())
    train_end = pd.Timestamp(dates[int(len(dates) * TRAIN_FRACTION)])
    validation_end = pd.Timestamp(
        dates[int(len(dates) * (TRAIN_FRACTION + VALIDATION_FRACTION))]
    )
    purge = pd.Timedelta(horizon, unit="D")

    train = frame[frame["date"] <= train_end]
    validation = frame[(frame["date"] > train_end + purge) & (frame["date"] <= validation_end)]
    test = frame[frame["date"] > validation_end + purge]

    logger.info(
        "Split at %s / %s (purge %d days): %d / %d / %d rows",
        train_end.date(), validation_end.date(), horizon, len(train), len(validation), len(test),
    )
    return Dataset(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
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

    One extra copy per peer group is appended, each with that peer's features set
    to NaN in a random `MASK_PROBABILITY` share of rows.
    """
    rng = np.random.default_rng(seed)
    augmented = x.copy()

    # One extra copy, not one per peer: each row independently has a random
    # peer's features blanked with probability MASK_PROBABILITY. This covers
    # every peer-down pattern across the copy while only doubling the training
    # rows, instead of quadrupling them -- the fit has to stay fast enough that
    # teaching staff can re-run the notebook.
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

    for name, strategy in (("majority_class", "most_frequent"), ("stratified_random", "stratified")):
        dummy = DummyClassifier(strategy=strategy, random_state=RANDOM_SEED)
        dummy.fit(x_train, y_train)
        results[name] = _score(y_test, dummy.predict(x_test))

    # Persistence: today's TOR class is the forecast for t+7. The baseline to
    # beat, because ~half of ratings do not change over a week.
    persistence = data.test["TOR"].astype(int)
    results["persistence"] = _score(y_test, persistence)

    default = lgb.LGBMClassifier(
        objective="multiclass", num_class=len(TOR_LABELS),
        random_state=RANDOM_SEED, verbose=-1,
    )
    default.fit(x_train, y_train)
    results["lightgbm_defaults"] = _score(y_test, default.predict(x_test))
    return results


def _score(y_true, y_pred) -> dict[str, float]:
    labels = sorted(set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist()))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        balanced = float(balanced_accuracy_score(y_true, y_pred))
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(balanced, 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="weighted")), 4),
    }


# --------------------------------------------------------------------------
# Hyperparameter optimisation
# --------------------------------------------------------------------------

def _suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.30, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
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


def _fit(
    params: dict[str, Any],
    x: pd.DataFrame,
    y: pd.Series,
    eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
) -> lgb.LGBMClassifier:
    """Fit LightGBM, early-stopping on `eval_set` when one is supplied."""
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=len(TOR_LABELS),
        random_state=RANDOM_SEED, n_jobs=-1, verbose=-1, **params,
    )
    if eval_set is None:
        model.fit(x, y)
    else:
        model.fit(
            x, y, eval_set=[eval_set], eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
    return model


def optimise(data: Dataset, n_trials: int = 60, use_wandb: bool = False) -> optuna.Study:
    """Optuna study using TimeSeriesSplit(5, gap=7) on the training period."""
    x_train, y_train = data.xy("train")
    splitter = TimeSeriesSplit(n_splits=5, gap=PURGE_DAYS)
    run = _start_wandb(n_trials) if use_wandb else None

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        fold_scores: list[float] = []
        fold_rounds: list[int] = []

        for fold_train, fold_valid in splitter.split(x_train):
            # Augmentation is applied inside the fold, on training rows only,
            # so the masked copies can never leak into a validation fold.
            x_fold, y_fold = augment_missing_features(
                x_train.iloc[fold_train], y_train.iloc[fold_train], seed=RANDOM_SEED + trial.number
            )
            model = _fit(
                params, x_fold, y_fold,
                eval_set=(x_train.iloc[fold_valid], y_train.iloc[fold_valid]),
            )
            predictions = model.predict(x_train.iloc[fold_valid])
            fold_scores.append(
                f1_score(y_train.iloc[fold_valid], predictions, average="macro")
            )
            fold_rounds.append(int(model.best_iteration_ or params["n_estimators"]))

        score = float(np.mean(fold_scores))
        trial.set_user_attr("fold_macro_f1", [round(s, 4) for s in fold_scores])
        trial.set_user_attr("mean_best_iteration", int(np.mean(fold_rounds)))
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
        study.best_value, study.best_trial.number,
        study.best_trial.user_attrs.get("fold_std", float("nan")),
    )
    return study


def _loggable(params: dict[str, Any]) -> dict[str, Any]:
    return {k: ("balanced" if v == "balanced" else v) for k, v in params.items() if v is not None}


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
            project="36120-at3-group08", group="student_d",
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

def fit_final_model(data: Dataset, study: optuna.Study) -> lgb.LGBMClassifier:
    """Refit the selected configuration on train + validation, with augmentation."""
    combined = pd.concat([data.train, data.validation], ignore_index=True)
    x = combined[data.feature_columns]
    y = combined["target_TOR"].astype(int)
    x_aug, y_aug = augment_missing_features(x, y)

    params = dict(study.best_params)
    rounds = study.best_trial.user_attrs.get("mean_best_iteration")
    if rounds:
        params["n_estimators"] = max(50, rounds)
    logger.info(
        "Refitting on %d rows (%d augmented) with %d boosting rounds",
        len(x), len(x_aug), params["n_estimators"],
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
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.sum() == 0:
            continue
        error += (in_bin.mean()) * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(error)


def evaluate(model: lgb.LGBMClassifier, data: Dataset) -> dict[str, Any]:
    """Evaluate once on the held-out test split."""
    x_test, y_test = data.xy("test")
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    present = sorted(set(y_test) | set(predictions))

    report = classification_report(
        y_test, predictions, labels=present,
        target_names=[TOR_LABELS[c] for c in present], output_dict=True, zero_division=0,
    )
    return {
        **_score(y_test, predictions),
        "log_loss": round(float(log_loss(y_test, probabilities, labels=list(range(len(TOR_LABELS))))), 4),
        "ece": round(expected_calibration_error(y_test, probabilities), 4),
        "mean_confidence": round(float(probabilities.max(axis=1).mean()), 4),
        "per_class_recall": {
            TOR_LABELS[c]: round(float(report[TOR_LABELS[c]]["recall"]), 4) for c in present
        },
        "per_class": report,
        "confusion_matrix": {
            "labels": [TOR_LABELS[c] for c in present],
            "matrix": confusion_matrix(y_test, predictions, labels=present).tolist(),
        },
        "per_token": _score_per_token(data.test, predictions),
        "degradation": degradation_table(model, data),
    }


def _score_per_token(test: pd.DataFrame, predictions) -> dict[str, dict[str, float]]:
    frame = test.assign(_predicted=predictions)
    return {
        str(token): {**_score(group["target_TOR"].astype(int), group["_predicted"]),
                     "rows": int(len(group))}
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

    params = dict(study.best_params)
    rounds = study.best_trial.user_attrs.get("mean_best_iteration")
    if rounds:
        params["n_estimators"] = max(50, rounds)

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
    model: lgb.LGBMClassifier, data: Dataset, study: optuna.Study,
    metrics: dict[str, Any], tokens: list[str],
) -> dict[str, Any]:
    """metrics.json in the shape GET /model-metadata reads (CONTRACT.md 6.5)."""
    training_dates = pd.concat([data.train, data.validation])["date"]
    return {
        "model_version": MODEL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": "Trading Opportunity Rating",
        "target_description": (
            "Five-class rating of the trading opportunity (Strong Sell .. Strong Buy) "
            "derived from the Opportunity Score OS = 0.40*M + 0.30*T + 0.20*V - 0.10*R, "
            "measured 7 days after the prediction date"
        ),
        "algorithm": "LightGBM Classifier",
        "forecast_horizon": FORECAST_HORIZON,
        "features": list(data.feature_columns),
        "feature_sources": {
            "Return20": "Student A /features",
            "MA5": "Student B /features",
            "MA20": "Student B /features",
            "MA5_MA20_Ratio": "derived: Student B MA5 / MA20",
            "Volatility20": "Student C /features",
            "RelativeVolume": "owned by Student D",
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
        "per_class_recall": metrics["per_class_recall"],
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
        "best_params": study.best_params,
        "best_cv_macro_f1": round(float(study.best_value), 4),
        "cv_strategy": "TimeSeriesSplit(n_splits=5, gap=7) on the training period",
        "search_space": SEARCH_SPACE,
        "n_trials": len(study.trials),
        "random_seed": RANDOM_SEED,
        "data_source": "Kraken public REST API /0/public/OHLC (interval=1440)",
        "library_versions": {
            "lightgbm": lgb.__version__, "optuna": optuna.__version__, "pandas": pd.__version__,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--tokens", nargs="*", default=None)
    parser.add_argument("--wandb", action="store_true", help="log trials to Weights & Biases")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    tokens = args.tokens or list(SUPPORTED_TOKENS)

    data, modelling = load_dataset(tokens=tokens)
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
        metrics["macro_f1"], metrics["balanced_accuracy"], metrics["ece"],
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    payload = build_metrics(model, data, study, metrics, tokens)
    bundle = {
        "model": model,
        "feature_columns": list(data.feature_columns),
        "labels": dict(enumerate(TOR_LABELS)),
        **{k: v for k, v in payload.items() if k != "model"},
    }
    joblib.dump(bundle, MODEL_DIR / "tor_lightgbm.joblib", compress=3)
    (MODEL_DIR / "metrics.json").write_text(json.dumps(payload, indent=2, default=str))

    logger.info("Wrote %s", MODEL_DIR / "tor_lightgbm.joblib")
    logger.info("Wrote %s", MODEL_DIR / "metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
