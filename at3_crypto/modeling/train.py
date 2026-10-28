"""Re-train the Student B Market Risk models from the hyperparameters chosen in the notebook.

The notebook is the record of the experiment (EDA, ablation, Optuna, evaluation). This
script re-fits the selected configuration without re-running the search, which is useful
for monthly retraining on a newer snapshot.

Usage:  uv run python -m at3_crypto.modeling.train --evaluate   # reproduce the test result
        uv run python -m at3_crypto.modeling.train              # refit on all labelled data
"""

import json
from pathlib import Path
from typing import Annotated

import joblib
import lightgbm as lgb
from loguru import logger
from mykrml_25740287.models.validation import chronological_split_with_gap
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import typer

from at3_crypto.config import MODELS_DIR
from at3_crypto.dataset import SNAPSHOT_PATH, load_snapshot
from at3_crypto.features import LEVEL_FEATURES, RESIDUAL_FEATURES, TARGET, build_panel, split_rows

SEED = 42
LGB_FIXED = {"random_state": SEED, "verbose": -1, "deterministic": True, "force_row_wise": True}
# features that disappear together when a peer API is down (derived features included)
PEER_GROUPS = {
    "Student A down": ["Return20", "MRI"],
    "Student C down": ["Volatility20", "GapRisk", "DR", "MRI"],
}
STUDENT_DIR = MODELS_DIR / "student_b"

app = typer.Typer(add_completion=False)


def augment_missing(X: pd.DataFrame, y: pd.Series, p: float = 0.25, seed: int = SEED):
    """Append a copy of the rows with each peer's features masked with probability p."""
    rng = np.random.default_rng(seed)
    X_copy = X.copy()
    for cols in PEER_GROUPS.values():
        cols = [c for c in cols if c in X_copy.columns]
        X_copy.loc[rng.random(len(X_copy)) < p, cols] = np.nan
    return pd.concat([X, X_copy], ignore_index=True), pd.concat([y, y], ignore_index=True)


def fit_models(rows: pd.DataFrame, metrics: dict):
    """Fit the primary (residual) and fallback (level) models with the saved settings."""
    hp = metrics["hyperparameters"]
    X_lvl, X_res, y = rows[LEVEL_FEATURES], rows[RESIDUAL_FEATURES], rows[TARGET]
    X_aug, y_aug = augment_missing(X_lvl.reset_index(drop=True), y.reset_index(drop=True))
    fallback = lgb.LGBMRegressor(**hp["fallback"], **LGB_FIXED).fit(X_aug, y_aug)
    primary = lgb.LGBMRegressor(**hp["primary"], **LGB_FIXED).fit(X_res, y - X_res["MRI"])
    return primary, fallback


def predict_primary(primary, shrink: float, X: pd.DataFrame) -> np.ndarray:
    """Forecast = MRI(t) + shrink x predicted 7-day change, clipped to the 0-100 scale."""
    return np.clip(X["MRI"].to_numpy() + shrink * primary.predict(X[RESIDUAL_FEATURES]), 0, 100)


@app.command()
def main(
    input_path: Path = SNAPSHOT_PATH,
    metrics_path: Path = STUDENT_DIR / "metrics.json",
    output_path: Path = STUDENT_DIR / "mri_lightgbm_retrained.joblib",
    evaluate: Annotated[
        bool, typer.Option(help="Fit on train+validation and score the test period once")
    ] = False,
):
    metrics = json.loads(metrics_path.read_text())
    labelled, _ = split_rows(build_panel(load_snapshot(input_path)))
    if evaluate:
        train, val, test = chronological_split_with_gap(
            labelled["date"], val_ratio=0.15, test_ratio=0.15, gap_days=7
        )
        primary, _ = fit_models(labelled[train | val], metrics)
        rows = labelled[test]
        mae = mean_absolute_error(rows[TARGET], predict_primary(primary, metrics["shrink"], rows))
        persistence = mean_absolute_error(rows[TARGET], rows["MRI"])
        logger.success(
            f"Test MAE {mae:.2f} vs persistence {persistence:.2f} ({1 - mae / persistence:.1%} better)"
        )
        return
    primary, fallback = fit_models(labelled, metrics)
    bundle = {
        "primary_model": primary,
        "fallback_model": fallback,
        "primary_formulation": "residual",
        "primary_features": RESIDUAL_FEATURES,
        "fallback_features": LEVEL_FEATURES,
        "shrink": metrics["shrink"],
        "model_version": metrics["model_version"],
    }
    joblib.dump(bundle, output_path)
    logger.success(
        f"Refitted on {len(labelled):,} rows; saved {output_path} (official bundle left unchanged)"
    )


if __name__ == "__main__":
    app()
