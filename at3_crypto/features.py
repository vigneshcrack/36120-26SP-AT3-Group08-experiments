"""Build the labelled Market Risk Index (Student B) dataset from the raw snapshot.

All features come from the group's shared package, so the experiments, the FastAPI
services and this module compute them identically (CONTRACT.md §4).

Usage:  uv run python -m at3_crypto.features
"""

from pathlib import Path

from loguru import logger
from mykrml_25740287.features.crypto import (
    FORECAST_HORIZON_DAYS,
    compute_features_panel,
    compute_market_risk_index,
    forward_target,
)
import pandas as pd
import typer

from at3_crypto.config import INTERIM_DATA_DIR
from at3_crypto.dataset import SNAPSHOT_PATH, load_snapshot

TARGET = "MRI_t_plus_7"
LEVEL_FEATURES = ["Return20", "Volatility20", "DR", "GapRisk"]
RESIDUAL_FEATURES = LEVEL_FEATURES + ["MRI"]
REQUIRED = ["Return20", "Volatility20", "MaxPrice20", "GapRisk", "DR", "TrendStrength", "MRI"]

app = typer.Typer(add_completion=False)


def build_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """Contract features, DR, same-day MRI and the t+7 label for every token and day."""
    panel = compute_features_panel(raw)
    panel["DR"] = (panel["MaxPrice20"] - panel["close"]) / panel["MaxPrice20"]
    panel["MRI"] = compute_market_risk_index(panel, panel["close"])
    panel[TARGET] = forward_target(
        panel["MRI"], horizon=FORECAST_HORIZON_DAYS, groups=panel["token"]
    )
    return panel


def split_rows(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(labelled rows for modelling, latest unlabelled rows that production predicts on)."""
    has_features = panel[REQUIRED].notna().all(axis=1)
    labelled = panel[has_features & panel[TARGET].notna()].reset_index(drop=True)
    latest = (
        panel[has_features & panel[TARGET].isna()].groupby("token").tail(1).reset_index(drop=True)
    )
    return labelled, latest


@app.command()
def main(
    input_path: Path = SNAPSHOT_PATH,
    output_path: Path = INTERIM_DATA_DIR / "mri_dataset.csv",
):
    """Write the labelled MRI dataset (warm-up and unlabelled rows removed)."""
    labelled, latest = split_rows(build_panel(load_snapshot(input_path)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labelled.to_csv(output_path, index=False)
    logger.success(
        f"Saved {len(labelled):,} labelled token-days to {output_path} "
        f"({len(latest)} latest rows kept aside for prediction)"
    )


if __name__ == "__main__":
    app()
