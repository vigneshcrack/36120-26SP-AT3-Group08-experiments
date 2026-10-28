"""Forecast next week's Market Risk Index for every token with the saved model bundle.

Uses the same logic as the Student B FastAPI service (primary residual model when all
inputs are available), but runs offline on the saved snapshot.

Usage:  uv run python -m at3_crypto.modeling.predict
"""

from pathlib import Path

import joblib
from loguru import logger
import pandas as pd
import typer

from at3_crypto.config import MODELS_DIR
from at3_crypto.dataset import SNAPSHOT_PATH, load_snapshot
from at3_crypto.features import build_panel, split_rows
from at3_crypto.modeling.train import predict_primary

app = typer.Typer(add_completion=False)


def forecast_latest(bundle: dict, panel: pd.DataFrame) -> pd.DataFrame:
    """One row per token: latest complete day, today's MRI and the 7-day forecast."""
    _, latest = split_rows(panel)
    out = latest[["token", "date", "MRI"]].copy()
    out["forecast_t_plus_7"] = predict_primary(bundle["primary_model"], bundle["shrink"], latest)
    out["prediction_date"] = out["date"] + pd.to_timedelta(7, unit="D")
    return out.sort_values("forecast_t_plus_7", ascending=False).reset_index(drop=True)


@app.command()
def main(
    input_path: Path = SNAPSHOT_PATH,
    model_path: Path = MODELS_DIR / "student_b" / "mri_lightgbm.joblib",
):
    table = forecast_latest(joblib.load(model_path), build_panel(load_snapshot(input_path)))
    logger.info("\n" + table.round(2).to_string(index=False))


if __name__ == "__main__":
    app()
