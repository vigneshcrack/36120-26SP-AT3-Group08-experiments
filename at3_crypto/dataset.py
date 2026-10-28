"""Download the Kraken daily-candle snapshot used by the experiments.

Usage:  uv run python -m at3_crypto.dataset            (all 8 tokens)
        uv run python -m at3_crypto.dataset --tokens BTC --tokens ETH

Kraken only returns the latest 720 daily candles, so a new download moves the window
forward; the notebooks therefore read the dated snapshot saved here, which keeps every
number in them reproducible.
"""

from pathlib import Path
from typing import Annotated

from loguru import logger
from mykrml_25740287.data.crypto import SUPPORTED_TOKENS, fetch_many_daily_ohlcv
import pandas as pd
import typer

from at3_crypto.config import RAW_DATA_DIR

SNAPSHOT_PATH = RAW_DATA_DIR / "kraken_daily_ohlcv.csv"

app = typer.Typer(add_completion=False)


def load_snapshot(path: Path = SNAPSHOT_PATH) -> pd.DataFrame:
    """Read the saved snapshot (one row per token and UTC day)."""
    return pd.read_csv(path, parse_dates=["date"])


@app.command()
def main(
    tokens: Annotated[
        list[str] | None, typer.Option(help="Tokens to download (default: all 8)")
    ] = None,
    output_path: Path = SNAPSHOT_PATH,
):
    """Download daily OHLCV candles (incomplete current day removed) and save them as CSV."""
    tokens = tokens or list(SUPPORTED_TOKENS)
    logger.info(f"Downloading {len(tokens)} tokens from the Kraken public OHLC API...")
    candles = fetch_many_daily_ohlcv(tokens)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candles.to_csv(output_path, index=False)
    logger.success(
        f"Saved {len(candles):,} rows ({candles['date'].min().date()} to "
        f"{candles['date'].max().date()}) to {output_path}"
    )


if __name__ == "__main__":
    app()
