"""Reusable report figures for the Market Risk Index experiment.

Usage:  uv run python -m at3_crypto.plots --token BTC
"""

from pathlib import Path

from loguru import logger
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import typer

from at3_crypto.config import FIGURES_DIR
from at3_crypto.dataset import load_snapshot
from at3_crypto.features import build_panel

app = typer.Typer(add_completion=False)


def plot_mri_history(panel, token: str, output_path: Path) -> Path:
    """Close price and same-day Market Risk Index for one token, on a shared date axis."""
    data = panel[panel["token"] == token]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    ax1.plot(data["date"], data["close"], color="#0d366b")
    ax1.set_ylabel(f"{token} close (USD)")
    ax2.plot(data["date"], data["MRI"], color="#e34948")
    ax2.set_ylabel("Market Risk Index")
    ax2.set_ylim(0, 100)
    fig.suptitle(f"{token}: price and Market Risk Index")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


@app.command()
def main(token: str = "BTC", output_dir: Path = FIGURES_DIR / "student_b"):
    """Save a price + risk history figure for one token."""
    path = plot_mri_history(
        build_panel(load_snapshot()),
        token.upper(),
        output_dir / f"mri_history_{token.lower()}.png",
    )
    logger.success(f"Saved {path}")


if __name__ == "__main__":
    app()
