"""Point the four-state raw-return experiment at the Nikkei 225 panel."""

from __future__ import annotations

from pathlib import Path

import run_sp500 as model


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
SOURCE = REPO / "data" / "nikkei225_daily_rebuilt" / "daily_returns.parquet"
CACHE = REPO / "data" / "arc_sm_joint_location_scale_hansen_parity_n225_input.npz"
FOUR_STATE_OUTPUT = ROOT / "output_n225"
FIXED_SHAPE_OUTPUT = ROOT / "output_n225_fixed_shape"


def configure() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"N225 raw-return panel not found: {SOURCE}")
    model.SOURCE = SOURCE
    model.CACHE = CACHE
    model.OUTPUT = FOUR_STATE_OUTPUT
