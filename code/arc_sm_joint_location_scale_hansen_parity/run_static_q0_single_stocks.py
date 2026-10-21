"""Fitted-static-q0 symmetric-t four-model experiment for single stocks.

This is the literal single-observation-per-day counterpart of
run_static_q0_symmetric_t_four_models.py.  Location, log scale-squared, and
eta belong to the raw-return symmetric-t reference; they are not estimated in
a separate preprocessing model.  Reference state updates use only reference
scores.  The h block is fixed while the reference is fitted and is used only
as the fitted zero point of the subsequent parity DCD-SM.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import urllib.parse
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import run_sp500 as core
import run_static_q0_symmetric_t_four_models as q0
import run_symmetric_t_four_models as symmetric


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
CACHE = REPO / "data" / "global_index_daily_yahoo" / "_yahoo_cache"
OUTPUT_ROOT = ROOT / "output_static_q0_single_stocks"
SAMPLE_START = pd.Timestamp("2016-01-15")
TRAINING_END = pd.Timestamp("2023-12-29")
OOS_END = pd.Timestamp("2026-07-24")
ASSETS = {
    "amzn": {"ticker": "AMZN", "label": "Amazon AMZN"},
    "toyota": {"ticker": "7203.T", "label": "Toyota 7203.T"},
    "aapl": {"ticker": "AAPL", "label": "Apple AAPL"},
    "ntt": {"ticker": "9432.T", "label": "NTT 9432.T"},
}


def unix_timestamp(value: str) -> int:
    return int(
        dt.datetime.fromisoformat(value)
        .replace(tzinfo=dt.timezone.utc)
        .timestamp()
    )


def fetch_yahoo_cache(ticker: str, target: Path) -> dict[str, object]:
    """Fetch a missing adjusted-close cache without replacing existing data."""
    period1 = unix_timestamp("2015-12-01")
    period2 = unix_timestamp("2026-07-27")
    symbol = urllib.parse.quote(ticker, safe="-=^")
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={period1}&period2={period2}&interval=1d"
        "&events=history&includeAdjustedClose=true"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    result = payload.get("chart", {}).get("result")
    if not result:
        raise RuntimeError(f"Yahoo returned no history for {ticker}: {payload}")
    chart = result[0]
    metadata = chart.get("meta", {})
    timezone_name = metadata.get("exchangeTimezoneName")
    timezone = ZoneInfo(timezone_name) if timezone_name else dt.timezone.utc
    timestamps = chart.get("timestamp") or []
    quote = chart.get("indicators", {}).get("quote", [{}])[0]
    adjusted = chart.get("indicators", {}).get("adjclose") or [{}]
    closes = adjusted[0].get("adjclose") or quote.get("close") or []
    volumes = quote.get("volume") or []
    records = []
    for timestamp, close, volume in zip(timestamps, closes, volumes):
        if close is None or not math.isfinite(float(close)):
            continue
        local_date = dt.datetime.fromtimestamp(int(timestamp), tz=timezone).date()
        records.append(
            {
                "date": local_date.isoformat(),
                "adj_close": float(close),
                "volume": (
                    float(volume)
                    if volume is not None and math.isfinite(float(volume))
                    else None
                ),
            }
        )
    output = {
        "ticker": ticker,
        "records": records,
        "error": "",
        "source": url,
        "exchange_timezone": timezone_name,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output), encoding="utf-8")
    return output


def load_asset(asset: str) -> tuple[core.RawPanel, dict[str, object]]:
    specification = ASSETS[asset]
    ticker = str(specification["ticker"])
    cache_path = CACHE / f"{ticker}.json"
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        source = str(cache_path)
    else:
        print(f"{ticker}: local cache missing; downloading Yahoo history", flush=True)
        payload = fetch_yahoo_cache(ticker, cache_path)
        source = str(payload["source"])
    if payload.get("error"):
        raise RuntimeError(f"{ticker} cache error: {payload['error']}")
    prices = pd.DataFrame(payload["records"])
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values("date").drop_duplicates("date", keep="last")
    prices["log_return"] = np.log(prices["adj_close"]).diff()
    data = prices.loc[
        prices["date"].between(SAMPLE_START, OOS_END),
        ["date", "log_return"],
    ].dropna().reset_index(drop=True)
    if data.empty or not np.isfinite(data["log_return"]).all():
        raise RuntimeError(f"{ticker} return sample is empty or non-finite")
    n_train = int(np.sum(data["date"] <= TRAINING_END))
    if n_train <= 22 or n_train == len(data):
        raise RuntimeError(f"{ticker} invalid split: {n_train}/{len(data)}")
    values = data["log_return"].to_numpy(dtype=float)
    offsets = np.arange(values.size + 1, dtype=int)
    panel = core.RawPanel(
        daily=pd.DataFrame(
            {"date": data["date"], "n_constituents": np.ones(values.size, dtype=int)}
        ),
        values=values,
        offsets=offsets,
    )
    return panel, {
        "asset": asset,
        "ticker": ticker,
        "label": specification["label"],
        "cache": str(cache_path),
        "source_used": source,
    }


def continue_reference(
    panel: core.RawPanel,
    n_train: int,
    anchor: np.ndarray,
    betas: np.ndarray,
    training: core.FourStateFilter,
) -> tuple[core.FourStateFilter, core.FourStateFilter, float]:
    training_rows = int(panel.offsets[n_train])
    oos_offsets = panel.offsets[n_train:] - training_rows
    oos = core.filter_reference(
        panel.values[training_rows:],
        oos_offsets,
        anchor,
        betas,
        initial_state=training.terminal_state,
        initial_score_history=training.terminal_score_history,
    )
    full = core.filter_reference(panel.values, panel.offsets, anchor, betas)
    error = float(
        max(
            np.max(np.abs(full.states[n_train:] - oos.states)),
            np.max(np.abs(full.contributions[n_train:] - oos.contributions)),
        )
    )
    if error > 2e-12:
        raise RuntimeError(f"reference continuation error: {error}")
    return oos, full, error


def arc_parameter_rows(specification: str, fit: dict[str, object]):
    return [
        {
            "specification": specification,
            "block": block,
            "horizon": horizon,
            "estimate": estimate,
            "hac22_se_conditional_on_reference": se,
            "p_two_sided": p,
        }
        for block, horizon, estimate, se, p in zip(
            ["even"] * 3 + ["odd"] * 3,
            [*core.WINDOWS, *core.WINDOWS],
            [*fit["kappa_plus"], *fit["kappa_minus"]],
            fit["se_godambe_hac22_conditional_on_hansen"],
            fit["p_two_sided_conditional_on_hansen"],
        )
    ]


def run_asset(asset: str) -> None:
    panel, asset_info = load_asset(asset)
    label = str(asset_info["label"])
    output = OUTPUT_ROOT / asset
    output.mkdir(parents=True, exist_ok=True)
    dates = pd.DatetimeIndex(panel.daily["date"])
    n_train = int(np.sum(dates <= TRAINING_END))
    training_rows = int(panel.offsets[n_train])
    train_offsets = panel.offsets[: n_train + 1]
    oos_offsets = panel.offsets[n_train:] - training_rows
    training_values = panel.values[:training_rows]

    print(f"{label}: fitting static symmetric-t", flush=True)
    anchor, static_t = symmetric.fit_static_symmetric_t(
        training_values, train_offsets
    )
    knots = core.make_clamped_knots(core.BREAKPOINTS, core.DEGREE)
    masses = core.bspline_masses(knots, core.DEGREE)
    helmert = core.helmert_submatrix(masses.size)
    print(f"{label}: fitting conditional static q0", flush=True)
    weights0, static_q0 = q0.fit_static_q0(
        training_values, train_offsets, anchor, knots, masses, helmert
    )

    print(f"{label}: fitting dynamic location-scale, eta fixed", flush=True)
    fixed_betas, fixed_summary, fixed_train = (
        symmetric.fixed.fit_dynamic_location_scale(
            training_values, train_offsets, anchor
        )
    )
    fixed_oos, fixed_full, fixed_continuation = continue_reference(
        panel, n_train, anchor, fixed_betas, fixed_train
    )

    print(f"{label}: fitting dynamic location-scale-eta", flush=True)
    dynamic_betas, dynamic_summary, dynamic_train, dynamic_inference = (
        symmetric.fit_dynamic_symmetric_t(
            training_values, train_offsets, anchor, fixed_betas
        )
    )
    dynamic_oos, dynamic_full, dynamic_continuation = continue_reference(
        panel, n_train, anchor, dynamic_betas, dynamic_train
    )
    if max(
        np.max(np.abs(fixed_full.states[:, 3])),
        np.max(np.abs(dynamic_full.states[:, 3])),
    ) > 1e-15:
        raise RuntimeError("lambda moved away from zero")

    fixed_u = core.dynamic_reference_pit(panel.values, panel.offsets, fixed_full.states)
    dynamic_u = core.dynamic_reference_pit(
        panel.values, panel.offsets, dynamic_full.states
    )
    fixed_phi = core.normalized_bspline_basis(fixed_u, knots, core.DEGREE, masses)
    dynamic_phi = core.normalized_bspline_basis(dynamic_u, knots, core.DEGREE, masses)
    fixed_geometry = q0.corrected_tangent_geometry(
        anchor,
        weights0,
        knots,
        masses,
        helmert,
        (0, 1),
        "modular reference location/log-scale-squared update scores",
    )
    dynamic_geometry = q0.corrected_tangent_geometry(
        anchor,
        weights0,
        knots,
        masses,
        helmert,
        (0, 1, 2),
        "modular reference location/log-scale-squared/eta update scores",
    )

    print(f"{label}: fitting fixed-eta parity DCD-SM around q0", flush=True)
    fixed_arc, _, fixed_correction_oos, fixed_arc_checks = q0.fit_arc_sm(
        fixed_phi,
        train_offsets,
        oos_offsets,
        fixed_train,
        fixed_oos,
        fixed_full,
        n_train,
        weights0,
        helmert,
        fixed_geometry,
        (0, 1),
    )
    print(f"{label}: fitting dynamic-eta parity DCD-SM around q0", flush=True)
    dynamic_arc, _, dynamic_correction_oos, dynamic_arc_checks = q0.fit_arc_sm(
        dynamic_phi,
        train_offsets,
        oos_offsets,
        dynamic_train,
        dynamic_oos,
        dynamic_full,
        n_train,
        weights0,
        helmert,
        dynamic_geometry,
        (0, 1, 2),
    )

    fixed_scores = q0.score_oos_pair(
        panel.values[training_rows:],
        oos_offsets,
        dates[n_train:],
        training_values,
        fixed_oos,
        fixed_correction_oos,
        knots,
        masses,
        weights0,
        q0.MODEL_NAMES["fixed"],
        q0.MODEL_NAMES["fixed_arc"],
    )
    dynamic_scores = q0.score_oos_pair(
        panel.values[training_rows:],
        oos_offsets,
        dates[n_train:],
        training_values,
        dynamic_oos,
        dynamic_correction_oos,
        knots,
        masses,
        weights0,
        q0.MODEL_NAMES["dynamic"],
        q0.MODEL_NAMES["dynamic_arc"],
    )
    scores = pd.concat((fixed_scores, dynamic_scores), ignore_index=True)
    summary = scores.groupby("model", as_index=False)[list(q0.SCORE_NAMES)].mean()
    pairs = (
        (q0.MODEL_NAMES["fixed_arc"], q0.MODEL_NAMES["fixed"]),
        (q0.MODEL_NAMES["dynamic_arc"], q0.MODEL_NAMES["dynamic"]),
        (q0.MODEL_NAMES["fixed"], q0.MODEL_NAMES["dynamic"]),
        (q0.MODEL_NAMES["fixed_arc"], q0.MODEL_NAMES["dynamic_arc"]),
    )
    comparisons = [
        core.score_comparison(scores, proposed, benchmark, score)
        for proposed, benchmark in pairs
        for score in q0.SCORE_NAMES
    ]
    arc_parameters = pd.DataFrame(
        arc_parameter_rows("fixed_eta", fixed_arc)
        + arc_parameter_rows("dynamic_eta", dynamic_arc)
    )
    dynamic_parameters = pd.DataFrame(dynamic_inference["parameters"])

    scores.to_csv(output / "daily_oos_scores.csv", index=False)
    summary.to_csv(output / "four_model_oos_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(
        output / "four_model_oos_comparisons.csv", index=False
    )
    arc_parameters.to_csv(output / "arc_sm_parameters.csv", index=False)
    dynamic_parameters.to_csv(output / "dynamic_t_parameters.csv", index=False)
    pd.DataFrame(
        {
            "component": np.arange(masses.size),
            "uniform_chart_weight": masses,
            "fitted_static_q0_weight": weights0,
        }
    ).to_csv(output / "static_q0_weights.csv", index=False)

    result = {
        "analysis": f"{label} raw-return symmetric-t four-model comparison around fitted static q0",
        "asset": asset_info,
        "sample": {
            "start": str(dates[0].date()),
            "training_end": str(dates[n_train - 1].date()),
            "oos_start": str(dates[n_train].date()),
            "end": str(dates[-1].date()),
            "training_observations": n_train,
            "oos_observations": int(len(dates) - n_train),
        },
        "specification": {
            "raw_return_joint_reference": True,
            "daily_cluster_size": 1,
            "reference_score": "reference-only; h side excluded",
            "static_q0": "conditional day-equal MLE after static symmetric-t MLE",
            "arc_chart": "w(q)=softmax(log(w0)+Hq)",
            "arc_projection": "off modular reference update scores",
            "residual_score_whitening": False,
            "parity_windows": list(core.WINDOWS),
        },
        "static_symmetric_t": static_t,
        "static_q0": static_q0,
        "fixed_eta_dynamic_location_scale": fixed_summary,
        "fixed_eta_arc_sm": fixed_arc,
        "dynamic_location_scale_eta": dynamic_summary,
        "dynamic_location_scale_eta_inference": dynamic_inference,
        "dynamic_eta_arc_sm": dynamic_arc,
        "geometries": {
            "fixed_eta_arc_sm": fixed_geometry.diagnostics,
            "dynamic_eta_arc_sm": dynamic_geometry.diagnostics,
        },
        "oos_summary": summary.to_dict(orient="records"),
        "oos_comparisons": comparisons,
        "checks": {
            "lambda_fixed_maximum_absolute_state": float(
                max(
                    np.max(np.abs(fixed_full.states[:, 3])),
                    np.max(np.abs(dynamic_full.states[:, 3])),
                )
            ),
            "fixed_reference_continuation": fixed_continuation,
            "dynamic_reference_continuation": dynamic_continuation,
            "fixed_eta_arc_sm": fixed_arc_checks,
            "dynamic_eta_arc_sm": dynamic_arc_checks,
        },
    }
    (output / "estimation.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    q0.write_report(
        output, result, summary, pd.DataFrame(comparisons), arc_parameters
    )
    print(summary.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", choices=(*ASSETS, "all"), default="all")
    args = parser.parse_args()
    assets = tuple(ASSETS) if args.asset == "all" else (args.asset,)
    for asset in assets:
        run_asset(asset)


if __name__ == "__main__":
    main()
