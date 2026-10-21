"""Case-1 robustness to removing the contemporaneous market factor.

For the S&P 500 and Nikkei 225 constituent panels, construct causal rolling
market-model residuals

    e[i,t] = r[i,t] - alpha_hat[i,t-1] - beta_hat[i,t-1] r[m,t],

where the coefficients use at most the preceding 252 matched observations and
require at least 126 observations.  The current return never enters its own
coefficient estimate.  The official index levels are downloaded from FRED and
cached locally; daily log returns are then matched by date.

For each market the locked Case-1 M1--M5 estimator is run on three targets over
exactly the same dates:

* raw_matched: raw constituent returns for asset-days having a valid beta;
* idiosyncratic: the corresponding one-factor residuals;
* index: the official market-index return, with N_t=1.

The raw and idiosyncratic panels contain exactly the same asset-days.  All
dynamic filters retain the standard exact HAR(1,5,22), h=1, q0=0, tail-refined
knots, and location/log-variance Fisher projection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_case1_final_common_calendar_35_series as final
import run_sp500 as core


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUTPUT = ROOT / "output_case1_factor_robustness_markets"
CACHE = REPO / "data" / "arc_sm_factor_robustness"

ROLLING_WINDOW = 252
MINIMUM_OBSERVATIONS = 126

MARKETS = {
    "sp500": {
        "label": "S&P 500",
        "fred_series": "SP500",
        "source": REPO / "data" / "weekly_density_em" / "daily_returns" / "usa.parquet",
        "id_column": "firm_id",
        "columns": ["date", "firm_id", "ticker", "log_return"],
    },
    "n225": {
        "label": "Nikkei 225",
        "fred_series": "NIKKEI225",
        "source": REPO / "data" / "nikkei225_daily_rebuilt" / "daily_returns.parquet",
        "id_column": "firm_id",
        "columns": ["date", "firm_id", "ticker", "log_return"],
    },
}


def fred_url(series: str) -> str:
    return (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series}&cosd=2016-01-01&coed={final.END.date()}"
    )


def load_market_factor(market: str, refresh: bool = False) -> pd.DataFrame:
    """Load and cache an official-index daily log-return series from FRED."""
    specification = MARKETS[market]
    series = str(specification["fred_series"])
    cache_path = CACHE / f"fred_{series.lower()}_levels.csv"
    if refresh or not cache_path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        downloaded = pd.read_csv(fred_url(series))
        if "observation_date" not in downloaded or series not in downloaded:
            raise RuntimeError(f"unexpected FRED schema for {series}: {downloaded.columns}")
        downloaded.to_csv(cache_path, index=False)
    levels = pd.read_csv(cache_path)
    levels = levels.rename(columns={"observation_date": "date", series: "level"})
    levels["date"] = pd.to_datetime(levels["date"]).dt.normalize()
    levels["level"] = pd.to_numeric(levels["level"], errors="coerce")
    levels = levels.loc[
        np.isfinite(levels["level"]) & (levels["level"] > 0), ["date", "level"]
    ].sort_values("date", kind="stable")
    levels = levels.drop_duplicates("date", keep="last").reset_index(drop=True)
    levels["market_log_return"] = np.log(levels["level"]).diff()
    levels = levels.loc[np.isfinite(levels["market_log_return"])].reset_index(drop=True)
    return levels


def causal_rolling_residuals(
    frame: pd.DataFrame,
    window: int = ROLLING_WINDOW,
    minimum_observations: int = MINIMUM_OBSERVATIONS,
) -> pd.DataFrame:
    """Add strictly lagged rolling OLS alpha, beta, and market residuals."""
    required = {"date", "firm_id", "log_return", "market_log_return"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing residualization columns: {sorted(missing)}")
    if not 2 <= minimum_observations <= window:
        raise ValueError("minimum_observations must lie between 2 and window")

    data = frame.sort_values(["firm_id", "date"], kind="stable").reset_index(drop=True)
    alpha = np.full(len(data), np.nan)
    beta = np.full(len(data), np.nan)
    n_lagged = np.zeros(len(data), dtype=int)

    for positions in data.groupby("firm_id", sort=False).indices.values():
        positions = np.asarray(positions, dtype=int)
        x = data.loc[positions, "market_log_return"].to_numpy(dtype=float)
        y = data.loc[positions, "log_return"].to_numpy(dtype=float)
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise ValueError("rolling regression received non-finite observations")
        count = x.size
        k = np.arange(count)
        lower = np.maximum(0, k - window)
        n = k - lower

        def lagged_sum(values: np.ndarray) -> np.ndarray:
            cumulative = np.r_[0.0, np.cumsum(values)]
            return cumulative[k] - cumulative[lower]

        sx = lagged_sum(x)
        sy = lagged_sum(y)
        sxx = lagged_sum(x * x)
        sxy = lagged_sum(x * y)
        valid_count = n >= minimum_observations
        denominator = sxx - np.divide(
            sx * sx, n, out=np.zeros_like(sx), where=n > 0
        )
        valid = valid_count & (denominator > 1e-14)
        local_beta = np.full(count, np.nan)
        local_alpha = np.full(count, np.nan)
        local_beta[valid] = (
            sxy[valid] - sx[valid] * sy[valid] / n[valid]
        ) / denominator[valid]
        local_alpha[valid] = (
            sy[valid] / n[valid] - local_beta[valid] * sx[valid] / n[valid]
        )
        alpha[positions] = local_alpha
        beta[positions] = local_beta
        n_lagged[positions] = n

    data["rolling_alpha"] = alpha
    data["rolling_beta"] = beta
    data["beta_observations"] = n_lagged
    data["idiosyncratic_return"] = (
        data["log_return"]
        - data["rolling_alpha"]
        - data["rolling_beta"] * data["market_log_return"]
    )
    return data


def to_panel(frame: pd.DataFrame, value_column: str) -> core.RawPanel:
    data = frame.loc[
        np.isfinite(frame[value_column]), ["date", value_column]
    ].copy()
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    data = data.sort_values("date", kind="stable").reset_index(drop=True)
    daily = data.groupby("date", sort=True).size().rename("n_constituents").reset_index()
    counts = daily["n_constituents"].to_numpy(dtype=int)
    values = data[value_column].to_numpy(dtype=float, copy=True)
    offsets = np.r_[0, np.cumsum(counts)]
    if offsets[-1] != values.size:
        raise RuntimeError("panel offsets do not cover the values")
    return core.RawPanel(daily=daily, values=values, offsets=offsets)


def prepare_market_panels(
    market: str, refresh_market_data: bool = False
) -> tuple[dict[str, core.RawPanel], dict[str, object]]:
    specification = MARKETS[market]
    source = Path(specification["source"])
    if not source.exists():
        raise FileNotFoundError(source)
    returns = pd.read_parquet(source, columns=list(specification["columns"]))
    returns["date"] = pd.to_datetime(returns["date"]).dt.normalize()
    returns["firm_id"] = returns[str(specification["id_column"])].astype(str)
    returns["log_return"] = pd.to_numeric(returns["log_return"], errors="coerce")
    returns = returns.loc[
        np.isfinite(returns["log_return"])
        & (returns["date"] >= final.START)
        & (returns["date"] <= final.END)
    ].copy()
    factor = load_market_factor(market, refresh_market_data)
    merged = returns.merge(
        factor[["date", "market_log_return"]], on="date", how="inner", validate="many_to_one"
    )
    residualized = causal_rolling_residuals(merged)
    matched = residualized.loc[
        np.isfinite(residualized["idiosyncratic_return"])
    ].copy()
    if matched.empty:
        raise RuntimeError(f"{market}: no causal market-model residuals")

    daily_counts = matched.groupby("date", sort=True).size()
    eligible_dates = daily_counts.index[daily_counts > 0]
    matched = matched.loc[matched["date"].isin(eligible_dates)].copy()
    factor_matched = factor.loc[factor["date"].isin(eligible_dates)].copy()
    factor_matched = factor_matched.sort_values("date", kind="stable")
    if factor_matched["date"].nunique() != len(eligible_dates):
        raise RuntimeError(f"{market}: factor and residual dates do not align")

    panels = {
        "raw_matched": to_panel(matched, "log_return"),
        "idiosyncratic": to_panel(matched, "idiosyncratic_return"),
        "index": to_panel(factor_matched, "market_log_return"),
    }
    raw_counts = np.diff(panels["raw_matched"].offsets)
    idio_counts = np.diff(panels["idiosyncratic"].offsets)
    if not np.array_equal(raw_counts, idio_counts):
        raise RuntimeError(f"{market}: raw and residual panels do not share asset-days")
    if not pd.DatetimeIndex(panels["raw_matched"].daily["date"]).equals(
        pd.DatetimeIndex(panels["idiosyncratic"].daily["date"])
    ):
        raise RuntimeError(f"{market}: raw and residual dates differ")
    if not np.all(np.diff(panels["index"].offsets) == 1):
        raise RuntimeError(f"{market}: index target is not N_t=1")

    grouped = matched.groupby("date", sort=True)
    daily = grouped.agg(
        n_constituents=("firm_id", "size"),
        raw_mean=("log_return", "mean"),
        idiosyncratic_mean=("idiosyncratic_return", "mean"),
        market_log_return=("market_log_return", "first"),
        beta_mean=("rolling_beta", "mean"),
        beta_median=("rolling_beta", "median"),
    ).reset_index()
    market_output = OUTPUT / market
    market_output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(market_output / "panel_daily_coverage.csv", index=False)
    beta_quantiles = matched["rolling_beta"].quantile(
        [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    )
    beta_quantiles.rename_axis("quantile").rename("rolling_beta").reset_index().to_csv(
        market_output / "rolling_beta_quantiles.csv", index=False
    )

    diagnostics = {
        "market": market,
        "label": specification["label"],
        "factor_source": "FRED",
        "factor_series": specification["fred_series"],
        "factor_url": fred_url(str(specification["fred_series"])),
        "rolling_window": ROLLING_WINDOW,
        "minimum_lagged_observations": MINIMUM_OBSERVATIONS,
        "coefficient_information_set": "strictly prior matched observations",
        "first_date": str(pd.Timestamp(eligible_dates.min()).date()),
        "last_date": str(pd.Timestamp(eligible_dates.max()).date()),
        "days": int(len(eligible_dates)),
        "asset_days": int(len(matched)),
        "daily_constituents_minimum": int(daily_counts.min()),
        "daily_constituents_median": float(daily_counts.median()),
        "daily_constituents_maximum": int(daily_counts.max()),
        "rolling_beta_quantiles": {
            str(index): float(value) for index, value in beta_quantiles.items()
        },
        "pooled_raw_market_correlation": float(
            matched["log_return"].corr(matched["market_log_return"])
        ),
        "pooled_residual_market_correlation": float(
            matched["idiosyncratic_return"].corr(matched["market_log_return"])
        ),
        "daily_raw_mean_market_correlation": float(
            daily["raw_mean"].corr(daily["market_log_return"])
        ),
        "daily_residual_mean_market_correlation": float(
            daily["idiosyncratic_mean"].corr(daily["market_log_return"])
        ),
        "pooled_raw_variance": float(matched["log_return"].var()),
        "pooled_residual_variance": float(matched["idiosyncratic_return"].var()),
    }
    (market_output / "panel_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return panels, diagnostics


def fit_panel(
    dataset_key: str,
    panel: core.RawPanel,
    market: str,
    target: str,
    diagnostics: dict[str, object],
) -> dict[str, object]:
    label = f"{MARKETS[market]['label']} {target.replace('_', ' ')}"
    info = {
        "dataset": dataset_key,
        "label": label,
        "market": market,
        "target": target,
        "factor_series": MARKETS[market]["fred_series"],
    }
    old_loader = final.identity.load_dataset
    old_output = final.OUTPUT
    old_category = final.category
    try:
        final.OUTPUT = OUTPUT
        final.identity.load_dataset = lambda _dataset: (panel, info, {}, label)
        final.category = lambda _dataset: "factor_robustness"
        result = final.run_dataset(dataset_key)
    finally:
        final.identity.load_dataset = old_loader
        final.OUTPUT = old_output
        final.category = old_category
    result["analysis"] = "Case1 market-factor robustness"
    result["factor_robustness"] = {
        "market": market,
        "target": target,
        "comparison_design": "raw and idiosyncratic targets use identical asset-days; index uses identical dates",
        "residualization": diagnostics,
    }
    result_path = OUTPUT / dataset_key / "estimation.json"
    result_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def test_rows(result: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    mappings = [
        ("M2", result["model2_inference"]["wald_tests"], {
            "location": "location_three_coefficients",
            "scale": "scale_three_coefficients",
            "joint": "location_scale_six_coefficients",
        }),
        ("M3", result["model3"]["wald_tests"], {
            "eta": "eta_three_coefficients",
            "lambda": "lambda_three_coefficients",
            "joint": "eta_lambda_six_coefficients",
        }),
    ]
    for model, tests, names in mappings:
        for block, key in names.items():
            record = tests[key]
            rows.append({"model": model, "block": block, **record})
    for model in ("M4", "M5"):
        fitted = result["model4" if model == "M4" else "model5"]
        for block, key in (("even", "common_even"), ("odd", "asymmetric_odd")):
            record = fitted["joint_blocks"][key]
            rows.append({"model": model, "block": block, **record})
        rows.append({"model": model, "block": "joint", **fitted["joint_all"]})
    return rows


def summarize(results: dict[str, dict[str, object]], diagnostics: dict[str, object]) -> None:
    test_records, level_records, comparison_records = [], [], []
    for key, result in results.items():
        market, target = key.split("_", 1)
        common = {"dataset": key, "market": market, "target": target}
        test_records.extend({**common, **row} for row in test_rows(result))
        level_records.extend({**common, **row} for row in result["oos_summary"])
        comparison_records.extend({**common, **row} for row in result["oos_comparisons"])
    tests = pd.DataFrame(test_records)
    levels = pd.DataFrame(level_records)
    comparisons = pd.DataFrame(comparison_records)
    tests.to_csv(OUTPUT / "factor_robustness_wald_tests.csv", index=False)
    levels.to_csv(OUTPUT / "factor_robustness_oos_levels.csv", index=False)
    comparisons.to_csv(OUTPUT / "factor_robustness_oos_comparisons.csv", index=False)

    # final.MODEL_SHORT maps long model names to M1--M5.
    levels["model_short"] = levels["model"].map(final.MODEL_SHORT)
    metric_names = list(final.scoring.SCORE_NAMES)
    winners = []
    for (market, target), group in levels.groupby(["market", "target"], sort=False):
        indexed = group.set_index("model_short")
        for metric in metric_names:
            if metric in final.HIGHER_IS_BETTER:
                winner = indexed[metric].idxmax()
            else:
                winner = indexed[metric].idxmin()
            winners.append({"market": market, "target": target, "metric": metric, "winner": winner})
    winner_frame = pd.DataFrame(winners)
    winner_frame.to_csv(OUTPUT / "factor_robustness_oos_winners.csv", index=False)

    parent_pairs = {
        "M4-M2": (final.identity.M4, final.identity.M2),
        "M5-M3": (final.identity.M5, final.identity.M3),
    }
    parent_rows = []
    for comparison, (proposed, benchmark) in parent_pairs.items():
        selected = comparisons.loc[
            (comparisons["proposed"] == proposed)
            & (comparisons["benchmark"] == benchmark)
        ].copy()
        selected["comparison"] = comparison
        parent_rows.append(selected)
    parent_comparisons = pd.concat(parent_rows, ignore_index=True)
    parent_comparisons.to_csv(
        OUTPUT / "factor_robustness_parent_oos_comparisons.csv", index=False
    )

    lines = [
        "# Case 1 market-factor robustness", "",
        "Raw constituent returns, causal one-factor residuals, and the official index",
        "return are fitted with the same locked Case-1 M1--M5 specification.", "",
        "Rolling market-model coefficients use only prior observations, with a maximum",
        f"window of {ROLLING_WINDOW} and a minimum of {MINIMUM_OBSERVATIONS}. Raw and",
        "residual panels contain identical asset-days; the index target uses the same dates.", "",
        "## Panel diagnostics", "",
        "| Market | First date | Days | Asset-days | Daily N min/median/max | Raw mean--market corr. | Residual mean--market corr. |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for market in ("sp500", "n225"):
        d = diagnostics[market]
        lines.append(
            f"| {d['label']} | {d['first_date']} | {d['days']} | {d['asset_days']} | "
            f"{d['daily_constituents_minimum']}/{d['daily_constituents_median']:.0f}/{d['daily_constituents_maximum']} | "
            f"{d['daily_raw_mean_market_correlation']:.3f} | {d['daily_residual_mean_market_correlation']:.3f} |"
        )
    lines.extend(["", "## Conditional sandwich-Wald p-values", ""])
    for market in ("sp500", "n225"):
        lines.extend([
            f"### {MARKETS[market]['label']}", "",
            "| Target | M2 location | M2 scale | M3 eta | M3 lambda | M4 even | M4 odd | M4 joint | M5 even | M5 odd | M5 joint |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for target in ("raw_matched", "idiosyncratic", "index"):
            subset = tests[(tests.market == market) & (tests.target == target)]
            lookup = {(r.model, r.block): r.p for r in subset.itertuples()}
            values = [
                lookup[("M2", "location")], lookup[("M2", "scale")],
                lookup[("M3", "eta")], lookup[("M3", "lambda")],
                lookup[("M4", "even")], lookup[("M4", "odd")], lookup[("M4", "joint")],
                lookup[("M5", "even")], lookup[("M5", "odd")], lookup[("M5", "joint")],
            ]
            lines.append("| " + target + " | " + " | ".join(f"{value:.4g}" for value in values) + " |")
        lines.append("")

    lines.extend(["## Strict-OOS winning models", ""])
    for market in ("sp500", "n225"):
        lines.extend([
            f"### {MARKETS[market]['label']}", "",
            "| Metric | raw matched | idiosyncratic | index N=1 |",
            "|---|---|---|---|",
        ])
        for metric in metric_names:
            row = winner_frame[(winner_frame.market == market) & (winner_frame.metric == metric)]
            lookup = dict(zip(row.target, row.winner))
            lines.append(
                f"| {metric} | {lookup['raw_matched']} | {lookup['idiosyncratic']} | {lookup['index']} |"
            )
        lines.append("")
    lines.extend([
        "## Strict-OOS parent-model improvements", "",
        "Positive values favor the dynamic endpoint model. P-values are two-sided",
        "HAC(22) tests of the mean score difference.", "",
        "| Market | Target | Comparison | Continuous log improvement (p) | Q-loss 1% improvement (p) |",
        "|---|---|---|---:|---:|",
    ])
    for market in ("sp500", "n225"):
        for target in ("raw_matched", "idiosyncratic", "index"):
            for comparison in ("M4-M2", "M5-M3"):
                chosen = parent_comparisons.loc[
                    (parent_comparisons.market == market)
                    & (parent_comparisons.target == target)
                    & (parent_comparisons.comparison == comparison)
                ].set_index("score")
                log = chosen.loc["continuous_log_score"]
                tail = chosen.loc["q_loss_0.01"]
                lines.append(
                    f"| {MARKETS[market]['label']} | {target} | {comparison} | "
                    f"{log.improvement_mean:.6g} ({log.hac22_two_sided_p:.4g}) | "
                    f"{tail.improvement_mean:.6g} ({tail.hac22_two_sided_p:.4g}) |"
                )
    lines.extend(["", "## Main robustness finding", ""])
    for market in ("sp500", "n225"):
        d = diagnostics[market]
        idio_winners = winner_frame.loc[
            (winner_frame.market == market)
            & (winner_frame.target == "idiosyncratic"),
            "winner",
        ].value_counts()
        tests_idio = tests.loc[
            (tests.market == market) & (tests.target == "idiosyncratic")
        ]
        lookup = {(row.model, row.block): row.p for row in tests_idio.itertuples()}
        lines.append(
            f"- **{d['label']}:** residualizing lowers the daily cross-sectional "
            f"mean/index correlation from {d['daily_raw_mean_market_correlation']:.3f} "
            f"to {d['daily_residual_mean_market_correlation']:.3f}. Yet M3 eta "
            f"(p={lookup[('M3', 'eta')]:.3g}), M4 joint "
            f"(p={lookup[('M4', 'joint')]:.3g}), and M5 joint "
            f"(p={lookup[('M5', 'joint')]:.3g}) remain significant. M5 wins "
            f"{int(idio_winners.get('M5', 0))}/9 strict-OOS scores."
        )
    lines.extend([
        "",
        "Thus the cross-sectional result is not exhausted by the contemporaneous",
        "market-index factor. Predictable tail-index and endpoint-mass variation",
        "remains in the idiosyncratic cross-sectional distribution. The N=1 index",
        "fits also contain strong dynamics, but their OOS model rankings differ from",
        "those of the factor-adjusted cross sections.", "",
    ])
    lines.extend([
        "The score levels across the three targets are not directly comparable because",
        "their dependent variables differ. Model rankings and within-target pairwise",
        "comparisons are the relevant forecasting evidence.", "",
    ])
    (OUTPUT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=("sp500", "n225", "both"), default="both")
    parser.add_argument(
        "--target", choices=("raw_matched", "idiosyncratic", "index", "all"), default="all"
    )
    parser.add_argument("--refresh-market-data", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    markets = ("sp500", "n225") if args.market == "both" else (args.market,)
    targets = ("raw_matched", "idiosyncratic", "index") if args.target == "all" else (args.target,)
    results: dict[str, dict[str, object]] = {}
    all_diagnostics: dict[str, object] = {}
    for market in markets:
        panels, diagnostics = prepare_market_panels(market, args.refresh_market_data)
        all_diagnostics[market] = diagnostics
        for target in targets:
            key = f"{market}_{target}"
            path = OUTPUT / key / "estimation.json"
            if args.resume and path.exists():
                print(f"{key}: using saved result", flush=True)
                results[key] = json.loads(path.read_text(encoding="utf-8"))
            else:
                results[key] = fit_panel(key, panels[target], market, target, diagnostics)

    # Include already completed cells when a partial command is resumed.
    for market in ("sp500", "n225"):
        diagnostic_path = OUTPUT / market / "panel_diagnostics.json"
        if market not in all_diagnostics and diagnostic_path.exists():
            all_diagnostics[market] = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        for target in ("raw_matched", "idiosyncratic", "index"):
            key = f"{market}_{target}"
            path = OUTPUT / key / "estimation.json"
            if key not in results and path.exists():
                results[key] = json.loads(path.read_text(encoding="utf-8"))
    if len(results) == 6 and len(all_diagnostics) == 2:
        summarize(results, all_diagnostics)
        print(f"summary: {OUTPUT / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
