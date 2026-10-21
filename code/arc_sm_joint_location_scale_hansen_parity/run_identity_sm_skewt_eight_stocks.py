"""Locked identity-SM skew-t M1--M5 experiment for eight added stocks.

The estimators are imported unchanged from the final identity-SM experiment.
Only the Yahoo data loader and output routing are new.  The universe was fixed
before looking at results as four broad US/Japan sector pairs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_direct_dynamic_skewt_shape_test as direct
import run_identity_sm_skewt_five_models as identity
import run_sp500 as core
import run_static_q0_single_stocks as stocks
import run_static_q0_symmetric_t_four_models as q0


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUTPUT_ROOT = ROOT / "output_identity_sm_skewt_eight_stocks_five_models"
DIRECT_OUTPUT_ROOT = OUTPUT_ROOT / "_direct_m1_m3_stage"
CACHE = REPO / "data" / "global_index_daily_yahoo" / "_yahoo_cache"
SAMPLE_START = pd.Timestamp("2016-07-05")
TRAINING_END = pd.Timestamp("2023-12-29")
OOS_END = pd.Timestamp("2026-07-24")

ASSETS = {
    "msft": {"ticker": "MSFT", "label": "Microsoft", "market": "US", "sector": "Technology/global"},
    "sony": {"ticker": "6758.T", "label": "Sony", "market": "Japan", "sector": "Technology/global"},
    "jpm": {"ticker": "JPM", "label": "JPMorgan", "market": "US", "sector": "Banking"},
    "mufg": {"ticker": "8306.T", "label": "MUFG", "market": "Japan", "sector": "Banking"},
    "xom": {"ticker": "XOM", "label": "ExxonMobil", "market": "US", "sector": "Energy"},
    "inpex": {"ticker": "1605.T", "label": "INPEX", "market": "Japan", "sector": "Energy"},
    "wmt": {"ticker": "WMT", "label": "Walmart", "market": "US", "sector": "Consumer/retail"},
    "fastretailing": {"ticker": "9983.T", "label": "Fast Retailing", "market": "Japan", "sector": "Consumer/retail"},
}

SCORE_LABELS = {
    "continuous_log_score": "Log",
    "crps": "CRPS",
    "categorical_log_score_50": "BinLog 50",
    "categorical_log_score_100": "BinLog 100",
    "categorical_log_score_200": "BinLog 200",
    "q_loss_0.01": "Q 1%",
    "q_loss_0.05": "Q 5%",
    "brier_0.01": "Brier 1%",
    "brier_0.05": "Brier 5%",
}


def load_asset(asset: str) -> tuple[core.RawPanel, dict[str, object]]:
    specification = ASSETS[asset]
    ticker = str(specification["ticker"])
    cache_path = CACHE / f"{ticker}.json"
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        source = str(cache_path)
    else:
        print(f"{ticker}: downloading Yahoo daily history", flush=True)
        payload = stocks.fetch_yahoo_cache(ticker, cache_path)
        source = str(payload["source"])
    if payload.get("error"):
        raise RuntimeError(f"{ticker} cache error: {payload['error']}")

    prices = pd.DataFrame(payload["records"])
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values("date").drop_duplicates("date", keep="last")
    prices["log_return"] = np.log(prices["adj_close"]).diff()
    data = prices.loc[
        prices["date"].between(SAMPLE_START, OOS_END), ["date", "log_return"]
    ].dropna().reset_index(drop=True)
    if data.empty or not np.isfinite(data["log_return"]).all():
        raise RuntimeError(f"{ticker} return sample is empty or non-finite")
    n_train = int(np.sum(data["date"] <= TRAINING_END))
    if n_train <= 22 or n_train == len(data):
        raise RuntimeError(f"{ticker} invalid split: {n_train}/{len(data)}")
    values = data["log_return"].to_numpy(dtype=float)
    panel = core.RawPanel(
        daily=pd.DataFrame({
            "date": data["date"],
            "n_constituents": np.ones(values.size, dtype=int),
        }),
        values=values,
        offsets=np.arange(values.size + 1, dtype=int),
    )
    return panel, {
        "dataset": asset,
        "asset": asset,
        "ticker": ticker,
        "label": specification["label"],
        "market": specification["market"],
        "sector": specification["sector"],
        "cache": str(cache_path),
        "source_used": source,
    }


def configure_imported_experiment() -> None:
    direct.OUTPUT_ROOT = DIRECT_OUTPUT_ROOT
    direct.data_source.load_dataset = load_asset
    identity.OUTPUT_ROOT = OUTPUT_ROOT
    identity.direct.OUTPUT_ROOT = DIRECT_OUTPUT_ROOT
    identity.five.load_dataset = load_asset


def add_m4_vs_m3(asset: str) -> pd.DataFrame:
    output = OUTPUT_ROOT / asset
    scores = pd.read_csv(output / "daily_oos_scores.csv")
    comparison = pd.DataFrame([
        core.score_comparison(
            scores, identity.MODEL_NAMES["m4"], identity.MODEL_NAMES["m3"], score
        )
        for score in q0.SCORE_NAMES
    ])
    comparison.to_csv(output / "m4_vs_m3_oos_comparisons.csv", index=False)
    return comparison


def pooled_m4_vs_m3(market: str | None) -> pd.DataFrame:
    selected = [
        asset for asset, specification in ASSETS.items()
        if market is None or specification["market"] == market
    ]
    frames = []
    common_dates: set[str] | None = None
    for asset in selected:
        frame = pd.read_csv(OUTPUT_ROOT / asset / "daily_oos_scores.csv")
        frame = frame.loc[frame["model"].isin((
            identity.MODEL_NAMES["m3"], identity.MODEL_NAMES["m4"]
        ))].copy()
        dates = set(frame["date"])
        common_dates = dates if common_dates is None else common_dates & dates
        frame.insert(0, "asset", asset)
        frames.append(frame)
    aligned = pd.concat(frames, ignore_index=True)
    aligned = aligned.loc[aligned["date"].isin(common_dates)]
    pooled = aligned.groupby(["date", "model"], as_index=False)[list(q0.SCORE_NAMES)].mean()
    comparison = pd.DataFrame([
        core.score_comparison(
            pooled, identity.MODEL_NAMES["m4"], identity.MODEL_NAMES["m3"], score
        )
        for score in q0.SCORE_NAMES
    ])
    comparison.insert(0, "common_oos_days", len(common_dates))
    comparison.insert(0, "pool", "All" if market is None else market)
    return comparison


def relative_percent(levels: pd.DataFrame, model: str, score: str) -> float:
    indexed = levels.set_index("model")
    proposed = float(indexed.loc[model, score])
    base = float(indexed.loc[identity.MODEL_NAMES["m2"], score])
    if score in {
        "continuous_log_score", "categorical_log_score_50",
        "categorical_log_score_100", "categorical_log_score_200",
    }:
        return 100.0 * (proposed - base) / abs(base)
    return 100.0 * (base - proposed) / abs(base)


def plot_oos_heatmap() -> None:
    figures = OUTPUT_ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    models = ("m3", "m4", "m5")
    arrays = []
    for asset in ASSETS:
        levels = pd.read_csv(OUTPUT_ROOT / asset / "five_model_oos_summary.csv")
        arrays.append(np.array([
            [relative_percent(levels, identity.MODEL_NAMES[model], score) for score in SCORE_LABELS]
            for model in models
        ]))
    bound = max(float(np.max(np.abs(values))) for values in arrays)
    fig, axes = plt.subplots(4, 2, figsize=(15, 15), constrained_layout=True)
    image = None
    for axis, (asset, specification), values in zip(axes.ravel(), ASSETS.items(), arrays):
        image = axis.imshow(values, cmap="RdYlGn", vmin=-bound, vmax=bound, aspect="auto")
        axis.set_title(f"{specification['label']} ({specification['market']})", fontweight="bold")
        axis.set_xticks(range(len(SCORE_LABELS)), [label.replace(" ", "\n", 1) for label in SCORE_LABELS.values()], fontsize=7)
        axis.set_yticks(range(3), ("M3: eta,lambda", "M4: q_eta,others", "M5: residual"))
        for row in range(3):
            for column in range(len(SCORE_LABELS)):
                axis.text(column, row, f"{values[row, column]:+.3f}", ha="center", va="center", fontsize=6)
    fig.colorbar(image, ax=axes, shrink=0.8, label="OOS improvement relative to M2 (%)")
    fig.suptitle("Eight added stocks: strict-OOS accuracy relative to Model 2", fontweight="bold")
    fig.savefig(figures / "eight_stocks_oos_relative_heatmap.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures / "eight_stocks_oos_relative_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_coefficient_heatmap(results: dict[str, dict[str, object]]) -> None:
    figures = OUTPUT_ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    columns = (
        "eta D", "eta W", "eta M", "lambda D", "lambda W", "lambda M",
        "q_eta D", "q_eta W", "q_eta M", "others D", "others W", "others M",
        "resid. D", "resid. W", "resid. M",
    )
    arrays = []
    for result in results.values():
        values = np.full((3, 15), np.nan)
        m3 = {(row["state"], int(row["window"])): row["t"] for row in result["model3"]["parameters"]}
        values[0, :6] = [m3[(state, window)] for state in ("eta", "lambda") for window in (1, 5, 22)]
        m4 = {(row["block"], int(row["horizon"])): row["t"] for row in result["model4"]["parameters"]}
        values[1, 6:12] = [m4[(block, window)] for block in ("q_eta", "others") for window in (1, 5, 22)]
        values[2, 12:15] = [row["t"] for row in result["model5"]["parameters"]]
        arrays.append(values)
    bound = max(float(np.nanmax(np.abs(values))) for values in arrays)
    fig, axes = plt.subplots(4, 2, figsize=(16, 15), constrained_layout=True)
    image = None
    for axis, (asset, specification), values in zip(axes.ravel(), ASSETS.items(), arrays):
        image = axis.imshow(np.ma.masked_invalid(values), cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
        axis.set_title(f"{specification['label']} ({specification['market']})", fontweight="bold")
        axis.set_xticks(range(15), columns, rotation=90, fontsize=7)
        axis.set_yticks(range(3), ("M3: eta,lambda", "M4: q_eta,others", "M5: residual"))
        for row in range(3):
            for column in range(15):
                if np.isfinite(values[row, column]):
                    star = "*" if abs(values[row, column]) >= 1.96 else ""
                    axis.text(column, row, f"{values[row, column]:+.2f}{star}", ha="center", va="center", fontsize=6)
    fig.colorbar(image, ax=axes, shrink=0.8, label="HAC(22) t statistic")
    fig.suptitle("Eight added stocks: dynamic-shape coefficient t statistics", fontweight="bold")
    fig.savefig(figures / "eight_stocks_coefficient_t_heatmap.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures / "eight_stocks_coefficient_t_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def write_report(
    results: dict[str, dict[str, object]], comparisons: dict[str, pd.DataFrame], pooled: pd.DataFrame
) -> None:
    lines = [
        "# Eight added US/Japan stocks: identity-SM skew-t M1--M5",
        "",
        "## Locked design",
        "",
        "The eight stocks and four sector pairs were fixed before inspecting any",
        "results. All use adjusted-close daily log returns, the S&P 500 sample",
        "bounds, h0(u)=1, HAR(1,5,22), no whitening, no parity split, and no outlier",
        "removal. M1--M5 are unchanged from the final identity-SM experiment.",
        "",
        "| sector | US | Japan |",
        "|---|---|---|",
    ]
    sectors = ("Technology/global", "Banking", "Energy", "Consumer/retail")
    for sector in sectors:
        us = next(v for v in ASSETS.values() if v["sector"] == sector and v["market"] == "US")
        jp = next(v for v in ASSETS.values() if v["sector"] == sector and v["market"] == "Japan")
        lines.append(f"| {sector} | {us['label']} `{us['ticker']}` | {jp['label']} `{jp['ticker']}` |")
    lines += [
        "",
        "## Samples and in-sample joint tests",
        "",
        "| stock | start | train end | OOS start | end | M3 eta | M3 lambda | M4 eta dir. | M4 others | M5 residual |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for asset, result in results.items():
        sample = result["sample"]
        m3 = result["model3"]["wald_tests"]
        lines.append(
            f"| {ASSETS[asset]['label']} | {sample['start']} | {sample['training_end']} | "
            f"{sample['oos_start']} | {sample['end']} | "
            f"{m3['eta_three_coefficients']['p']:.6g} | {m3['lambda_three_coefficients']['p']:.6g} | "
            f"{result['model4']['joint_blocks']['q_eta']['p']:.6g} | "
            f"{result['model4']['joint_blocks']['others']['p']:.6g} | "
            f"{result['model5']['joint_all']['p']:.6g} |"
        )
    lines += [
        "",
        "## Strict-OOS score levels for all five models",
        "",
        "Higher Log and lower CRPS are better. The complete nine-score levels are",
        "also saved in `all_five_model_oos_levels.csv`.",
        "",
        "| stock | model | Log | CRPS |",
        "|---|---|---:|---:|",
    ]
    for asset in ASSETS:
        levels = pd.read_csv(OUTPUT_ROOT / asset / "five_model_oos_summary.csv")
        for model_number, model_key in enumerate(("m1", "m2", "m3", "m4", "m5"), start=1):
            row = levels.set_index("model").loc[identity.MODEL_NAMES[model_key]]
            lines.append(
                f"| {ASSETS[asset]['label']} | M{model_number} | "
                f"{row['continuous_log_score']:.9g} | {row['crps']:.9g} |"
            )
    lines += [
        "",
        "## Strict-OOS M4 versus M3",
        "",
        "Positive improvement favors M4; parentheses give paired HAC(22) p-values.",
        "",
        "| stock | " + " | ".join(SCORE_LABELS.values()) + " |",
        "|---|" + "---:|" * len(SCORE_LABELS),
    ]
    for asset, comparison in comparisons.items():
        indexed = comparison.set_index("score")
        cells = [
            f"{indexed.loc[score, 'improvement_mean']:+.6g} ({indexed.loc[score, 'hac22_two_sided_p']:.4g})"
            for score in SCORE_LABELS
        ]
        lines.append(f"| {ASSETS[asset]['label']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Equal-weight pooled M4 versus M3",
        "",
        "Each pool is restricted to dates common to every constituent stock; stock",
        "differences are averaged within date before HAC(22), preserving cross-stock",
        "dependence.",
        "",
        "| pool | common days | score | improvement | HAC(22) p |",
        "|---|---:|---|---:|---:|",
    ]
    for row in pooled.to_dict("records"):
        lines.append(
            f"| {row['pool']} | {row['common_oos_days']} | {SCORE_LABELS[row['score']]} | "
            f"{row['improvement_mean']:+.9g} | {row['hac22_two_sided_p']:.6g} |"
        )
    lines += [
        "",
        "## Numerical audit",
        "",
        "All reported optimizations converged from every requested start. The table",
        "records the largest objective gap across successful starts and the largest",
        "continuation/nesting identity error for each stock.",
        "",
        "| stock | M2 starts | M3 starts | M4 starts | M5 starts | max objective gap | max identity error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for asset, result in results.items():
        starts = [
            f"{result[f'model{k}']['successful_starts']}/{result[f'model{k}']['total_starts']}"
            for k in (2, 3, 4, 5)
        ]
        gaps = [float(result[f"model{k}"]["largest_successful_objective_gap"]) for k in (2, 3, 4, 5)]
        checks = result["checks"]
        errors = [
            float(checks["saved_current_model2_likelihood_error"]),
            float(checks["saved_current_model3_likelihood_error"]),
            float(checks["model2_continuation_error"]),
            float(checks["model3_zero_nesting_error"]),
            float(checks["model3_continuation_error"]),
            float(checks["model4"]["continuation_error"]),
            float(checks["model4"]["q_zero_nesting_error"]),
            float(checks["model5"]["continuation_error"]),
            float(checks["model5"]["q_zero_nesting_error"]),
        ]
        lines.append(
            f"| {ASSETS[asset]['label']} | {' | '.join(starts)} | "
            f"{max(gaps):.3g} | {max(abs(value) for value in errors):.3g} |"
        )
    lines += [
        "",
        "Complete daily scores, all five-model OOS levels and comparisons,",
        "coefficients, estimation JSON, and numerical checks are stored in each",
        "stock subdirectory. Raw Yahoo payloads remain in the ignored data cache.",
    ]
    (OUTPUT_ROOT / "EIGHT_STOCK_IDENTITY_SM_SKEWT_RESULTS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_asset(asset: str) -> dict[str, object]:
    print(f"{asset}: fitting direct M1--M3 stage", flush=True)
    direct.run_dataset(asset)
    print(f"{asset}: fitting complete identity-SM M1--M5", flush=True)
    return identity.run_dataset(asset)


def summarize_all() -> None:
    results = {
        asset: json.loads((OUTPUT_ROOT / asset / "estimation.json").read_text(encoding="utf-8"))
        for asset in ASSETS
    }
    comparisons = {asset: add_m4_vs_m3(asset) for asset in ASSETS}
    pooled = pd.concat(
        (pooled_m4_vs_m3("US"), pooled_m4_vs_m3("Japan"), pooled_m4_vs_m3(None)),
        ignore_index=True,
    )
    pooled.to_csv(OUTPUT_ROOT / "pooled_m4_vs_m3_oos.csv", index=False)
    level_frames = []
    audit_rows = []
    for asset, result in results.items():
        levels = pd.read_csv(OUTPUT_ROOT / asset / "five_model_oos_summary.csv")
        levels.insert(0, "market", ASSETS[asset]["market"])
        levels.insert(0, "stock", ASSETS[asset]["label"])
        levels.insert(0, "asset", asset)
        level_frames.append(levels)
        checks = result["checks"]
        identity_errors = [
            checks["saved_current_model2_likelihood_error"],
            checks["saved_current_model3_likelihood_error"],
            checks["model2_continuation_error"],
            checks["model3_zero_nesting_error"],
            checks["model3_continuation_error"],
            checks["model4"]["continuation_error"],
            checks["model4"]["q_zero_nesting_error"],
            checks["model5"]["continuation_error"],
            checks["model5"]["q_zero_nesting_error"],
        ]
        audit_rows.append({
            "asset": asset,
            **{
                f"model{k}_successful_starts": result[f"model{k}"]["successful_starts"]
                for k in (2, 3, 4, 5)
            },
            **{
                f"model{k}_total_starts": result[f"model{k}"]["total_starts"]
                for k in (2, 3, 4, 5)
            },
            "maximum_objective_gap": max(
                result[f"model{k}"]["largest_successful_objective_gap"] for k in (2, 3, 4, 5)
            ),
            "maximum_continuation_or_nesting_error": max(abs(value) for value in identity_errors),
        })
    pd.concat(level_frames, ignore_index=True).to_csv(
        OUTPUT_ROOT / "all_five_model_oos_levels.csv", index=False
    )
    pd.DataFrame(audit_rows).to_csv(OUTPUT_ROOT / "numerical_audit.csv", index=False)
    write_report(results, comparisons, pooled)
    plot_oos_heatmap()
    plot_coefficient_heatmap(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", choices=(*ASSETS, "all"), default="all")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    configure_imported_experiment()
    if args.summarize_only:
        summarize_all()
        return
    selected = tuple(ASSETS) if args.asset == "all" else (args.asset,)
    for asset in selected:
        run_asset(asset)
        add_m4_vs_m3(asset)
    if set(selected) == set(ASSETS):
        summarize_all()


if __name__ == "__main__":
    main()
