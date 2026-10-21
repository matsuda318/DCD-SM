"""Identity-SM skew-t M1--M5 experiment for three daily FX returns.

The econometric specification is imported without modification from
``run_identity_sm_skewt_five_models``.  This file only supplies Yahoo Finance
daily FX data, keeps the S&P 500 calendar bounds, and writes an FX-specific
combined report and figures.

Quotes follow Yahoo's displayed orientation:

* USD/JPY: JPY=X (JPY per USD)
* USD/EUR: EUR=X (EUR per USD)
* EUR/JPY: EURJPY=X (JPY per EUR)
* USD/CHF: CHF=X (CHF per USD; Hansen 1994 currency pair)
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
OUTPUT_ROOT = ROOT / "output_identity_sm_skewt_fx_five_models"
DIRECT_OUTPUT_ROOT = OUTPUT_ROOT / "_direct_m1_m3_stage"
CACHE = REPO / "data" / "fx_daily_yahoo" / "_yahoo_cache"
SAMPLE_START = pd.Timestamp("2016-07-05")
TRAINING_END = pd.Timestamp("2023-12-29")
OOS_END = pd.Timestamp("2026-07-24")

FX = {
    "usdjpy": {"ticker": "JPY=X", "label": "USD/JPY"},
    "usdeur": {"ticker": "EUR=X", "label": "USD/EUR"},
    "eurjpy": {"ticker": "EURJPY=X", "label": "EUR/JPY"},
    "usdchf": {"ticker": "CHF=X", "label": "USD/CHF"},
}

SCORE_LABELS = {
    "continuous_log_score": "Log",
    "crps": "CRPS",
    "categorical_log_score_50": "BinLog\n50",
    "categorical_log_score_100": "BinLog\n100",
    "categorical_log_score_200": "BinLog\n200",
    "q_loss_0.01": "Q 1%",
    "q_loss_0.05": "Q 5%",
    "brier_0.01": "Brier\n1%",
    "brier_0.05": "Brier\n5%",
}


def load_fx(pair: str) -> tuple[core.RawPanel, dict[str, object]]:
    specification = FX[pair]
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
        prices["date"].between(SAMPLE_START, OOS_END),
        ["date", "adj_close", "log_return"],
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
        "dataset": pair,
        "asset": pair,
        "ticker": ticker,
        "label": specification["label"],
        "quote_orientation": specification["label"],
        "cache": str(cache_path),
        "source_used": source,
    }


def configure_imported_experiment() -> None:
    """Route the unchanged imported estimators to the FX loader/output."""
    direct.OUTPUT_ROOT = DIRECT_OUTPUT_ROOT
    direct.data_source.load_dataset = load_fx
    identity.OUTPUT_ROOT = OUTPUT_ROOT
    identity.direct.OUTPUT_ROOT = DIRECT_OUTPUT_ROOT
    identity.five.load_dataset = load_fx


def add_m4_vs_m3(pair: str) -> pd.DataFrame:
    output = OUTPUT_ROOT / pair
    scores = pd.read_csv(output / "daily_oos_scores.csv")
    rows = [
        core.score_comparison(
            scores,
            identity.MODEL_NAMES["m4"],
            identity.MODEL_NAMES["m3"],
            score,
        )
        for score in q0.SCORE_NAMES
    ]
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / "m4_vs_m3_oos_comparisons.csv", index=False)
    return comparison


def pooled_m4_vs_m3() -> pd.DataFrame:
    """Equal-weight pairs by date before HAC, preserving cross-pair dependence."""
    frames = []
    for pair in FX:
        frame = pd.read_csv(OUTPUT_ROOT / pair / "daily_oos_scores.csv")
        frame = frame.loc[frame["model"].isin((
            identity.MODEL_NAMES["m3"], identity.MODEL_NAMES["m4"]
        ))].copy()
        frames.append(frame)
    pooled = (
        pd.concat(frames, ignore_index=True)
        .groupby(["date", "model"], as_index=False)[list(q0.SCORE_NAMES)]
        .mean()
    )
    comparison = pd.DataFrame([
        core.score_comparison(
            pooled, identity.MODEL_NAMES["m4"], identity.MODEL_NAMES["m3"], score
        )
        for score in q0.SCORE_NAMES
    ])
    comparison.to_csv(OUTPUT_ROOT / "pooled_equal_weight_m4_vs_m3_oos.csv", index=False)
    return comparison


def relative_percent(levels: pd.DataFrame, model: str, benchmark: str, score: str) -> float:
    indexed = levels.set_index("model")
    proposed = float(indexed.loc[model, score])
    base = float(indexed.loc[benchmark, score])
    if score in {
        "continuous_log_score",
        "categorical_log_score_50",
        "categorical_log_score_100",
        "categorical_log_score_200",
    }:
        return 100.0 * (proposed - base) / abs(base)
    return 100.0 * (base - proposed) / abs(base)


def plot_oos_heatmap() -> None:
    figures = OUTPUT_ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    models = ("m3", "m4", "m5")
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), constrained_layout=True)
    axes = np.asarray(axes).ravel()
    arrays = []
    for pair in FX:
        levels = pd.read_csv(OUTPUT_ROOT / pair / "five_model_oos_summary.csv")
        arrays.append(np.array([
            [
                relative_percent(
                    levels, identity.MODEL_NAMES[model], identity.MODEL_NAMES["m2"], score
                )
                for score in SCORE_LABELS
            ]
            for model in models
        ]))
    bound = max(float(np.nanmax(np.abs(values))) for values in arrays)
    bound = max(bound, 1e-9)
    image = None
    for axis, (pair, specification), values in zip(axes, FX.items(), arrays):
        image = axis.imshow(values, cmap="RdYlGn", vmin=-bound, vmax=bound, aspect="auto")
        axis.set_title(specification["label"], fontweight="bold")
        axis.set_xticks(range(len(SCORE_LABELS)), SCORE_LABELS.values(), fontsize=8)
        axis.set_yticks(range(3), ("M3: eta,lambda", "M4: q_eta,others", "M5: residual"))
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                axis.text(column, row, f"{values[row, column]:+.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=axes, shrink=0.86, label="OOS improvement relative to M2 (%)")
    fig.suptitle("Daily FX identity-SM skew-t: strict-OOS accuracy relative to Model 2", fontweight="bold")
    fig.savefig(figures / "identity_sm_skewt_fx_oos_relative_heatmap.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures / "identity_sm_skewt_fx_oos_relative_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_coefficient_heatmap(results: dict[str, dict[str, object]]) -> None:
    figures = OUTPUT_ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    columns = (
        "eta D", "eta W", "eta M", "lambda D", "lambda W", "lambda M",
        "q_eta D", "q_eta W", "q_eta M", "others D", "others W", "others M",
        "resid. D", "resid. W", "resid. M",
    )
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), constrained_layout=True)
    axes = np.asarray(axes).ravel()
    all_arrays = []
    for result in results.values():
        values = np.full((3, len(columns)), np.nan)
        m3 = {(row["state"], int(row["window"])): row["t"] for row in result["model3"]["parameters"]}
        values[0, :6] = [m3[(state, window)] for state in ("eta", "lambda") for window in (1, 5, 22)]
        m4 = {(row["block"], int(row["horizon"])): row["t"] for row in result["model4"]["parameters"]}
        values[1, 6:12] = [m4[(block, window)] for block in ("q_eta", "others") for window in (1, 5, 22)]
        values[2, 12:15] = [row["t"] for row in result["model5"]["parameters"]]
        all_arrays.append(values)
    bound = max(float(np.nanmax(np.abs(values))) for values in all_arrays)
    image = None
    for axis, (pair, specification), values in zip(axes, FX.items(), all_arrays):
        masked = np.ma.masked_invalid(values)
        image = axis.imshow(masked, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
        axis.set_title(specification["label"], fontweight="bold")
        axis.set_xticks(range(len(columns)), columns, rotation=90, fontsize=7)
        axis.set_yticks(range(3), ("M3: eta,lambda", "M4: q_eta,others", "M5: residual"))
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                if np.isfinite(values[row, column]):
                    star = "*" if abs(values[row, column]) >= 1.96 else ""
                    axis.text(column, row, f"{values[row, column]:+.2f}{star}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=axes, shrink=0.86, label="HAC(22) t statistic")
    fig.suptitle("Daily FX identity-SM skew-t: dynamic-shape coefficient t statistics", fontweight="bold")
    fig.savefig(figures / "identity_sm_skewt_fx_coefficient_t_heatmap.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures / "identity_sm_skewt_fx_coefficient_t_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def write_combined_report(
    results: dict[str, dict[str, object]],
    m4_m3: dict[str, pd.DataFrame],
    pooled: pd.DataFrame,
) -> None:
    lines = [
        "# Daily FX identity-SM skew-t M1--M5 results",
        "",
        "## Locked specification",
        "",
        "The M1--M5 specification is identical to the final S&P 500 identity-SM",
        "experiment: h0(u)=1, HAR(1,5,22), fixed-anchor Fisher projections, no",
        "whitening, and no parity split. The sample bounds and training cutoff are",
        "also fixed to the S&P 500 experiment.",
        "",
        "USD/JPY, USD/EUR, and USD/CHF are quoted as units of the second currency",
        "per USD; EUR/JPY is JPY per EUR. The first three non-CHF returns obey an",
        "approximate triangular identity and must not be interpreted as three",
        "independent replications. USD/CHF is the currency pair used in Hansen's",
        "1994 exchange-rate application (with a different historical sample).",
        "",
        "## Samples",
        "",
        "| pair | ticker | start | training end | OOS start | end | train days | OOS days |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for pair, result in results.items():
        sample = result["sample"]
        lines.append(
            f"| {FX[pair]['label']} | {FX[pair]['ticker']} | {sample['start']} | "
            f"{sample['training_end']} | {sample['oos_start']} | {sample['end']} | "
            f"{sample['training_days']} | {sample['oos_days']} |"
        )
    lines += [
        "",
        "## In-sample conditional Godambe-HAC(22) joint tests",
        "",
        "| pair | M3 eta | M3 lambda | M3 joint | M4 eta dir. | M4 others | M4 joint | M5 residual |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pair, result in results.items():
        tests = result["model3"]["wald_tests"]
        lines.append(
            f"| {FX[pair]['label']} | {tests['eta_three_coefficients']['p']:.6g} | "
            f"{tests['lambda_three_coefficients']['p']:.6g} | "
            f"{tests['eta_lambda_six_coefficients']['p']:.6g} | "
            f"{result['model4']['joint_blocks']['q_eta']['p']:.6g} | "
            f"{result['model4']['joint_blocks']['others']['p']:.6g} | "
            f"{result['model4']['joint_all']['p']:.6g} | "
            f"{result['model5']['joint_all']['p']:.6g} |"
        )
    lines += [
        "",
        "## Strict-OOS log-score and CRPS levels",
        "",
        "| pair/model | log score | CRPS |",
        "|---|---:|---:|",
    ]
    for pair in FX:
        levels = pd.read_csv(OUTPUT_ROOT / pair / "five_model_oos_summary.csv").set_index("model")
        for model in ("m1", "m2", "m3", "m4", "m5"):
            row = levels.loc[identity.MODEL_NAMES[model]]
            lines.append(
                f"| {FX[pair]['label']} {model.upper()} | "
                f"{row['continuous_log_score']:.9f} | {row['crps']:.9f} |"
            )
    lines += [
        "",
        "## Direct strict-OOS M4 versus M3",
        "",
        "Positive improvement favors M4; parentheses give the paired HAC(22) p-value.",
        "",
        "| pair | " + " | ".join(label.replace("\n", " ") for label in SCORE_LABELS.values()) + " |",
        "|---|" + "---:|" * len(SCORE_LABELS),
    ]
    for pair, comparison in m4_m3.items():
        indexed = comparison.set_index("score")
        cells = [
            f"{indexed.loc[score, 'improvement_mean']:+.6g} "
            f"({indexed.loc[score, 'hac22_two_sided_p']:.4g})"
            for score in SCORE_LABELS
        ]
        lines.append(f"| {FX[pair]['label']} | " + " | ".join(cells) + " |")
    pooled_indexed = pooled.set_index("score")
    lines += [
        "",
        "## Equal-weight pooled strict-OOS M4 versus M3",
        "",
        "The four pair-specific score differences are averaged within each date",
        "before applying HAC(22). This preserves contemporaneous dependence across",
        "the currencies and is preferable to treating four wins as independent.",
        "",
        "| score | improvement | HAC(22) p |",
        "|---|---:|---:|",
    ]
    for score, label in SCORE_LABELS.items():
        row = pooled_indexed.loc[score]
        lines.append(
            f"| {label.replace(chr(10), ' ')} | {row['improvement_mean']:+.9g} | "
            f"{row['hac22_two_sided_p']:.6g} |"
        )
    lines += [
        "",
        "## Numerical audit",
        "",
        "The objective gap is the largest difference among successful starts.",
        "Structural error is the maximum saved/current, nesting, and continuation",
        "check reported by the five-model estimator.",
        "",
        "| pair | static eta | static lambda | M3 starts/gap | M4 starts/gap | M5 starts/gap | max structural error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for pair, result in results.items():
        checks = result["checks"]
        structural = [
            checks["saved_current_model2_likelihood_error"],
            checks["saved_current_model3_likelihood_error"],
            checks["model2_continuation_error"],
            checks["model3_zero_nesting_error"],
            checks["model3_continuation_error"],
            checks["model4"]["q_zero_nesting_error"],
            checks["model4"]["continuation_error"],
            checks["model5"]["q_zero_nesting_error"],
            checks["model5"]["continuation_error"],
        ]
        lines.append(
            f"| {FX[pair]['label']} | {result['model1']['eta']:.9g} | "
            f"{result['model1']['lambda']:.9g} | "
            f"{result['model3']['successful_starts']}/{result['model3']['largest_successful_objective_gap']:.6g} | "
            f"{result['model4']['successful_starts']}/{result['model4']['largest_successful_objective_gap']:.6g} | "
            f"{result['model5']['successful_starts']}/{result['model5']['largest_successful_objective_gap']:.6g} | "
            f"{max(structural):.3g} |"
        )
    lines += [
        "",
        "Complete daily scores, all structural comparisons, coefficients, state",
        "paths, estimation JSON, and numerical checks are stored in the per-pair",
        "subdirectories. Yahoo payloads are cached under `data/fx_daily_yahoo/`.",
    ]
    (OUTPUT_ROOT / "FX_IDENTITY_SM_SKEWT_FIVE_MODEL_RESULTS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_pair(pair: str) -> dict[str, object]:
    print(f"{pair}: fitting direct M1--M3 stage", flush=True)
    direct.run_dataset(pair)
    print(f"{pair}: fitting complete identity-SM M1--M5", flush=True)
    return identity.run_dataset(pair)


def summarize_all() -> None:
    results = {
        pair: json.loads((OUTPUT_ROOT / pair / "estimation.json").read_text(encoding="utf-8"))
        for pair in FX
    }
    identity.write_combined(list(results.values()))
    direct.write_combined([
        json.loads((DIRECT_OUTPUT_ROOT / pair / "estimation.json").read_text(encoding="utf-8"))
        for pair in FX
    ])
    m4_m3 = {pair: add_m4_vs_m3(pair) for pair in FX}
    pooled = pooled_m4_vs_m3()
    write_combined_report(results, m4_m3, pooled)
    plot_oos_heatmap()
    plot_coefficient_heatmap(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=(*FX, "all"), default="all")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    configure_imported_experiment()
    if args.summarize_only:
        summarize_all()
        return
    pairs = tuple(FX) if args.pair == "all" else (args.pair,)
    results = {pair: run_pair(pair) for pair in pairs}
    for pair in pairs:
        add_m4_vs_m3(pair)
    if set(pairs) == set(FX):
        summarize_all()


if __name__ == "__main__":
    main()
