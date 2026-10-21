"""Past-only rolling covariance PCA, followed by the locked Case-1 M1--M5.

The first principal component is estimated from centered, unscaled returns
over at most 252 strictly prior trading dates (minimum 126). Only stocks
observed on every date in that window enter that date's PCA. Current returns
do not affect stock eligibility, centering, or loadings. With current missing
observations, the factor score is least squares on observed eligible stocks.
Raw and residual targets use identical asset-days and are fitted separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh

import run_case1_factor_robustness_markets as market_factor
import run_case1_final_common_calendar_35_series as final

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output_case1_pca1_robustness_markets"
CACHE = ROOT.parent / "data" / "arc_sm_pca1_robustness"
WINDOW, MINIMUM = 252, 126
VERSION = "rolling-covariance-pc1-v1"


def json_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def rolling_pc1(values, window=WINDOW, minimum=MINIMUM):
    """Return same-shaped residuals, loadings and daily diagnostics.

    Loading signs are aligned using only earlier loadings, not future dates.
    No full-sample standardization or balanced-panel selection is performed.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not 2 <= minimum <= window:
        raise ValueError("expected date-by-stock returns and 2 <= minimum <= window")
    days, stocks = values.shape
    residuals = np.full_like(values, np.nan)
    loadings = np.full_like(values, np.nan)
    records = []
    previous = np.full(stocks, np.nan)
    for t in range(minimum, days):
        start = max(0, t - window)
        history = values[start:t]
        eligible = np.isfinite(history).all(axis=0)
        if eligible.sum() < 2:
            continue
        means = history[:, eligible].mean(axis=0)
        centered = history[:, eligible] - means
        variance = np.sum(centered * centered, axis=0)
        nonconstant = variance > 1e-16
        ids = np.flatnonzero(eligible)[nonconstant]
        centered = centered[:, nonconstant]
        means = means[nonconstant]
        if len(ids) < 2:
            continue
        # The smaller Gram matrix gives the same leading right singular vector.
        if centered.shape[0] <= centered.shape[1]:
            gram = centered @ centered.T
            eigenvalues, vectors = eigh(gram, subset_by_index=[len(gram)-1, len(gram)-1])
            loading = centered.T @ vectors[:, 0] / np.sqrt(eigenvalues[0])
        else:
            gram = centered.T @ centered
            eigenvalues, vectors = eigh(gram, subset_by_index=[len(gram)-1, len(gram)-1])
            loading = vectors[:, 0]
        loading /= np.linalg.norm(loading)
        overlap = np.isfinite(previous[ids])
        orientation = float(loading[overlap] @ previous[ids][overlap]) if overlap.any() else float(loading.sum())
        if orientation < 0:
            loading = -loading
        previous[:] = np.nan
        previous[ids] = loading
        loadings[t, ids] = loading
        available = np.isfinite(values[t, ids])
        observed_ids, observed_loading = ids[available], loading[available]
        norm = float(observed_loading @ observed_loading)
        if len(observed_ids) < 2 or norm < 1e-12:
            continue
        current = values[t, observed_ids] - means[available]
        score = float(observed_loading @ current / norm)
        common = observed_loading * score
        residuals[t, observed_ids] = current - common
        records.append({
            "row": t, "history_start_row": start, "history_end_row": t-1,
            "history_days": t-start, "eligible_stocks": len(ids),
            "observed_stocks": len(observed_ids), "pc1_score": score,
            "pc1_explained_fraction": float(eigenvalues[0] / np.sum(centered**2)),
            "raw_mean": float(values[t, observed_ids].mean()),
            "residual_mean": float(residuals[t, observed_ids].mean()),
            "raw_second_moment": float(np.mean(current**2)),
            "residual_second_moment": float(np.mean(residuals[t, observed_ids]**2)),
            "projection_error": float(abs(observed_loading @ residuals[t, observed_ids])),
        })
        if days > 500 and t % 500 == 0:
            print(f"PCA {t}/{days}: N={len(observed_ids)}, explained={records[-1]['pc1_explained_fraction']:.3f}", flush=True)
    return residuals, loadings, pd.DataFrame(records)


def prepare(market: str, resume=False):
    source = Path(market_factor.MARKETS[market]["source"])
    fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
    folder = CACHE / market
    cache_path = folder / "matched_panel.parquet"
    metadata_path = folder / "metadata.json"
    if resume and cache_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["source_sha256"] == fingerprint and metadata["version"] == VERSION:
            matched = pd.read_parquet(cache_path)
            return {name: market_factor.to_panel(matched, column) for name, column in
                    (("raw_matched", "raw_return"), ("pca_residual", "residual"))}, metadata
    data = pd.read_parquet(source, columns=["date", "firm_id", "log_return"])
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    data["firm_id"] = data["firm_id"].astype(str)
    data = data.loc[data.date.between(final.START, final.END)].copy()
    if data.duplicated(["date", "firm_id"]).any():
        raise ValueError("duplicate asset-days")
    wide = data.pivot(index="date", columns="firm_id", values="log_return").sort_index()
    values = wide.to_numpy(dtype=float)
    residuals, loadings, diagnostics = rolling_pc1(values)
    if diagnostics.empty:
        raise ValueError("no eligible PCA days")
    day, stock = np.nonzero(np.isfinite(residuals))
    matched = pd.DataFrame({"date": wide.index.to_numpy()[day],
                            "firm_id": wide.columns.to_numpy()[stock],
                            "raw_return": values[day, stock], "residual": residuals[day, stock]})
    diagnostics["date"] = wide.index.to_numpy()[diagnostics["row"].to_numpy()]
    diagnostics["history_end_date"] = wide.index.to_numpy()[diagnostics["history_end_row"].to_numpy()]
    assert (diagnostics.history_end_date < diagnostics.date).all()
    folder.mkdir(parents=True, exist_ok=True)
    matched.to_parquet(cache_path, index=False)
    pd.DataFrame(loadings, index=wide.index, columns=wide.columns).to_parquet(folder / "loadings.parquet")
    out = OUTPUT / market
    out.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(out / "pca_daily_diagnostics.csv", index=False)
    panel_hash = hashlib.sha256(pd.util.hash_pandas_object(matched, index=False).to_numpy().tobytes()).hexdigest()
    metadata = {
        "version": VERSION, "market": market, "source_sha256": fingerprint,
        "panel_sha256": panel_hash, "factors_removed": 1,
        "method": "rolling covariance PCA: demeaned returns, no variance standardization",
        "window": WINDOW, "minimum_history": MINIMUM,
        "eligibility": "complete strictly prior rolling window; no future-based stock selection",
        "current_missing": "least-squares projection on observed eligible stocks",
        "factor_t": "uses contemporaneous returns only to define the response residual; PCA parameters use t-1 information",
        "start": str(matched.date.min().date()), "end": str(matched.date.max().date()),
        "days": int(matched.date.nunique()), "asset_days": len(matched),
        "daily_N_min_median_max": [float(v) for v in matched.groupby("date").size().agg(["min", "median", "max"])],
        "pc1_explained_fraction_mean": float(diagnostics.pc1_explained_fraction.mean()),
        "maximum_projection_error": float(diagnostics.projection_error.max()),
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
    }
    json_write(metadata_path, metadata)
    json_write(out / "panel_diagnostics.json", metadata)
    return {name: market_factor.to_panel(matched, column) for name, column in
            (("raw_matched", "raw_return"), ("pca_residual", "residual"))}, metadata


def fit(market, target, panel, metadata, resume=False):
    key = f"{market}_{target}"
    path = OUTPUT / key / "estimation.json"
    if resume and path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("pca_robustness", {}).get("panel_sha256") == metadata["panel_sha256"]:
            print(f"{key}: using completed fit", flush=True)
            return saved
    old_loader, old_output, old_category = final.identity.load_dataset, final.OUTPUT, final.category
    label = f"{market_factor.MARKETS[market]['label']} {target} (PC1 matched)"
    info = {"dataset": key, "label": label, "market": market, "target": target}
    try:
        final.OUTPUT = OUTPUT
        final.identity.load_dataset = lambda _: (panel, info, {}, label)
        final.category = lambda _: "pca1_robustness"
        result = final.run_dataset(key)
    finally:
        final.identity.load_dataset, final.OUTPUT, final.category = old_loader, old_output, old_category
    result["analysis"] = "Case1 past-only PCA1 robustness"
    result["pca_robustness"] = metadata
    json_write(path, result)
    return result


def summarize():
    tests, levels, comparisons = [], [], []
    complete = []
    diagnostics = []
    for market in ("sp500", "n225"):
        for target in ("raw_matched", "pca_residual"):
            key = f"{market}_{target}"
            path = OUTPUT / key / "estimation.json"
            if not path.exists():
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            if "pca_robustness" not in result:
                continue
            common = {"dataset": key, "market": market, "target": target}
            tests.extend({**common, **r} for r in market_factor.test_rows(result))
            levels.extend({**common, **r} for r in result["oos_summary"])
            comparisons.extend({**common, **r} for r in result["oos_comparisons"])
            diagnostics.append({**common,
                "m3_objective_gap": result["model3"]["largest_successful_objective_gap"],
                "m3_mean_gradient": result["model3"]["maximum_absolute_mean_day_gradient"],
                "m4_objective_gap": result["model4"]["largest_successful_objective_gap"],
                "m5_objective_gap": result["model5"]["largest_successful_objective_gap"]})
            complete.append(key)
    if not complete:
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(tests).to_csv(OUTPUT / "pca1_wald_tests.csv", index=False)
    pd.DataFrame(levels).to_csv(OUTPUT / "pca1_oos_levels.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUTPUT / "pca1_oos_comparisons.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(OUTPUT / "pca1_optimizer_diagnostics.csv", index=False)
    lines = ["# Case 1: past-only PCA1 residual robustness", "",
             f"Completed cells: {', '.join(complete)} ({len(complete)}/4).", "",
             "Hansen skew-t M1--M5; one covariance principal component removed.",
             "PCA uses centered unscaled returns from the previous 126--252 trading days.",
             "Only stocks with a complete prior window enter; raw and residual samples match exactly.",
             "All estimators are refitted on data through 2023-12-29; OOS starts in January 2024.", "",
             "## Conditional HAC(22) Wald p-values", "",
             "| Market | Target | M3 eta | M3 lambda | M4 even | M4 odd | M5 even | M5 odd |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for key in complete:
        records = [r for r in tests if r["dataset"] == key]
        lookup = {(r["model"], r["block"]): r["p"] for r in records}
        market, target = key.split("_", 1)
        values = [lookup[x] for x in (("M3", "eta"),("M3", "lambda"),("M4", "even"),("M4", "odd"),("M5", "even"),("M5", "odd"))]
        lines.append(f"| {market} | {target} | " + " | ".join(f"{v:.6g}" for v in values) + " |")
    lines.extend(["", "## OOS improvements over parent models", "",
                  "Positive values favor M4/M5; p-values are two-sided HAC(22), unadjusted.", "",
                  "| Dataset | Comparison | Log score improvement (p) | Q-loss 1% improvement (p) |",
                  "|---|---|---:|---:|"])
    for key in complete:
        for child, parent in ((final.identity.M4, final.identity.M2),(final.identity.M5, final.identity.M3)):
            rows = {r["score"]: r for r in comparisons if r["dataset"] == key and r["proposed"] == child and r["benchmark"] == parent}
            cells = [f"{rows[s]['improvement_mean']:.8g} ({rows[s]['hac22_two_sided_p']:.6g})" for s in ("continuous_log_score", "q_loss_0.01")]
            lines.append(f"| {key} | {final.MODEL_SHORT[child]}-{final.MODEL_SHORT[parent]} | " + " | ".join(cells) + " |")
    lines.extend(["", "## OOS winning models", "",
                  "These are point-estimate rankings, not significance counts.", "",
                  "| Dataset | Metric | Winner |", "|---|---|---|"])
    winner_rows = []
    for key in complete:
        model_rows = [r for r in levels if r["dataset"] == key]
        for metric in final.scoring.SCORE_NAMES:
            winner = (max if metric in final.HIGHER_IS_BETTER else min)(model_rows, key=lambda r: r[metric])
            short = final.MODEL_SHORT[winner["model"]]
            winner_rows.append({"dataset": key, "metric": metric, "winner": short})
            lines.append(f"| {key} | {metric} | {short} |")
    pd.DataFrame(winner_rows).to_csv(OUTPUT / "pca1_oos_winners.csv", index=False)
    lines.extend(["", "## Optimization diagnostics", "",
                  "M3 can converge to inferior solutions from some starts; the best successful fit is used.",
                  "Individual start outcomes are retained in logs. A successful exit is not evidence of a global optimum.", "",
                  "| Dataset | M3 largest objective gap | M3 maximum mean gradient | M4 objective gap | M5 objective gap |",
                  "|---|---:|---:|---:|---:|"])
    for row in diagnostics:
        lines.append(f"| {row['dataset']} | {row['m3_objective_gap']:.6g} | {row['m3_mean_gradient']:.6g} | {row['m4_objective_gap']:.6g} | {row['m5_objective_gap']:.6g} |")
    lines.extend(["", "Inference is conditional on estimated upstream paths and PCA preprocessing.",
                  "Residualization does not establish cross-sectional independence.",
                  "Score levels across raw and residual targets are not directly comparable.", ""])
    (OUTPUT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def run_workers():
    """Run independent fits in child processes; retain logs and completion status."""
    from datetime import datetime, timezone
    import os
    environment = os.environ.copy()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[key] = "1"
    logs = OUTPUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    status = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "workers": {}}
    processes = []
    try:
        for market in ("sp500", "n225"):
            for target in ("raw_matched", "pca_residual"):
                key = f"{market}_{target}"
                stream = (logs / f"{key}.log").open("w", encoding="utf-8")
                command = [sys.executable, "-u", str(Path(__file__).resolve()), "--market", market,
                           "--target", target, "--resume", "--no-summary"]
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                           env=environment, cwd=ROOT.parent)
                processes.append((key, process, stream))
                status["workers"][key] = {"pid": process.pid, "status": "running", "log": str(logs / f"{key}.log")}
        json_write(OUTPUT / "run_status.json", status)
        pending = list(processes)
        while pending:
            for key, process, stream in pending[:]:
                code = process.poll()
                if code is not None:
                    stream.close()
                    status["workers"][key].update(status="complete" if code == 0 else "failed", exit_code=code)
                    pending.remove((key, process, stream))
                    summarize()
                    json_write(OUTPUT / "run_status.json", status)
            if pending:
                import time
                time.sleep(5)
        status["status"] = "complete" if all(p.returncode == 0 for _, p, _ in processes) else "failed"
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        json_write(OUTPUT / "run_status.json", status)
        summarize()
    finally:
        for _, _, stream in processes:
            if not stream.closed:
                stream.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=("sp500", "n225", "both"), default="both")
    parser.add_argument("--target", choices=("raw_matched", "pca_residual", "all"), default="all")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-workers", action="store_true")
    parser.add_argument("--no-summary", action="store_true")
    args = parser.parse_args()
    if args.run_workers:
        run_workers()
        return
    if args.summarize_only:
        summarize()
        return
    for market in (("sp500", "n225") if args.market == "both" else (args.market,)):
        panels, metadata = prepare(market, args.resume)
        assert np.array_equal(panels["raw_matched"].offsets, panels["pca_residual"].offsets)
        print(json.dumps(metadata), flush=True)
        if not args.prepare_only:
            for target in (("raw_matched", "pca_residual") if args.target == "all" else (args.target,)):
                fit(market, target, panels[target], metadata, args.resume)
                if not args.no_summary:
                    summarize()


if __name__ == "__main__":
    main()
