"""Case-1 symmetric-t M1--M5, frozen upstream, daily-cluster Wald(0).

Reuse the archived filter and OOS orchestration, with explicitly replaced
static and dynamic estimators. Lambda is zero in every reference state.
M3 has three free eta coefficients; M2/M4/M5 have six. No LR fits are run.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
from scipy.stats import chi2

import run_case1_conditional_lr as archived

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUTPUT = ROOT / "output_symmetric_t_m1_m5_calibration"
INPUT_MANIFEST = ROOT / "output_case1_conditional_lr/input_manifest.json"
SCHEMA = "symmetric-t-frozen-upstream-wald0-v1"
MAIN_FAMILIES = ("S&P 500", "N225", "S&P 500 sectors", "N225 sectors")


def source_hash():
    digest = hashlib.sha256()
    modules = (Path(__file__), Path(archived.__file__), Path(archived.kernels.__file__),
               Path(archived.core.__file__), Path(archived.shape.__file__),
               Path(archived.scoring.__file__), Path(archived.endpoint.__file__))
    for path in modules:
        digest.update(str(path.relative_to(REPO)).encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def fit_static_symmetric(values, warm):
    sigma = max(float(np.std(values)), 1e-6)
    unit = np.array([sigma, 1., 1.])

    def objective(theta):
        state = np.r_[theta * unit, 0.]
        try:
            likelihood, scores = archived.kernels.reference_day(values, 0, len(values), state)
            value, gradient = -likelihood, -scores[:3] * unit
            if np.isfinite(value) and np.isfinite(gradient).all():
                return value, gradient
        except (FloatingPointError, ValueError, OverflowError):
            pass
        return 1e100, np.zeros(3)

    starts = [np.r_[values.mean(), np.log(sigma * sigma),
                     archived.core.parameters_to_state(eta, 0.)[0]] / unit
              for eta in (3.5, 5., 8., 15.)]
    if warm:
        starts.insert(0, np.asarray(warm["model1"]["state"])[:3] / unit)
    fit = archived.maximize(objective, starts,
                           [(-.1/sigma, .1/sigma), (-20., 0.), (-12., 12.)], "M1 symmetric t")
    anchor = np.r_[np.asarray(fit["theta"]) * unit, 0.]
    eta, skew, _, _ = archived.core.state_to_parameters(anchor[2:])
    assert skew == 0.
    return {**fit, "state": anchor.tolist(), "mean": float(anchor[0]),
            "scale": float(np.exp(anchor[1]/2)), "eta": float(eta), "lambda": 0.,
            "parameter_units": unit.tolist(),
            "log_likelihood": float(-fit["mean_nll"] * len(values)),
            "shape_anchor_bound_reached": bool(abs(anchor[2]) >= 11.999)}


def active_run(run, indices):
    indices = np.asarray(indices, dtype=int)

    def reduced(theta, gradient):
        full = np.zeros(6)
        full[indices] = theta
        result = run(full, gradient)
        if gradient:
            result = (*result[:4], result[4][:, indices], *result[5:])
        return result

    return reduced


def wald_covariance(theta, run, counts, blocks):
    theta = np.asarray(theta, dtype=float)
    counts = np.asarray(counts, dtype=float)
    scores = counts[:, None] * run(theta, True)[4]
    if not np.isfinite(scores).all():
        raise FloatingPointError("nonfinite daily parameter scores")
    dimension = len(theta)
    hessian = np.empty((dimension, dimension))
    for index in range(dimension):
        step = np.zeros(dimension)
        step[index] = 1e-4 * max(1., abs(theta[index]))
        hessian[:, index] = -(counts @ run(theta+step, True)[4]
                             - counts @ run(theta-step, True)[4]) / (2*step[index])
    hessian = (hessian + hessian.T)/2
    centered = scores - scores.mean(axis=0, keepdims=True)
    meat = centered.T @ centered
    eigenvalues = np.linalg.eigvalsh(hessian)
    diagnostics = {"bandwidth": 0, "hessian": hessian.tolist(), "meat": meat.tolist(),
                   "hessian_eigenvalues": eigenvalues.tolist(), "valid": False,
                   "maximum_mean_gradient": float(abs(scores.sum(axis=0)/counts.sum()).max())}
    tests = [{"block": name, "df": len(selected), "wald_valid": False,
              "wald": None, "wald_p": None} for name, selected in blocks]
    if eigenvalues[0] <= 1e-10 * max(eigenvalues[-1], 1.):
        diagnostics["invalid_reason"] = "nonpositive or numerically singular Hessian"
        return tests, diagnostics
    inverse = np.linalg.inv(hessian)
    covariance = inverse @ meat @ inverse.T
    covariance = (covariance + covariance.T)/2
    eig = np.linalg.eigvalsh(covariance)
    diagnostics.update(covariance=covariance.tolist(), covariance_eigenvalues=eig.tolist())
    if eig[0] <= 1e-10 * max(eig[-1], np.finfo(float).tiny):
        diagnostics["invalid_reason"] = "nonpositive or numerically singular sandwich covariance"
        return tests, diagnostics
    for test, (_, selected) in zip(tests, blocks):
        selected = np.asarray(selected, dtype=int)
        statistic = float(theta[selected] @ np.linalg.solve(
            covariance[np.ix_(selected, selected)], theta[selected]))
        test.update(wald_valid=True, wald=statistic, wald_p=float(chi2.sf(statistic, len(selected))))
    diagnostics["valid"] = True
    return tests, diagnostics


def fit_layer_wald(run, units, counts, block_names, warm, label):
    indices = np.array([0, 2, 4]) if label == "M3" else np.arange(6)
    reduced = active_run(run, indices)
    dimension = len(indices)
    nobs = int(counts.sum())

    def objective(theta):
        try:
            filtered = reduced(np.asarray(theta, dtype=float), True)
            value = -float(counts @ filtered[3])/nobs
            gradient = -np.asarray(counts @ filtered[4])/nobs
            if np.isfinite(value) and np.isfinite(gradient).all():
                return value, gradient
        except (FloatingPointError, ValueError, OverflowError):
            pass
        return 1e100, np.zeros(dimension)

    rng = np.random.default_rng(20261005)
    # M3 uses exactly the eta coordinates of the archived six-coordinate starts.
    full_starts = [np.zeros(6), np.ones(6), -np.ones(6), rng.normal(size=6), 3*rng.normal(size=6)]
    starts = [theta[indices] for theta in full_starts]
    if warm is not None:
        starts.insert(0, (np.asarray(warm)/units).ravel()[indices])
    unrestricted = archived.maximize(objective, starts, [(-500., 500.)]*dimension, label)
    active_theta = np.asarray(unrestricted["theta"])
    full_theta = np.zeros(6)
    full_theta[indices] = active_theta
    fit = {"coefficients": (full_theta.reshape(units.shape)*units).tolist(),
           "parameter_units": units.tolist(),
           "unrestricted": {**unrestricted, "active_theta": active_theta.tolist(),
                            "theta": full_theta.tolist()},
           "active_indices": indices.tolist(), "free_dimension": dimension,
           "layer_log_likelihood": float(counts @ reduced(active_theta, False)[3]),
           "zero_layer_log_likelihood": float(counts @ reduced(np.zeros(dimension), False)[3]),
           "lr_tests": []}
    blocks = [("eta", [0, 1, 2])] if label == "M3" else [
        *block_names, ("joint", list(range(6)))]
    if CURRENT_INFERENCE:
        tests, diagnostic = wald_covariance(active_theta, reduced, counts, blocks)
    else:
        tests, diagnostic = [], {"computed": False, "reason": "OOS-only supplementary application"}
    fit.update(wald_tests=tests, wald_diagnostics=diagnostic)
    return fit, run(full_theta, False)


CURRENT_INFERENCE = True


def configure(output, inference):
    global CURRENT_INFERENCE
    CURRENT_INFERENCE = inference
    archived.OUTPUT = Path(output)
    archived.SCHEMA = SCHEMA
    archived.source_hash = source_hash
    archived.fit_static = fit_static_symmetric
    archived.fit_layer = fit_layer_wald
    for model in ("M1", "M2", "M3", "M4", "M5"):
        setattr(archived.old.identity, model, model)


def run_dataset(entry, output):
    configure(output, entry["role"] == "main")
    input_path = REPO/entry["input"]
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != entry["input_sha256"]:
        raise RuntimeError("input fingerprint changed: " + entry["dataset"])
    out = Path(output)/entry["dataset"]
    result = archived.run_dataset(entry)
    result["specification"] = {
        "reference": "symmetric standardized Student t; lambda exactly zero",
        "models": "M1 static t; M2 location/scale; M3 eta after M2; M4 even/odd after M2; M5 even/odd after M3",
        "likelihood": "sum of individual log densities, including actual daily N",
        "drivers": "daily means, HAR(1,5,22), zero presample scores",
        "projection": "fixed fitted M1 geometry, remove location/log variance only; no whitening",
        "static_sm": "identity, h0=1, q0=0",
        "inference": "fixed-upstream daily-cluster sandwich Wald, bandwidth zero" if CURRENT_INFERENCE else "OOS only",
        "role": entry["role"], "breakpoints": archived.old.base.BREAKPOINTS.tolist()}
    all_tests = []
    for model in ("M2", "M3", "M4", "M5"):
        all_tests.extend({"model": model, **test} for test in result["models"][model]["wald_tests"])
    result["wald_tests"] = all_tests
    assert result["models"]["M1"]["lambda"] == 0.
    assert np.all(np.asarray(result["models"]["M3"]["coefficients"])[:, 1] == 0.)
    result["audit"]["lambda_static"] = 0.
    result["audit"]["lambda_dynamic_coefficients_maximum"] = 0.
    archived.dump(out/"estimation.json", result)
    pd.DataFrame(all_tests).to_csv(out/"wald_tests.csv", index=False)
    legacy_file = out/"lr_tests.csv"
    if legacy_file.exists():
        # The reused orchestration writes an empty legacy table; no LR was fit.
        assert not legacy_file.read_text(encoding="utf-8").strip()
        legacy_file.unlink()
    return {"dataset": entry["dataset"], "completed": True, "inference": CURRENT_INFERENCE}


def aggregate(manifest, output):
    samples, parameters, tests, levels, comparisons, likelihoods, statuses = [], [], [], [], [], [], []
    for entry in manifest["entries"]:
        path = Path(output)/entry["dataset"]/"estimation.json"
        if not path.exists():
            statuses.append({"dataset": entry["dataset"], "complete": False})
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["source_hash"] != manifest["source_hash"] or result["input_sha256"] != entry["input_sha256"]:
            raise RuntimeError("mixed output fingerprints")
        if "reference" not in result["specification"]:
            statuses.append({"dataset": entry["dataset"], "complete": False})
            continue
        common = {key: entry[key] for key in ("dataset", "label", "family", "role")}
        samples.append(entry)
        statuses.append({"dataset": entry["dataset"], "complete": True})
        for model, fit in result["models"].items():
            parameters.append({**common, "model": model, "coefficients": json.dumps(
                fit["state"] if model == "M1" else fit["coefficients"]),
                "free_dimension": 3 if model == "M1" else fit["free_dimension"],
                "lambda_fixed_zero": True})
        tests.extend({**common, **row} for row in result["wald_tests"])
        levels.extend({**common, **row} for row in result["oos_summary"])
        comparisons.extend({**common, **row} for row in result["oos_comparisons"])
        likelihoods.extend({**common, "model": model, "training_log_likelihood": ll}
                           for model, ll in result["training_log_likelihoods"].items())
    for name, rows in (("samples", samples), ("parameters", parameters), ("wald_tests", tests),
                       ("oos_levels", levels), ("oos_comparisons", comparisons),
                       ("training_log_likelihoods", likelihoods)):
        pd.DataFrame(rows).to_csv(Path(output)/("case1_"+name+".csv"), index=False)
    status = {"schema": SCHEMA, "source_hash": manifest["source_hash"],
              "requested": len(statuses), "completed": sum(row["complete"] for row in statuses),
              "datasets": statuses, "invalid_wald": sum(not row["wald_valid"] for row in tests)}
    archived.dump(Path(output)/"run_status.json", status)
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--datasets", default="main")
    parser.add_argument("--include-stocks-oos", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output == archived.inputs.OUTPUT.resolve() or output == archived.old.OUTPUT.resolve():
        raise ValueError("new experiment must use a separate output directory")
    source = json.loads(INPUT_MANIFEST.read_text(encoding="utf-8"))
    entries = [{**entry, "role": "main"} for entry in source["entries"]
               if entry["target"] == "raw" and entry["family"] in MAIN_FAMILIES]
    assert len(entries) == 19
    if args.include_stocks_oos:
        entries.extend({**entry, "role": "OOS_only"} for entry in source["entries"]
                       if entry["target"] == "raw" and entry["family"] == "12 stocks")
    if args.datasets != "main":
        selected = args.datasets.split(",")
        entries = [entry for entry in entries if entry["dataset"] in selected]
        if len(entries) != len(selected):
            raise ValueError("unknown or duplicate datasets")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": SCHEMA, "source_hash": source_hash(), "entries": entries,
                "inference": "Wald(0), fixed upstream; no individual-stock parameter tests",
                "lambda": "static and dynamic zero", "model3_free_parameters": 3}
    existing_path = output/"input_manifest.json"
    if existing_path.exists() and json.loads(existing_path.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError("changed input/source/scope; use a separate output directory")
    archived.dump(existing_path, manifest)
    failures = []
    if not args.summarize_only:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            tasks = {pool.submit(run_dataset, entry, output): entry["dataset"] for entry in entries}
            for future in as_completed(tasks):
                try:
                    print(json.dumps(future.result()), flush=True)
                except Exception:
                    failures.append({"dataset": tasks[future], "traceback": traceback.format_exc()})
                    archived.dump(output/"failed_datasets.json", failures)
                    print("FAILED", tasks[future], failures[-1]["traceback"], flush=True)
                aggregate(manifest, output)
    status = aggregate(manifest, output)
    if failures or status["completed"] != len(entries):
        raise RuntimeError("incomplete empirical estimation; failures preserved")
    print("COMPLETE", json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
