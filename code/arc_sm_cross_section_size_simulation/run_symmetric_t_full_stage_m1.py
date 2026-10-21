"""Frozen own-N DGP, iid/common10, with M1--M5 estimated per replication.

The generator uses the Simulation evidence own-N score recursion and scaled coefficients. A fixed
0.01 change of return units allows reuse of the empirical reference kernels.
All reported OOS scores are converted back to the generator's return units.
Burn-in warms filters but is excluded from every likelihood and Wald meat.
"""
from __future__ import annotations

import os
for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import time
import traceback
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from numba import njit

import run_symmetric_t_observed_N_scaled as generator

ROOT, REPO = generator.ROOT, generator.REPO
estimator = generator.sim.estimator
e = estimator.archived
reference_day = e.kernels.reference_day
UNIT_SCALE = .01
OUTPUT = REPO / "work/simulation/checkpoints"
REPORT = REPO / "work/simulation/report"
MODELS = ("M1", "M2", "M3", "M4", "M5")
PAIRS = (("M2", "M3"), ("M2", "M4"), ("M3", "M5"), ("M2", "M5"), ("M4", "M5"))


@lru_cache(maxsize=1)
def source_hash():
    digest = hashlib.sha256(generator.fingerprint().encode())
    digest.update(estimator.source_hash().encode())
    digest.update(Path(__file__).read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def dump(path, value):
    e.dump(Path(path), value)


@njit(cache=True)
def static_likelihood(values, offsets, state):
    likelihood = 0.
    score = np.zeros(4)
    for t in range(len(offsets) - 1):
        contribution, derivative = reference_day(values, offsets[t], offsets[t + 1], state)
        likelihood += contribution
        score += derivative
    return likelihood, score


def fit_static(values, offsets):
    days = len(offsets) - 1
    weights = np.repeat(1 / np.diff(offsets), np.diff(offsets)) / days
    center = float(weights @ values)
    sigma = max(float(np.sqrt(weights @ ((values - center) ** 2))), 1e-6)
    units = np.array([sigma, 1., 1.])

    def objective(theta):
        state = np.r_[theta * units, 0.]
        try:
            likelihood, scores = static_likelihood(values, offsets, state)
            value, gradient = -likelihood / days, -scores[:3] * units / days
            if np.isfinite(value) and np.isfinite(gradient).all():
                return value, gradient
        except (FloatingPointError, ValueError, OverflowError):
            pass
        return 1e100, np.zeros(3)

    starts = [np.r_[center, np.log(sigma * sigma),
                   e.core.parameters_to_state(eta, 0.)[0]] / units
              for eta in (3.5, 5., 8., 15.)]
    bounds = [(-.1 / sigma, .1 / sigma), (-20., 0.), (-12., 12.)]
    fit = e.maximize(objective, starts, bounds, "M1 daily mean")
    anchor = np.r_[np.asarray(fit["theta"]) * units, 0.]
    eta, skew, _, _ = e.core.state_to_parameters(anchor[2:])
    assert skew == 0.
    fit.update(state=anchor.tolist(), mean=float(anchor[0]), scale=float(np.exp(anchor[1] / 2)),
               eta=float(eta), **{"lambda": 0.}, parameter_units=units.tolist(),
               log_likelihood=float(-fit["mean_nll"] * days),
               bound_reached=any(abs(x - a) < 1e-6 or abs(x - b) < 1e-6
                                 for x, (a, b) in zip(fit["theta"], bounds)))
    return fit


def retained_run(run, burn):
    """Exclude warmup samples, preserving derivatives through their history."""
    def retained(theta, gradient):
        result = list(run(theta, gradient))
        result[3] = result[3][burn:]
        if gradient:
            result[4] = result[4][burn:]
        return tuple(result)
    return retained


def parameter_units(drivers, burn, targets):
    averages = e.fixed_denominator_score_averages(drivers, (1, 5, 22))
    return np.array([[target / max(float(np.median(abs(a[burn:, j]))), 1e-12)
                      for j, target in enumerate(targets)] for a in averages])


def fit_dynamic(run, units, burn, training, model):
    blocks = [("location", [0, 2, 4]), ("scale", [1, 3, 5])] if model == "M2" else [
        ("even", [0, 1, 2]), ("odd", [3, 4, 5])]
    # The existing empirical optimizer uses five training-only starts, with
    # eta-only free coordinates for M3. No true parameter is supplied.
    fit, filtered = estimator.fit_layer_wald(retained_run(run, burn), units,
                                            np.ones(training), blocks, None, model)
    return fit, filtered


def endpoint_run(values, offsets, reference, geometry, burn):
    knots, masses, helmert, directions, projection = geometry
    pit = e.core.dynamic_reference_pit(values, offsets, reference[0])
    phi = np.ascontiguousarray(e.core.normalized_bspline_basis(pit, knots, 3, masses))
    loading = directions @ helmert.T
    zero = e.kernels.endpoint_filter(phi, offsets, reference[1], masses, loading, projection,
                                     np.zeros((2, 3)), np.ones((2, 3)), False)
    averages = e.fixed_denominator_score_averages(zero[1], (1, 5, 22))
    units = np.array([[1e-2 / max(float(np.median(abs(averages[h][burn:, b]))) *
                                  np.linalg.norm(directions[b]), 1e-12)
                       for h in range(3)] for b in range(2)])

    def run(theta, gradient):
        return e.kernels.endpoint_filter(phi, offsets, reference[1], masses, loading,
                                         projection, theta.reshape(2, 3) * units, units, gradient)
    return run, units


def fitted_geometry(anchor):
    knots = e.core.make_clamped_knots(e.old.base.BREAKPOINTS, 3)
    masses = e.core.bspline_masses(knots, 3)
    helmert = e.core.helmert_submatrix(len(masses))
    directions = np.vstack(e.endpoint.endpoint_directions(helmert))
    shape = e.shape.shape_geometry(anchor)
    spline = e.scoring.corrected_tangent_geometry(anchor, masses, knots, masses, helmert,
                                                 (0, 1), "estimated M1 in full-stage simulation")
    geometry = (knots, masses, helmert, directions, directions @ spline.projection_q_on_hansen)
    return shape, geometry


def validate_stage(model, fit, filtered, burn, anchor):
    audit = {"likelihood_reproduction_error": abs(float(filtered[3].sum()) - fit["layer_log_likelihood"])}
    if model in ("M2", "M3"):
        active = 0 if model == "M2" else 2
        audit["recursion_error"] = e.recursion_error(filtered[0][:, active:active + 2], filtered[2],
                                                    anchor[active:active + 2], np.asarray(fit["coefficients"]))
    else:
        audit["recursion_error"] = e.recursion_error(filtered[0], filtered[1], np.zeros(2),
                                                    np.asarray(fit["coefficients"]), True)
        audit["minimum_weight"] = float(filtered[2].min())
    assert audit["likelihood_reproduction_error"] < 1e-7, audit
    assert audit["recursion_error"] < 1e-9, audit
    return audit


def convert_scores_to_generator_units(scores):
    scores = scores.copy()
    scores["continuous_log_score"] += np.log(UNIT_SCALE)
    for name in e.scoring.SCORE_NAMES:
        if name == "crps" or name.startswith("q_loss_"):
            scores[name] /= UNIT_SCALE
    return scores


def evaluate_oos(values, offsets, split_day, burn, models, refs, geometry, output, name):
    """Rebuild causal full paths without refitting on OOS observations."""
    anchor = np.asarray(models["M1"]["state"])
    days = len(offsets) - 1
    blank = np.zeros((days, 4))
    unit = np.ones((3, 2))
    full_refs = {"M1": e.kernels.reference_filter(values, offsets, anchor, np.zeros((3, 2)),
                                                  unit, blank, np.zeros((2, 2)), False, False)}
    errors, audits = [], {}
    for model in ("M2", "M3"):
        if model not in models or model == "M3" and "M2" not in full_refs:
            continue
        fit = models[model]
        frozen = blank if model == "M2" else full_refs["M2"][0]
        projection = np.zeros((2, 2)) if model == "M2" else e.shape.shape_geometry(anchor).projection_shape_on_ms
        try:
            full_refs[model] = e.kernels.reference_filter(values, offsets, anchor,
                np.asarray(fit["coefficients"]), np.asarray(fit["parameter_units"]),
                frozen, projection, model == "M3", False)
            np.testing.assert_array_equal(full_refs[model][0][:split_day], refs[model][0])
            audits[model + "_training_full_path_equal"] = True
        except (FloatingPointError, ValueError, OverflowError) as error:
            errors.append(dict(model=model, stage="OOS reference", error=repr(error)))
    knots, masses, helmert, directions, projection = geometry
    start, split = int(offsets[burn]), int(offsets[split_day])
    oos_days = days - split_day
    dates = pd.date_range("2000-01-01", periods=oos_days, freq="D")
    frames = []

    def score(reference, correction, parent, child):
        with redirect_stdout(io.StringIO()):
            frame = e.scoring.score_oos_pair(values[split:], offsets[split_day:] - split, dates,
                values[start:split], e.as_reference(reference, split_day), correction,
                knots, masses, masses, parent, child)
        return frame.drop(columns="date")

    identity = SimpleNamespace(weights=np.tile(masses, (oos_days, 1)), contributions=np.zeros(oos_days))
    frame = score(full_refs["M1"], identity, "M1", "_discard")
    frames.append(frame.loc[frame.model.eq("M1")])
    for parent, child in (("M2", "M4"), ("M3", "M5")):
        if parent not in full_refs:
            continue
        reference = full_refs[parent]
        corrected = None
        if child in models:
            try:
                pit = e.core.dynamic_reference_pit(values, offsets, reference[0])
                phi = np.ascontiguousarray(e.core.normalized_bspline_basis(pit, knots, 3, masses))
                fit = models[child]
                corrected = e.kernels.endpoint_filter(phi, offsets, reference[1], masses,
                    directions @ helmert.T, projection, np.asarray(fit["coefficients"]),
                    np.asarray(fit["parameter_units"]), False)
                np.testing.assert_array_equal(corrected[0][:split_day], refs[child][0])
                audits[child + "_training_full_path_equal"] = True
            except (FloatingPointError, ValueError, OverflowError) as error:
                errors.append(dict(model=child, stage="OOS correction", error=repr(error)))
                corrected = None
        correction = identity if corrected is None else SimpleNamespace(
            weights=corrected[2][split_day:], contributions=corrected[3][split_day:])
        try:
            frame = score(reference, correction, parent, child)
        except (FloatingPointError, ValueError, OverflowError) as error:
            errors.append(dict(model=child, stage="OOS scoring", error=repr(error)))
            corrected = None
            frame = score(reference, identity, parent, child)
        if corrected is None:
            frame = frame.loc[frame.model.eq(parent)]
        for model, group in frame.groupby("model"):
            if not np.isfinite(group[list(e.scoring.SCORE_NAMES)].to_numpy()).all():
                errors.append(dict(model=model, stage="OOS scoring", error="nonfinite score"))
                continue
            expected = reference[3][split_day:]
            if model == child:
                expected = expected + corrected[3][split_day:]
            np.testing.assert_allclose(group.continuous_log_score, expected, rtol=1e-10, atol=1e-10)
            frames.append(group)
    daily = convert_scores_to_generator_units(pd.concat(frames, ignore_index=True))
    assert daily.groupby("model").size().eq(oos_days).all()
    daily_path = output / "daily" / (name + ".csv.gz")
    daily_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(daily_path, index=False, compression="gzip")
    levels = daily.groupby("model")[list(e.scoring.SCORE_NAMES)].mean()
    comparisons = []
    for parent, child in PAIRS:
        if parent in levels.index and child in levels.index:
            for metric in e.scoring.SCORE_NAMES:
                sign = 1 if "log_score" in metric else -1
                comparisons.append(dict(parent=parent, child=child, metric=metric,
                    improvement=float(sign * (levels.loc[child, metric] - levels.loc[parent, metric]))))
    return dict(levels=levels.reset_index().to_dict("records"), comparisons=comparisons,
                failures=errors, audit=audits, daily_sha256=hashlib.sha256(daily_path.read_bytes()).hexdigest())


def fit_case(entry, design, output, source, score_oos=True):
    name = entry["dataset"]
    path = output / "cases" / (name + ".json")
    if path.exists():
        result = generator.sim.load(path)
        assert result["source_sha256"] == source and result["entry"]["input_sha256"] == entry["input_sha256"]
        return result
    started = time.monotonic()
    input_path = REPO / entry["input"]
    assert hashlib.sha256(input_path.read_bytes()).hexdigest() == entry["input_sha256"]
    with np.load(input_path) as raw:
        values = np.ascontiguousarray(raw["values"] * UNIT_SCALE)
        offsets = raw["offsets"].copy()
        split_day, burn = int(raw["training"]), int(raw["burn"])
    split = int(offsets[split_day])
    tv, to = values[:split], offsets[:split_day + 1]
    checkpoint_path = output / "checkpoints" / (name + ".json")
    checkpoint = generator.sim.load(checkpoint_path) if checkpoint_path.exists() else {}
    if checkpoint:
        assert checkpoint["source_sha256"] == source and checkpoint["input_sha256"] == entry["input_sha256"]
    checkpoint.update(source_sha256=source, input_sha256=entry["input_sha256"])
    result = dict(entry=entry, source_sha256=source, models={}, tests=[], failures=[], audit={}, oos={})
    try:
        if "M1" not in checkpoint:
            start = int(to[burn])
            checkpoint["M1"] = fit_static(tv[start:], to[burn:] - start)
            dump(checkpoint_path, checkpoint)
        result["models"]["M1"] = checkpoint["M1"]
        anchor = np.asarray(checkpoint["M1"]["state"])
        shape, geometry = fitted_geometry(anchor)
        result["audit"]["geometry_anchor"] = anchor.tolist()
        static = e.kernels.reference_filter(tv, to, anchor, np.zeros((3, 2)), np.ones((3, 2)),
                                            np.zeros((split_day, 4)), np.zeros((2, 2)), False, False)
        result["audit"]["M1_likelihood_error"] = abs(float(static[3][burn:].sum()) - checkpoint["M1"]["log_likelihood"])
        assert result["audit"]["M1_likelihood_error"] < 1e-7
    except Exception as error:
        result["failures"].append(dict(model="M1", stage="M1/geometry", error=repr(error), traceback=traceback.format_exc()))
        result.update(success=False, seconds=time.monotonic() - started)
        dump(path, result)
        return result
    refs = {"M1": static}
    for model in MODELS[1:]:
        try:
            if model == "M2":
                units = parameter_units(static[2], burn, (1e-4, 1e-2))
                def run(theta, gradient):
                    return e.kernels.reference_filter(tv, to, anchor, theta.reshape(3, 2) * units,
                        units, np.zeros((split_day, 4)), np.zeros((2, 2)), False, gradient)
            elif model == "M3":
                null = e.kernels.reference_filter(tv, to, anchor, np.zeros((3, 2)), np.ones((3, 2)),
                                                  refs["M2"][0], shape.projection_shape_on_ms, True, False)
                units = parameter_units(null[2], burn, (1e-2, 1e-2))
                def run(theta, gradient):
                    return e.kernels.reference_filter(tv, to, anchor, theta.reshape(3, 2) * units,
                        units, refs["M2"][0], shape.projection_shape_on_ms, True, gradient)
            else:
                parent = "M2" if model == "M4" else "M3"
                run, units = endpoint_run(tv, to, refs[parent], geometry, burn)
            if model not in checkpoint:
                checkpoint[model], _ = fit_dynamic(run, units, burn, design.training, model)
                dump(checkpoint_path, checkpoint)
            fit = checkpoint[model]
            np.testing.assert_array_equal(units, fit["parameter_units"])
            filtered = run(np.asarray(fit["unrestricted"]["theta"]), False)
            retained = retained_run(run, burn)(np.asarray(fit["unrestricted"]["theta"]), False)
            result["audit"][model] = validate_stage(model, fit, retained, burn, anchor)
            refs[model] = filtered
            result["models"][model] = fit
            result["tests"].extend(dict(model=model, **test) for test in fit["wald_tests"])
            assert model != "M3" or np.all(np.asarray(fit["coefficients"])[:, 1] == 0.)
        except Exception as error:
            result["failures"].append(dict(model=model, stage="estimation/Wald", error=repr(error), traceback=traceback.format_exc()))
    if score_oos:
        try:
            result["oos"] = evaluate_oos(values, offsets, split_day, burn, result["models"], refs, geometry, output, name)
        except Exception as error:
            result["failures"].append(dict(model="all", stage="OOS", error=repr(error), traceback=traceback.format_exc()))
    result.update(success=len(result["models"]) == 5, seconds=time.monotonic() - started)
    dump(path, result)
    print(name, "COMPLETE", result["success"], f"{result['seconds']:.1f}s", flush=True)
    return result


def one_case(task, design, output, source, score_oos):
    r, dgp, case, n = task
    output = Path(output)
    name = f"rep{r:04d}_{dgp}_{case}_N{n}"
    path = output / "cases" / (name + ".json")
    if path.exists():
        result = generator.sim.load(path)
        assert result["source_sha256"] == source
        return name
    try:
        entry = generator.generate(r, dgp, case, n, design, output)
    except Exception as error:
        entry = dict(dataset=name, replication=r, dgp=dgp, case=case, N=n)
        dump(path, dict(entry=entry, source_sha256=source, models={}, tests=[], oos={}, audit={}, success=False,
             failures=[dict(model="generation", stage="generation", error=repr(error), traceback=traceback.format_exc())]))
        print(name, "GENERATION FAILED", repr(error), flush=True)
        return name
    fit_case(entry, design, output, source, score_oos)
    return name


def case_validation(result, score_oos=True):
    """A written case record is not proof that its numerical work succeeded."""
    reasons = []
    if not result['success'] or set(result['models']) != set(MODELS):
        reasons.append('missing model fits')
    expected = {('M2', 'location'), ('M2', 'scale'), ('M2', 'joint'), ('M3', 'eta'),
                ('M4', 'even'), ('M4', 'odd'), ('M4', 'joint'),
                ('M5', 'even'), ('M5', 'odd'), ('M5', 'joint')}
    tests = result['tests']
    if (len(tests) != len(expected) or {(row['model'], row['block']) for row in tests} != expected
            or any(not row['wald_valid'] or row['wald_p'] is None
                   or not np.isfinite(row['wald_p']) or not 0 <= row['wald_p'] <= 1
                   or row['wald'] is None or not np.isfinite(row['wald']) or row['wald'] < 0
                   for row in tests)):
        reasons.append('missing or invalid Wald tests')
    if result['failures'] or result['oos'].get('failures'):
        reasons.append('recorded numerical failures')
    if score_oos:
        levels = result['oos'].get('levels', [])
        if (len(levels) != len(MODELS) or {row['model'] for row in levels} != set(MODELS)
                or any(not np.isfinite([row.get(metric, np.nan) for metric in e.scoring.SCORE_NAMES]).all()
                       for row in levels)):
            reasons.append('missing or nonfinite OOS model scores')
    return reasons


def summarize(output, report, tasks, source, state, worker_errors, score_oos=True):
    report.mkdir(parents=True, exist_ok=True)
    saved = []
    for r, dgp, case, n in tasks:
        path = generator.sim.case_path(output, r, dgp, case, n)
        if path.exists():
            result = generator.sim.load(path)
            assert result["source_sha256"] == source
            saved.append(result)
    tests, failures, parameters, levels, comparisons = [], [], [], [], []
    for result in saved:
        common = {k: result["entry"][k] for k in ("dataset", "replication", "dgp", "case", "N")}
        tests.extend({**common, **row} for row in result["tests"])
        failures.extend({**common, **{k: v for k, v in row.items() if k != "traceback"}}
                        for row in result["failures"] + result["oos"].get("failures", []))
        levels.extend({**common, **row} for row in result["oos"].get("levels", []))
        comparisons.extend({**common, **row} for row in result["oos"].get("comparisons", []))
        for model, fit in result["models"].items():
            parameters.append({**common, "model": model, "eta": fit.get("eta"),
                "bound_reached": fit.get("bound_reached", False),
                "coefficients": json.dumps(fit.get("coefficients", fit.get("state"))),
                "projected_gradient_max": fit["projected_gradient_max"] if model == "M1"
                    else fit["unrestricted"]["projected_gradient_max"]})
    frame = pd.DataFrame(tests)
    rows = []
    combinations = sorted(set((d, c, n) for _, d, c, n in tasks))
    grouped = dict(tuple(frame.groupby(["dgp", "case", "N", "model", "block"]))) if len(frame) else {}
    for dgp, case, n in combinations:
        for model in MODELS[1:]:
            blocks = ("eta",) if model == "M3" else (("location", "scale", "joint") if model == "M2" else ("even", "odd", "joint"))
            for block in blocks:
                keys = (dgp, case, n, model, block)
                group = grouped.get(keys, pd.DataFrame(columns=["wald_valid", "wald_p"]))
                valid = group.loc[group.wald_valid.astype(bool)]
                count = len(valid)
                rejected = int((valid.wald_p < .05).sum())
                lower, upper = generator.sim.legacy.wilson(rejected, count) if count else (None, None)
                expected = sum((d, c, sample) == keys[:3] for _, d, c, sample in tasks)
                rows.append({**dict(zip(("dgp", "case", "N", "model", "block"), keys)),
                    "requested": expected, "attempted_tests": len(group), "valid": count,
                    "invalid": len(group) - count, "missing_tests": expected - len(group), "rejected": rejected,
                    "rejection_rate": rejected / count if count else None, "wilson_lower": lower, "wilson_upper": upper})
    pd.DataFrame(rows).to_csv(report / "rejection_summary.csv", index=False)
    for name, records in (("replication_tests", tests), ("failures", failures), ("parameters", parameters),
                           ("oos_levels", levels), ("oos_comparisons", comparisons)):
        pd.DataFrame(records).to_csv(report / (name + ".csv"), index=False)
    oos_summary = []
    if comparisons:
        comparison_frame = pd.DataFrame(comparisons)
        for keys, group in comparison_frame.groupby(["dgp", "case", "N", "parent", "child", "metric"]):
            gains = group.improvement
            oos_summary.append({**dict(zip(("dgp", "case", "N", "parent", "child", "metric"), keys)),
                "valid": len(gains), "mean_improvement": float(gains.mean()),
                "median_improvement": float(gains.median()),
                "mc_standard_error": float(gains.std(ddof=1) / np.sqrt(len(gains))) if len(gains) > 1 else None,
                "positive_improvements": int((gains > 0).sum())})
    pd.DataFrame(oos_summary).to_csv(report / "oos_summary.csv", index=False)
    failed_settings = [dict(dataset=r['entry']['dataset'], reasons=reasons) for r in saved
                       if (reasons := case_validation(r, score_oos))]
    processing_complete = len(saved) == len(tasks) and not worker_errors
    validation_passed = bool(processing_complete and not failed_settings)
    status = dict(state=('failed' if state == 'complete' and not validation_passed else state),
                  completed=len(saved), expected=len(tasks),
                  all_models_fitted=sum(r["success"] for r in saved),
                  oos_settings=sum(bool(r["oos"].get("levels")) for r in saved),
                  valid_tests=sum(bool(t["wald_valid"]) for t in tests), invalid_tests=sum(not t["wald_valid"] for t in tests),
                  failure_records=len(failures), worker_errors=worker_errors,
                  source_sha256=source, updated_utc=datetime.now(timezone.utc).isoformat(),
                  processing_complete=processing_complete, validation_passed=validation_passed,
                  failed_settings=failed_settings, score_oos_required=score_oos,
                  finished=validation_passed)
    dump(report / "run_status.json", status)
    dump(output / "run_status.json", status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--training", type=int, default=1886)
    parser.add_argument("--oos", type=int, default=642)
    parser.add_argument("--burn", type=int, default=500)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[1, 20, 50, 100])
    parser.add_argument("--dgps", nargs="+", choices=generator.sim.DGPS, default=list(generator.sim.DGPS))
    parser.add_argument("--cases", nargs="+", choices=generator.CASES, default=list(generator.CASES))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    if min(args.replications, args.workers, args.oos, *args.sample_sizes) < 1 or args.training <= 22 or args.burn < 0:
        parser.error("positive counts, training > 22, burn >= 0 required")
    design = generator.sim.Design(training=args.training, oos=args.oos, burn=args.burn,
                                   common_share=.10, sample_sizes=tuple(args.sample_sizes))
    output, report = args.output.resolve(), args.report.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    source = source_hash()
    tasks = [(r, d, c, n) for r in range(args.replications) for d in args.dgps for c in args.cases
             for n in args.sample_sizes if c == "iid" or n != 1]
    metadata = dict(source_sha256=source, generator_source_sha256=generator.fingerprint(),
        design=asdict(design), replications=args.replications, dgps=args.dgps, cases=args.cases,
        estimation="M1 static symmetric t -> M2 location/scale -> M3 eta; M4 after M2; M5 after M3",
        fisher_geometry="recomputed at the fitted M1 anchor per setting; m/log variance removed, no whitening",
        inference="conditional daily mean sandwich Wald(0); fitted upstream held fixed in downstream covariance",
        initialization="same observed burn-in as the known-M2 run; excluded from M1 and dynamic likelihoods and Wald meat",
        unit_scale=UNIT_SCALE, reported_score_units="original generator return units",
        no_truth_in_estimation=True, score_oos=not args.fit_only,
        generation="unchanged own-N generator, fixed SP500 coefficients times sqrt(N/500), same seed streams",
        calibration={k: v.tolist() for k, v in generator.BASE_CALIBRATION().items()}, expected_settings=len(tasks))
    metadata["environment"] = {"python": sys.version, "packages": {
        package: version(package) for package in ("numpy", "pandas", "scipy", "numba")}}
    metadata = json.loads(json.dumps(metadata))
    if (output / "design.json").exists():
        assert generator.sim.load(output / "design.json") == metadata
    dump(output / "design.json", metadata)
    dump(report / "design.json", metadata)
    errors = {}
    summarize(output, report, tasks, source, "running", errors, not args.fit_only)
    pending = [task for task in tasks if not generator.sim.case_path(output, *task).exists()]
    if not args.finalize_only:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(one_case, task, design, output, source, not args.fit_only): task for task in pending}
            completed = 0
            for future in as_completed(futures):
                task = futures[future]
                try:
                    future.result()
                except Exception:
                    errors[str(task)] = traceback.format_exc()
                    print("WORKER FAILED", task, errors[str(task)], flush=True)
                completed += 1
                if completed % 10 == 0 or completed == len(pending):
                    status = summarize(output, report, tasks, source, "running", errors, not args.fit_only)
                    print("PROGRESS", status, flush=True)
    status = summarize(output, report, tasks, source, "complete", errors, not args.fit_only)
    print("FINAL", status, flush=True)
    if not status["finished"]:
        raise RuntimeError("Simulation validation failed. Inspect run_status.json and report/failures.csv.")


if __name__ == "__main__":
    import run_symmetric_t_full_stage_m1 as entrypoint
    entrypoint.main()
