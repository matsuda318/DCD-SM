"""Gaussian-copula experiments using the empirical M1--M5 estimator.

Use the unchanged empirical observation-sum estimator and restricted LR fits.
Add conditional daily-cluster HAC(22) sandwich Wald tests to the SAME fits.
True reference/state paths are used for generation only, never for estimation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import pandas as pd
from scipy.special import ndtr, logsumexp
from scipy.stats import chi2

import run_simulation as sim
import run_common_factor_simulation as copula
import run_case1_conditional_lr as empirical
from estimate_score_har import hac_meat, numerical_hessian

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUTPUT = ROOT / "output_full_stage_copula_lr_wald"
CASES = ("iid", "shared_factor")
BLOCKS = {
    "M2": (("location", [0, 2, 4]), ("scale", [1, 3, 5])),
    "M3": (("eta", [0, 2, 4]), ("lambda", [1, 3, 5])),
    "M4": (("even", [0, 1, 2]), ("odd", [3, 4, 5])),
    "M5": (("even", [0, 1, 2]), ("odd", [3, 4, 5])),
}


@dataclass(frozen=True)
class Design:
    training: int = 1886
    oos: int = 642
    burn: int = 500
    seed: int = 20261005
    common_share: float = .30
    sample_sizes: tuple[int, ...] = (1, 20, 50, 100)
    dgps: tuple[str, ...] = sim.DGP_NAMES


def fingerprint():
    paths = [Path(__file__), Path(sim.__file__), Path(copula.__file__),
             Path(empirical.__file__), sim.MODEL_ROOT / "case1_lr_kernels.py",
             REPO / "dcd_sp500_daily_composite_score_arch" / "estimate_score_har.py",
             sim.CALIBRATION]
    h = hashlib.sha256()
    for path in paths:
        h.update(str(path.relative_to(REPO)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def dump(path, value):
    empirical.dump(path, value)


class IndependentNormalSampler:
    """Same marginal inverse CDF as the common-factor sampler, independent Phi(Z)."""
    def __init__(self, grid, cdf):
        self.grid, self.cdf = grid, cdf

    def __call__(self, count, q_state, masses, helmert, inverse, rng, *, reference_state):
        logits = np.log(masses) + helmert @ q_state
        weights = np.exp(logits - logsumexp(logits))
        mixture = np.maximum.accumulate(self.cdf @ weights)
        mixture[0], mixture[-1] = 0, 1
        return np.clip(np.interp(ndtr(rng.standard_normal(count)), mixture, self.grid), 1e-12, 1-1e-12)


def generate(replication, design, output):
    """Legacy master-market DGP and coefficients, rescaled to raw-return units."""
    sim.TRAINING, sim.OOS, sim.BURN = design.training, design.oos, design.burn
    knots = sim.core.make_clamped_knots(sim.base.BREAKPOINTS, 3)
    masses = sim.core.bspline_masses(knots, 3)
    helmert = sim.core.helmert_submatrix(len(masses))
    directions = sim.endpoint.endpoint_directions(helmert)
    grid, _, cdf, inverse = sim.prepare_component_sampler(knots, masses)
    calibrated = sim.calibration()
    anchor = np.asarray(calibrated["anchor"])
    shape_geometry = sim.shape.shape_geometry(anchor)
    q_geometry = sim.scoring.corrected_tangent_geometry(
        anchor, masses, knots, masses, helmert, (0, 1), "legacy master DGP")
    # Constant change of units only: empirical filters require raw-return scales.
    original_anchor = json.loads(sim.CALIBRATION.read_text(encoding="utf-8"))["model1"]["state"]
    scale = float(np.exp(.5 * original_anchor[1]))
    dates = pd.bdate_range(end=empirical.old.TRAINING_END, periods=design.training).append(
        pd.bdate_range(start="2024-01-02", periods=design.oos)).to_numpy(dtype="datetime64[ns]")
    cache = REPO / "data" / output.name / f"replication_{replication:04d}"
    cache.mkdir(parents=True, exist_ok=True)
    entries, diagnostics = [], []
    for dgp_index, dgp in enumerate(design.dgps):
        for case in CASES:
            rng = np.random.default_rng(design.seed + replication + 100000*dgp_index)
            sampler = (IndependentNormalSampler(grid, cdf) if case=="iid"
                       else copula.CommonFactorSampler(grid, cdf, design.common_share))
            panel_cache = cache / f"{dgp}_{case}_master.npz"
            audit_cache = cache / f"{dgp}_{case}_sampler.csv"
            if panel_cache.exists():
                with np.load(panel_cache) as saved:
                    values = saved["values"]
            else:
                master = sim.generate_master_panel(dgp, rng, calibrated, knots, masses,
                    helmert, directions, shape_geometry, q_geometry.projection_q_on_hansen,
                    inverse, sample_state_uniforms=sampler)
                values = scale * master["values"]
                np.savez_compressed(panel_cache, values=values)
                if case=="shared_factor":
                    pd.DataFrame(sampler.audit).to_csv(audit_cache, index=False)
            if case=="shared_factor":
                audit = pd.read_csv(audit_cache)
                diagnostics.append({"replication":replication, "dgp":dgp,
                    "days":len(audit), "maximum_common_share_error":float(audit.share_error.abs().max()),
                    "latent_rho_min":float(audit.latent_rho.min()),
                    "latent_rho_max":float(audit.latent_rho.max())})
            for n in design.sample_sizes:
                key = f"rep{replication:04d}_{dgp}_{case}_N{n}"
                path = cache / f"{dgp}_{case}_N{n}.npz"
                if not path.exists():
                    flat = np.ascontiguousarray(values[:, :n]).ravel()
                    offsets = np.arange(len(dates)+1, dtype=np.int64)*n
                    np.savez_compressed(path, values=flat, offsets=offsets, dates=dates)
                entries.append({"dataset":key, "label":key, "family":"simulation",
                    "target":case, "parent_dataset":dgp, "input":str(path.relative_to(REPO)),
                    "input_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                    "replication":replication, "dgp":dgp, "case":case, "N":n})
    dump(output / "generation" / f"replication_{replication:04d}.json",
         {"entries":entries, "diagnostics":diagnostics})
    return entries


def training_runs(result):
    """Reconstruct the exact frozen-upstream filters used in the empirical fits."""
    with np.load(REPO / result["dataset"]["input"]) as data:
        train = int((data["dates"] <= np.datetime64(empirical.old.TRAINING_END)).sum())
        offsets = data["offsets"][:train+1].copy()
        values = data["values"][:int(offsets[-1])].copy()
    anchor = np.asarray(result["models"]["M1"]["state"])
    blank = np.zeros((train, 4))
    k = empirical.kernels
    runs = {}
    fit2 = result["models"]["M2"]
    units2 = np.asarray(fit2["parameter_units"])
    def run2(theta, gradient):
        return k.reference_filter(values, offsets, anchor, theta.reshape(3,2)*units2,
            units2, blank, np.zeros((2,2)), False, gradient)
    runs["M2"] = run2
    f2 = run2(np.asarray(fit2["unrestricted"]["theta"]), False)
    projection3 = empirical.shape.shape_geometry(anchor).projection_shape_on_ms
    fit3 = result["models"]["M3"]
    units3 = np.asarray(fit3["parameter_units"])
    def run3(theta, gradient):
        return k.reference_filter(values, offsets, anchor, theta.reshape(3,2)*units3,
            units3, f2[0], projection3, True, gradient)
    runs["M3"] = run3
    f3 = run3(np.asarray(fit3["unrestricted"]["theta"]), False)
    knots = empirical.core.make_clamped_knots(empirical.old.base.BREAKPOINTS, 3)
    masses = empirical.core.bspline_masses(knots, 3)
    helmert = empirical.core.helmert_submatrix(len(masses))
    directions = np.vstack(empirical.endpoint.endpoint_directions(helmert))
    geometry = empirical.scoring.corrected_tangent_geometry(
        anchor, masses, knots, masses, helmert, (0,1), "same fitted M1 geometry")
    loading, projection = directions @ helmert.T, directions @ geometry.projection_q_on_hansen
    for model, frozen in (("M4", f2), ("M5", f3)):
        u = empirical.core.dynamic_reference_pit(values, offsets, frozen[0])
        phi = np.ascontiguousarray(empirical.core.normalized_bspline_basis(u, knots, 3, masses))
        units = np.asarray(result["models"][model]["parameter_units"])
        def run(theta, gradient, phi=phi, units=units, scores=frozen[1]):
            return k.endpoint_filter(phi, offsets, scores, masses, loading, projection,
                theta.reshape(2,3)*units, units, gradient)
        runs[model] = run
    return runs, np.diff(offsets)


def sandwich(theta, run, counts, blocks, bandwidth=22):
    """Hessian and HAC meat of DAILY SUM parameter scores; no extra N/T factors."""
    theta = np.asarray(theta, dtype=float)
    counts = np.asarray(counts, dtype=float)
    fitted = run(theta, True)
    clusters = counts[:, None] * fitted[4]
    if not np.isfinite(clusters).all():
        raise FloatingPointError("nonfinite parameter-score clusters")
    # Derivatives of the full recursive gradient, including all six coefficients.
    hessian = np.empty((6,6))
    for j in range(6):
        shift = np.zeros(6); shift[j] = 1e-4 * max(1., abs(theta[j]))
        plus = counts @ run(theta+shift, True)[4]
        minus = counts @ run(theta-shift, True)[4]
        hessian[:,j] = -(plus-minus)/(2*shift[j])
    hessian = (hessian+hessian.T)/2
    eigenvalues = np.linalg.eigvalsh(hessian)
    condition = float(np.linalg.cond(hessian))
    diagnostic = {"hessian_eigenvalues":eigenvalues.tolist(),
        "hessian_condition":condition if np.isfinite(condition) else None,
        "maximum_mean_gradient":float(np.max(np.abs(clusters.sum(axis=0)/counts.sum()))),
        "bandwidth":bandwidth, "valid":False}
    tests = [{"block":name, "df":len(indices), "wald_valid":False,
              "wald":None, "wald_p":None} for name,indices in (*blocks, ("joint",list(range(6))))]
    if eigenvalues[0] <= 1e-10*max(eigenvalues[-1], 1.):
        diagnostic["invalid_reason"] = "nonpositive or numerically singular Hessian"
        return tests, diagnostic
    inverse = np.linalg.inv(hessian)
    meat = hac_meat(clusters, min(bandwidth, len(clusters)-1))
    covariance = inverse @ meat @ inverse
    covariance = (covariance+covariance.T)/2
    eig = np.linalg.eigvalsh(covariance)
    diagnostic.update(covariance_eigenvalues=eig.tolist(), covariance=covariance.tolist())
    if eig[0] <= 1e-10*max(eig[-1], np.finfo(float).tiny):
        diagnostic["invalid_reason"] = "nonpositive or numerically singular sandwich covariance"
        return tests, diagnostic
    for test, (_, indices) in zip(tests, (*blocks, ("joint",list(range(6))))):
        indices = np.asarray(indices)
        statistic = float(theta[indices] @ np.linalg.solve(covariance[np.ix_(indices,indices)],theta[indices]))
        test.update(wald_valid=True, wald=statistic, wald_p=float(chi2.sf(statistic,len(indices))))
    diagnostic["valid"] = True
    return tests, diagnostic


def fit_case(entry, output, source):
    destination = output / "cases" / (entry["dataset"]+".json")
    if destination.exists():
        return json.loads(destination.read_text(encoding="utf-8"))
    started = time.monotonic()
    empirical.OUTPUT = output / "fits"
    record = {"entry":entry, "source_sha256":source, "success":False,
              "oos_success":False, "tests":[], "oos_levels":[], "oos_comparisons":[]}
    try:
        try:
            result = empirical.run_dataset(entry)
            record["oos_success"] = True
        except Exception as forecast_error:
            # A future unstable path must not erase valid training-sample tests.
            checkpoint_path = empirical.OUTPUT / entry["dataset"] / "fit_checkpoint.json"
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {}
            if not all(m in checkpoint for m in ("M1","M2","M3","M4","M5")):
                raise
            record["oos_error"] = repr(forecast_error)
            result = {"dataset":entry, "models":checkpoint, "oos_summary":[], "oos_comparisons":[],
                      "audit":{}, "training_log_likelihoods":{}}
        if result["models"]["M1"]["shape_anchor_bound_reached"]:
            raise RuntimeError("M1 shape anchor reached its numerical bound")
        runs, counts = training_runs(result)
        diagnostics = {}
        for model in ("M2", "M3", "M4", "M5"):
            fit = result["models"][model]
            theta = np.asarray(fit["unrestricted"]["theta"])
            reproduced = float(counts @ runs[model](theta, False)[3])
            if abs(reproduced-fit["layer_log_likelihood"]) > 1e-7:
                raise RuntimeError(f"{model}: likelihood reconstruction failed")
            try:
                wald_tests, diagnostic = sandwich(theta, runs[model], counts, BLOCKS[model])
            except Exception as error:
                diagnostic = {"valid":False, "invalid_reason":repr(error)}
                wald_tests = [{"block":t["block"],"df":t["df"],"wald_valid":False,
                               "wald":None,"wald_p":None} for t in fit["lr_tests"]]
            diagnostics[model] = diagnostic
            for lr_test, wald_test in zip(fit["lr_tests"], wald_tests):
                assert lr_test["block"] == wald_test["block"]
                record["tests"].append({"model":model, **lr_test, **wald_test,
                                        "lr_valid":True})
        record.update(success=True, wald_diagnostics=diagnostics,
            oos_levels=result["oos_summary"], oos_comparisons=result["oos_comparisons"],
            training_log_likelihoods=result["training_log_likelihoods"],
            recursion_audit=result["audit"],
            static_estimate={k:result["models"]["M1"][k] for k in ("mean","scale","eta","lambda")})
    except Exception as error:
        record.update(error=repr(error), traceback=traceback.format_exc())
        print(entry["dataset"], "FAILED", repr(error), flush=True)
    record["seconds"] = time.monotonic()-started
    dump(destination, record)
    return record


def one_replication(replication, design, output, source):
    output = Path(output)
    generated = output / "generation" / f"replication_{replication:04d}.json"
    if generated.exists():
        entries = json.loads(generated.read_text(encoding="utf-8"))["entries"]
    else:
        print("GENERATE", replication, flush=True)
        entries = generate(replication, design, output)
    for entry in entries:
        fit_case(entry, output, source)
    return replication


def wilson(rejected, count):
    if not count:
        return None, None
    z = 1.959963984540054
    rate = rejected/count; denominator = 1+z*z/count
    center = (rate+z*z/(2*count))/denominator
    half = z*np.sqrt(rate*(1-rate)/count+z*z/(4*count**2))/denominator
    return float(center-half), float(center+half)


def summarize(output, requested, design, source, state="running", failures=None):
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((output/"cases").glob("*.json"))]
    records = [r for r in records if r["entry"]["replication"] in requested]
    tests, levels, comparisons = [], [], []
    for r in records:
        if r["source_sha256"] != source:
            raise RuntimeError("mixed experiment sources")
        common = {k:r["entry"][k] for k in ("replication","dgp","case","N","dataset")}
        tests.extend({**common, **t} for t in r["tests"])
        levels.extend({**common, **t} for t in r["oos_levels"])
        comparisons.extend({**common, **t} for t in r["oos_comparisons"])
    tf = pd.DataFrame(tests)
    expected = len(requested)*len(design.dgps)*len(CASES)*len(design.sample_sizes)
    rows = []
    for dgp in design.dgps:
        for case in CASES:
            for n in design.sample_sizes:
                for model, blocks in BLOCKS.items():
                    for block in (*[b[0] for b in blocks], "joint"):
                        sub = tf.loc[(tf.dgp==dgp)&(tf.case==case)&(tf.N==n)&(tf.model==model)&(tf.block==block)] if not tf.empty else pd.DataFrame()
                        row = {"dgp":dgp,"case":case,"N":n,"model":model,"block":block,
                               "requested":len(requested),"fits_succeeded":len(sub),
                               "interpretation":"shape-null diagnostic" if dgp=="m2_shape_null" and model!="M2" else "rejection frequency"}
                        for name, column, validity in (("LR","p","lr_valid"),("Wald","wald_p","wald_valid")):
                            valid = sub.loc[sub[validity].astype(bool)&np.isfinite(pd.to_numeric(sub[column],errors="coerce"))] if not sub.empty else pd.DataFrame()
                            count = len(valid)
                            rejected = int((valid[column]<.05).sum()) if count else 0
                            low, high = wilson(rejected,count)
                            row.update({name+"_valid":count, name+"_invalid":len(sub)-count,
                                name+"_rejected":rejected, name+"_rate":rejected/count if count else None,
                                name+"_wilson_low":low, name+"_wilson_high":high})
                        rows.append(row)
    tf.to_csv(output/"tests.csv",index=False)
    pd.DataFrame(rows).to_csv(output/"rejection_summary.csv",index=False)
    pd.DataFrame(levels).to_csv(output/"oos_levels.csv",index=False)
    pd.DataFrame(comparisons).to_csv(output/"oos_comparisons.csv",index=False)
    failed = [{"dataset":r["entry"]["dataset"],"error":r.get("error")} for r in records if not r["success"]]
    audit = {"state":state,"requested_replications":len(requested),"requested_cases":expected,
        "completed_cases":len(records),"successful_cases":len(records)-len(failed),
        "fit_failures":failed,"worker_failures":failures or {},"tests":len(tests),
        "oos_failures":[{"dataset":r["entry"]["dataset"],"error":r.get("oos_error",r.get("error"))}
                        for r in records if not r.get("oos_success",r["success"])],
        "invalid_wald_tests":int((~tf.wald_valid.astype(bool)).sum()) if not tf.empty else 0,
        "source_sha256":source,"finished":len(records)==expected,
        "all_fits_succeeded":len(records)==expected and not failed,
        "upstream_uncertainty":"reestimated across replications; held fixed in each conditional test"}
    dump(output/"run_status.json",audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications",type=int,default=100)
    parser.add_argument("--start",type=int,default=0)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--training",type=int,default=1886)
    parser.add_argument("--oos",type=int,default=642)
    parser.add_argument("--burn",type=int,default=500)
    parser.add_argument("--sample-sizes",type=int,nargs="+",default=[1,20,50,100],choices=[1,20,50,100])
    parser.add_argument("--dgps",nargs="+",default=list(sim.DGP_NAMES),choices=sim.DGP_NAMES)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    args = parser.parse_args()
    if args.training<=22 or min(args.oos,args.replications,args.workers)<1 or min(args.burn,args.start)<0:
        parser.error("training>22; positive OOS, replications, workers; nonnegative burn and start")
    output = args.output.resolve()
    if output.parent != ROOT or not output.name.startswith("output_full_stage_copula_lr_wald"):
        parser.error("use a separate output_full_stage_copula_lr_wald* directory here")
    output.mkdir(parents=True,exist_ok=True)
    (output/"cases").mkdir(exist_ok=True)
    design = Design(training=args.training,oos=args.oos,burn=args.burn,
        sample_sizes=tuple(sorted(set(args.sample_sizes))),dgps=tuple(dict.fromkeys(args.dgps)))
    source = fingerprint()
    manifest = json.loads(json.dumps({"design":asdict(design),"source_sha256":source,
        "empirical_estimator_source":empirical.source_hash(),
        "generation":"legacy 100-constituent score-driven master; independent normal PITs or Gaussian common-factor PITs",
        "marginal":"same conditional DCD-SM family; paths driven by each master realization, not oracle inputs to estimation",
        "common_share":"30% of transformed return variance, calibrated as in original copula experiment",
        "units":"constant rescaling from unit returns to original empirical M1 scale; no coefficient/dependence rescaling",
        "estimator":"unchanged empirical M1--M5; individual observation-sum likelihood; fitted upstream frozen",
        "inference":"same unrestricted fit: restricted conditional LR, and conditional sandwich-HAC(22) Wald; 3- and 6-coefficient blocks",
        "subpanel_caveat":"N<100 observes only a subset of the master history driving true states; nominal iid size is not guaranteed for the observed subpanel filter",
        "initialization":"500-day generator burn; estimator starts from M1 and zero presample scores",
        "OOS":"all nine empirical score metrics, training coefficients fixed; 642 future days"}))
    path = output/"design.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8"))!=manifest:
        parser.error("source/design differs; use a new output directory")
    dump(path,manifest)
    import scipy, numba
    dump(output/"environment.json",{"python":platform.python_version(),"numpy":np.__version__,
        "scipy":scipy.__version__,"pandas":pd.__version__,"numba":numba.__version__})
    requested = list(range(args.start,args.start+args.replications))
    pending = [r for r in requested if not all((output/"cases"/f"rep{r:04d}_{dgp}_{case}_N{n}.json").exists()
        for dgp in design.dgps for case in CASES for n in design.sample_sizes)]
    failures = {}
    summarize(output,requested,design,source)
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(one_replication,r,design,output,source):r for r in pending}
            while futures:
                done,_ = wait(futures,timeout=20,return_when=FIRST_COMPLETED)
                for future in done:
                    replication = futures.pop(future)
                    try:
                        future.result()
                    except Exception as error:
                        failures[str(replication)] = {"error":repr(error),"traceback":traceback.format_exc()}
                        print("WORKER FAILED",replication,repr(error),flush=True)
                progress = summarize(output,requested,design,source,failures=failures)
                print("PROGRESS",progress["completed_cases"],"/",progress["requested_cases"],
                      "fit failures",len(progress["fit_failures"]),"invalid Wald",progress["invalid_wald_tests"],flush=True)
    except BaseException:
        summarize(output,requested,design,source,state="interrupted",failures=failures)
        raise
    progress = summarize(output,requested,design,source,state="complete",failures=failures)
    if not progress["all_fits_succeeded"] or failures or progress["oos_failures"]:
        summarize(output,requested,design,source,state="completed_with_failures",failures=failures)
    print(json.dumps(progress,indent=2),flush=True)


if __name__=="__main__":
    main()
