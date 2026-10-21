"""Refit Case 1 using observation-sum, frozen-upstream conditional LR tests.

Original and factor-removed cross sections plus N=1 index applications.
Restricted block fits reestimate the other three coefficients in the SAME
layer; fitted upstream paths and M1 Fisher geometry are always held fixed.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import traceback
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import erfcx, logsumexp
import case1_lr_kernels as kernels
import prepare_case1_conditional_lr_inputs as inputs
import run_case1_final_common_calendar_35_series as old
import run_sp500 as core
import run_direct_dynamic_skewt_shape_test as shape
import run_static_q0_symmetric_t_four_models as scoring
import run_static_sm_standard_even_odd_model5 as endpoint
from estimate_score_har import fixed_denominator_score_averages

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUTPUT = inputs.OUTPUT
SCHEMA = "conditional-lr-observation-sum-v1"


def source_hash():
    h = hashlib.sha256()
    for name in ("run_case1_conditional_lr.py","case1_lr_kernels.py"):
        h.update((ROOT/name).read_bytes())
    return h.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    temporary.replace(path)


def log_chi2_survival(statistic,df):
    """Stable log-p for the only two block dimensions (3 and 6)."""
    x = statistic/2
    if x == 0:
        return 0.0
    if df == 3:
        return -x+math.log(float(erfcx(math.sqrt(x)))+2*math.sqrt(x/math.pi))
    if df == 6:
        return -x+float(logsumexp([0.0,math.log(x),2*math.log(x)-math.log(2)]))
    raise ValueError("expected three- or six-coefficient null")


def maximize(objective,starts,bounds,label):
    fits, diagnostics = [], []
    for start in starts:
        start = np.asarray(start,dtype=float)
        if start.size != len(bounds):
            raise ValueError("optimizer start has wrong dimension")
        start = np.array([np.clip(x,a,b) for x,(a,b) in zip(start,bounds)])
        fit = minimize(objective,start,jac=True,method="L-BFGS-B",bounds=bounds,
            options={"maxiter":2000,"ftol":1e-14,"gtol":2e-8,"maxls":70,"maxcor":15})
        grad = np.asarray(fit.jac,dtype=float)
        projected = grad.copy()
        for i,(a,b) in enumerate(bounds):
            if fit.x[i]<=a+1e-7 and grad[i]>0 or fit.x[i]>=b-1e-7 and grad[i]<0:
                projected[i]=0
        residual = float(np.max(np.abs(projected)))
        valid = bool(fit.success and np.isfinite(fit.fun) and fit.fun<1e50 and residual<=2e-6)
        diagnostics.append({"success":bool(fit.success),"accepted":valid,"message":str(fit.message),
            "iterations":int(fit.nit),"evaluations":int(fit.nfev),"mean_nll":float(fit.fun),
            "projected_gradient_max":residual})
        if valid:
            fits.append((fit,residual))
    if not fits:
        raise RuntimeError(f"{label}: no converged optimizer start: {diagnostics}")
    optimum,residual = min(fits,key=lambda z:float(z[0].fun))
    if np.any(np.abs(optimum.x)>499.9) and all(a==-500 for a,b in bounds):
        raise RuntimeError(f"{label}: coefficient bound reached")
    return {"theta":np.asarray(optimum.x).tolist(),"mean_nll":float(optimum.fun),
        "projected_gradient_max":residual,"successful_starts":len(fits),
        "total_starts":len(starts),"largest_successful_objective_gap":float(max(f.fun for f,r in fits)-optimum.fun),
        "starts":diagnostics}


def fit_static(values,warm):
    sigma = float(np.std(values))
    sigma = max(sigma,1e-6)
    unit = np.array([sigma,1,1,1],dtype=float)
    def objective(theta):
        state = theta*unit
        try:
            ll,score = kernels.reference_day(values,0,len(values),state)
            value,grad = -ll,-score*unit
            if np.isfinite(value) and np.all(np.isfinite(grad)):
                return value,grad
        except (FloatingPointError,ValueError,OverflowError):
            pass
        return 1e100,np.zeros(4)
    starts = [np.r_[values.mean(),np.log(sigma*sigma),core.parameters_to_state(eta,skew)]/unit
        for eta in (3.5,5,8,15) for skew in (-0.12,0,0.12)]
    if warm:
        starts.insert(0,np.asarray(warm["model1"]["state"])/unit)
    fit = maximize(objective,starts,[(-0.1/sigma,0.1/sigma),(-20,0),(-12,12),(-12,12)],"M1")
    anchor = np.asarray(fit["theta"])*unit
    eta,skew,_,_ = core.state_to_parameters(anchor[2:])
    return {**fit,"state":anchor.tolist(),"mean":float(anchor[0]),"scale":float(np.exp(anchor[1]/2)),
            "eta":float(eta),"lambda":float(skew),"log_likelihood":float(-fit["mean_nll"]*len(values)),
            "shape_anchor_bound_reached":bool(np.max(np.abs(anchor[2:]))>=11.999)}


def fit_layer(run,units,counts,block_names,warm,label):
    """Three- and six-coefficient LR tests conditional on an identical upstream."""
    nobs = int(counts.sum())
    def objective(theta):
        try:
            filtered = run(np.asarray(theta,dtype=float),True)
            value = -float(counts @ filtered[3])/nobs
            grad = -np.asarray(counts @ filtered[4])/nobs
            if np.isfinite(value) and np.all(np.isfinite(grad)):
                return value,grad
        except (FloatingPointError,ValueError,OverflowError):
            pass
        return 1e100,np.zeros(6)
    rng = np.random.default_rng(20261005)
    starts = [np.zeros(6),np.ones(6),-np.ones(6),rng.normal(size=6),3*rng.normal(size=6)]
    if warm is not None:
        starts.insert(0,(np.asarray(warm)/units).ravel())
    unrestricted = maximize(objective,starts,[(-500,500)]*6,label)
    null = run(np.zeros(6),False)
    restricted = {}
    # Reference layers flatten horizon then state; endpoint layers flatten
    # direction then horizon. Pass the corresponding explicit block indices.
    blocks = block_names
    for name,zero_indices in blocks:
        zero_indices = np.asarray(zero_indices,dtype=int)
        free = np.setdiff1d(np.arange(6),zero_indices)
        def wrap(z):
            theta = np.zeros(6)
            theta[free]=z
            value,gradient = objective(theta)
            return value,gradient[free]
        initial = np.asarray(unrestricted["theta"])[free]
        fit = maximize(wrap,[initial,np.zeros(3),np.ones(3),-np.ones(3),rng.normal(size=3)],
                       [(-500,500)]*3,f"{label} {name}=0")
        theta = np.zeros(6); theta[free]=fit["theta"]
        restricted[name] = {**fit,"full_theta":theta.tolist(),"zero_indices":zero_indices.tolist()}
    # Explicitly include each restricted solution in the unrestricted search.
    # A negative LR is an optimization failure, never evidence for the null.
    candidates = [np.asarray(v["full_theta"]) for v in restricted.values()]
    best_restricted = min(v["mean_nll"] for v in restricted.values())
    if unrestricted["mean_nll"] > best_restricted+1e-13:
        unrestricted = maximize(objective,[np.asarray(unrestricted["theta"]),*candidates],
                                 [(-500,500)]*6,label+" nesting retry")
    theta = np.asarray(unrestricted["theta"])
    fitted = run(theta,False)
    ll = float(counts @ fitted[3])
    null_ll = float(counts @ null[3])
    tests = []
    for name,rest in list(restricted.items())+[("joint",None)]:
        reduced = run(np.asarray(rest["full_theta"]),False) if rest is not None else null
        ll0 = float(counts @ reduced[3])
        raw_statistic = 2*(ll-ll0)
        if raw_statistic < -1e-6:
            raise RuntimeError(f"{label}/{name}: negative LR {raw_statistic}; fit nesting failed")
        statistic = max(0,raw_statistic)
        df = 6 if name=="joint" else 3
        logp = log_chi2_survival(statistic,df)
        tests.append({"block":name,"lr":statistic,"df":df,"p":float(np.exp(logp)),
            "log_p":logp,"minus_log10_p":-logp/math.log(10),
            "unrestricted_layer_log_likelihood":ll,"restricted_layer_log_likelihood":ll0,
            "raw_lr":raw_statistic,"conditioning":"fixed fitted upstream and M1 Fisher geometry"})
    return {"coefficients":(theta.reshape(units.shape)*units).tolist(),"parameter_units":units.tolist(),
        "unrestricted":unrestricted,"restricted":restricted,"layer_log_likelihood":ll,
        "zero_layer_log_likelihood":null_ll,"lr_tests":tests},fitted


def as_reference(filtered,begin=0):
    states,scores,drivers,contributions,*_ = filtered
    states = states[begin:]
    eta_skew = np.array([core.state_to_parameters(s[2:])[:2] for s in states])
    return core.FourStateFilter(states=states,means=states[:,0],log_variances=states[:,1],
        variances=np.exp(states[:,1]),eta=eta_skew[:,0],skew=eta_skew[:,1],scores=scores[begin:],
        contributions=contributions[begin:],terminal_state=states[-1].copy(),terminal_score_history=scores[-22:].copy())


def recursion_error(states,drivers,anchor,coefficients,endpoint_layer=False):
    errors = []
    for t in range(len(states)):
        expected = np.asarray(anchor).copy()
        for h,w in enumerate((1,5,22)):
            avg = drivers[max(0,t-w):t].sum(axis=0)/w
            expected += (coefficients[:,h] if endpoint_layer else coefficients[h])*avg
        errors.append(float(np.max(np.abs(states[t]-expected))))
    return max(errors,default=0)


def run_dataset(entry, *, daily_mean=False):
    key = entry["dataset"]
    out = OUTPUT/key
    out.mkdir(parents=True,exist_ok=True)
    fingerprint = source_hash()
    checkpoint_path = out/"fit_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {}
    if checkpoint and (checkpoint.get("source_hash")!=fingerprint or checkpoint.get("input_sha256")!=entry["input_sha256"]):
        raise RuntimeError(f"{key}: checkpoint source or data changed")
    if (out/"estimation.json").exists():
        result = json.loads((out/"estimation.json").read_text(encoding="utf-8"))
        if result.get("source_hash")==fingerprint and result.get("input_sha256")==entry["input_sha256"]:
            return result
        raise RuntimeError(f"{key}: completed output source or data changed")
    checkpoint.update(schema=SCHEMA,source_hash=fingerprint,input_sha256=entry["input_sha256"])
    raw = np.load(REPO/entry["input"])
    values,offsets = np.asarray(raw["values"]),np.asarray(raw["offsets"])
    dates = pd.DatetimeIndex(raw["dates"])
    n_train = int((dates<=old.TRAINING_END).sum())
    tr = int(offsets[n_train]); tv,to = values[:tr],offsets[:n_train+1]
    actual_counts = np.diff(to)
    # Kernels already return cross-sectional means. Unit date weights give
    # sum_t (1/N_t) sum_i log f_t; archived callers retain observation sums.
    counts = np.ones(n_train) if daily_mean else actual_counts
    warm_path = old.OUTPUT/key/"estimation.json"
    warm = json.loads(warm_path.read_text(encoding="utf-8")) if warm_path.exists() else {}
    if "M1" not in checkpoint:
        print(key,"M1",flush=True)
        checkpoint["M1"] = fit_static(tv,warm)
        dump(checkpoint_path,checkpoint)
    anchor = np.asarray(checkpoint["M1"]["state"])
    blank = np.zeros((n_train,4))
    zunit = np.ones((3,2))
    static = kernels.reference_filter(tv,to,anchor,np.zeros((3,2)),zunit,blank,np.zeros((2,2)),False,False)
    avg = fixed_denominator_score_averages(static[2],(1,5,22))
    units2 = np.array([[target/max(float(np.median(np.abs(a[:,j]))),1e-12)
        for j,target in enumerate((1e-4,1e-2))] for a in avg])
    run2 = lambda theta,g: kernels.reference_filter(tv,to,anchor,theta.reshape(3,2)*units2,units2,blank,np.zeros((2,2)),False,g)
    if "M2" not in checkpoint:
        print(key,"M2 conditional LR",flush=True)
        warm2 = np.asarray(warm["model2_betas"])[:,:2] if warm else None
        checkpoint["M2"],_ = fit_layer(run2,units2,counts,[("location",[0,2,4]),("scale",[1,3,5])],warm2,"M2")
        dump(checkpoint_path,checkpoint)
    c2 = np.asarray(checkpoint["M2"]["coefficients"])
    f2 = run2((c2/units2).ravel(),False)
    full2 = kernels.reference_filter(values,offsets,anchor,c2,units2,np.zeros((len(dates),4)),np.zeros((2,2)),False,False)
    g3 = shape.shape_geometry(anchor)
    nested3 = kernels.reference_filter(tv,to,anchor,np.zeros((3,2)),zunit,f2[0],g3.projection_shape_on_ms,True,False)
    avg3 = fixed_denominator_score_averages(nested3[2],(1,5,22))
    units3 = np.array([[1e-2/max(float(np.median(np.abs(a[:,j]))),1e-12) for j in range(2)] for a in avg3])
    run3 = lambda theta,g: kernels.reference_filter(tv,to,anchor,theta.reshape(3,2)*units3,units3,f2[0],g3.projection_shape_on_ms,True,g)
    if "M3" not in checkpoint:
        print(key,"M3 conditional LR",flush=True)
        warm3 = np.asarray(warm["model3_betas"]) if warm else None
        checkpoint["M3"],_ = fit_layer(run3,units3,counts,[("eta",[0,2,4]),("lambda",[1,3,5])],warm3,"M3")
        dump(checkpoint_path,checkpoint)
    c3 = np.asarray(checkpoint["M3"]["coefficients"])
    f3 = run3((c3/units3).ravel(),False)
    full3 = kernels.reference_filter(values,offsets,anchor,c3,units3,full2[0],g3.projection_shape_on_ms,True,False)
    knots = core.make_clamped_knots(old.base.BREAKPOINTS,3)
    masses = core.bspline_masses(knots,3)
    helmert = core.helmert_submatrix(len(masses))
    directions = np.vstack(endpoint.endpoint_directions(helmert))
    geom = scoring.corrected_tangent_geometry(anchor,masses,knots,masses,helmert,(0,1),"fixed M1 q score off m/log variance")
    loading,projection = directions@helmert.T,directions@geom.projection_q_on_hansen
    endpoint_full,endpoint_train = {},{}
    for model,reference in (("M4",full2),("M5",full3)):
        u = core.dynamic_reference_pit(values,offsets,reference[0])
        phi = core.normalized_bspline_basis(u,knots,3,masses)
        phi_train = np.ascontiguousarray(phi[:tr])
        reference_scores = np.ascontiguousarray(reference[1][:n_train])
        null = kernels.endpoint_filter(phi_train,to,reference_scores,masses,loading,projection,np.zeros((2,3)),np.ones((2,3)),False)
        av = fixed_denominator_score_averages(null[1],(1,5,22))
        units = np.array([[1e-2/max(float(np.median(np.abs(av[h][:,b])))*float(np.linalg.norm(directions[b])),1e-12)
            for h in range(3)] for b in range(2)])
        run = lambda theta,g: kernels.endpoint_filter(phi_train,to,reference_scores,masses,loading,projection,theta.reshape(2,3)*units,units,g)
        if model not in checkpoint:
            print(key,model,"conditional LR",flush=True)
            warmq = np.asarray(warm[f"model{model[-1]}"]["coefficients"]) if warm else None
            checkpoint[model],_ = fit_layer(run,units,counts,[("even",[0,1,2]),("odd",[3,4,5])],warmq,model)
            dump(checkpoint_path,checkpoint)
        coeff = np.asarray(checkpoint[model]["coefficients"])
        endpoint_train[model] = run((coeff/units).ravel(),False)
        endpoint_full[model] = kernels.endpoint_filter(phi,offsets,reference[1],masses,loading,projection,coeff,units,False)
        del phi,phi_train
    audit = {
        "M2_recursion_error":recursion_error(f2[0][:,:2],f2[2],anchor[:2],c2),
        "M3_recursion_error":recursion_error(f3[0][:,2:],f3[2],anchor[2:],c3),
        "M2_training_full_path_error":float(np.max(np.abs(f2[0]-full2[0][:n_train]))),
        "M3_training_full_path_error":float(np.max(np.abs(f3[0]-full3[0][:n_train]))),
        "M3_joint_null_parent_error":float(np.max(np.abs(nested3[3]-f2[3]))),
        "M1_likelihood_reproduction_error":float(abs(counts@static[3]-checkpoint["M1"]["log_likelihood"])),
        "retained_score_history":22}
    for model in ("M4","M5"):
        f = endpoint_train[model]
        coeff = np.asarray(checkpoint[model]["coefficients"])
        audit[f"{model}_recursion_error"] = recursion_error(f[0],f[1],np.zeros(2),coeff,True)
        audit[f"{model}_training_full_path_error"] = float(np.max(np.abs(f[0]-endpoint_full[model][0][:n_train])))
        audit[f"{model}_minimum_full_weight"] = float(endpoint_full[model][2].min())
    if max(v for name,v in audit.items() if name.endswith("recursion_error") or name.endswith("path_error") or name.endswith("parent_error"))>1e-9:
        raise RuntimeError(f"{key}: filter audit failed: {audit}")
    print(key,"OOS scores",flush=True)
    oos_values,oos_offsets = values[tr:],offsets[n_train:]-tr
    static_full = kernels.reference_filter(values,offsets,anchor,np.zeros((3,2)),zunit,np.zeros((len(dates),4)),np.zeros((2,2)),False,False)
    static_correction = SimpleNamespace(weights=np.tile(masses,(len(dates)-n_train,1)),contributions=np.zeros(len(dates)-n_train))
    pair1 = scoring.score_oos_pair(oos_values,oos_offsets,dates[n_train:],tv,as_reference(static_full,n_train),static_correction,
                                  knots,masses,masses,old.identity.M1,"_discard")
    score_parts = [pair1.loc[pair1.model==old.identity.M1]]
    for parent,model,ref in (("M2","M4",full2),("M3","M5",full3)):
        correction = SimpleNamespace(weights=endpoint_full[model][2][n_train:],contributions=endpoint_full[model][3][n_train:])
        score_parts.append(scoring.score_oos_pair(oos_values,oos_offsets,dates[n_train:],tv,as_reference(ref,n_train),correction,
            knots,masses,masses,getattr(old.identity,parent),getattr(old.identity,model)))
    scores = pd.concat(score_parts,ignore_index=True)
    if not np.isfinite(scores[list(scoring.SCORE_NAMES)].to_numpy()).all():
        raise RuntimeError(f"{key}: nonfinite OOS scores")
    levels = scores.groupby("model",as_index=False)[list(scoring.SCORE_NAMES)].mean()
    comparisons = old.identity.compare(scores)
    likelihoods = {"M1":checkpoint["M1"]["log_likelihood"],"M2":float(counts@f2[3]),"M3":float(counts@f3[3]),
                  "M4":float(counts@f2[3])+checkpoint["M4"]["layer_log_likelihood"],
                  "M5":float(counts@f3[3])+checkpoint["M5"]["layer_log_likelihood"]}
    result = {"schema":SCHEMA,"source_hash":fingerprint,"input_sha256":entry["input_sha256"],
        "dataset":entry,"specification":{"inference":"conditional LR with fitted upstream paths fixed",
        "likelihood":("sum of daily cross-sectional mean log densities; equal date weights"
                      if daily_mean else "sum of individual log densities, including actual per-date N_t"),
        "drivers":"date-average scores; exact HAR(1,5,22), zero presample history",
        "static_sm":"h=1, q0=0", "breakpoints":old.base.BREAKPOINTS.tolist(),
        "projection":"fixed M1 Fisher geometry; remove m/log variance only; no whitening",
        "three_coefficient_null":"other three coefficients reestimated in the tested layer",
        "chi_square_calibration":"conditional common marginal density and cross-sectional independence, with regularity; fixed-upstream inference"},
        "models":{model:checkpoint[model] for model in ("M1","M2","M3","M4","M5")},
        "training_log_likelihoods":likelihoods,"audit":audit,
        "oos_summary":levels.to_dict(orient="records"),"oos_comparisons":comparisons.to_dict(orient="records")}
    scores.to_csv(out/"daily_oos_scores.csv",index=False)
    levels.to_csv(out/"oos_levels.csv",index=False)
    comparisons.to_csv(out/"oos_comparisons.csv",index=False)
    pd.DataFrame([{ "model":model,**test} for model in ("M2","M3","M4","M5") for test in checkpoint[model]["lr_tests"]]).to_csv(out/"lr_tests.csv",index=False)
    dump(out/"estimation.json",result)
    print(key,"COMPLETE",flush=True)
    return result


def summarize(manifest):
    results,missing = [],[]
    for entry in manifest["entries"]:
        path = OUTPUT/entry["dataset"]/"estimation.json"
        if not path.exists():
            missing.append(entry["dataset"])
            continue
        results.append(json.loads(path.read_text(encoding="utf-8")))
    if missing:
        dump(OUTPUT/"completion_audit.json",{"status":"incomplete","completed":len(results),"expected":75,"missing":missing})
        return False
    tests,levels,comparisons,parameters,audits,likelihoods = [],[],[],[],[],[]
    for result in results:
        e = result["dataset"]
        common = {k:e[k] for k in ("dataset","label","family","target","parent_dataset")}
        audits.append({**common,**result["audit"]})
        likelihoods.append({**common,**result["training_log_likelihoods"]})
        for model in ("M2","M3","M4","M5"):
            fit = result["models"][model]
            tests.extend({**common,"model":model,**test} for test in fit["lr_tests"])
            coeff = np.asarray(fit["coefficients"])
            for j,block in enumerate(("location","scale") if model=="M2" else ("eta","lambda") if model=="M3" else ("even","odd")):
                for h,w in enumerate((1,5,22)):
                    parameters.append({**common,"model":model,"block":block,"window":w,
                        "estimate":float(coeff[h,j] if model in ("M2","M3") else coeff[j,h])})
        levels.extend({**common,**row} for row in result["oos_summary"])
        comparisons.extend({**common,**row} for row in result["oos_comparisons"])
    for name,rows in (("case1_lr_tests",tests),("case1_oos_levels",levels),("case1_oos_comparisons",comparisons),
                      ("case1_parameters",parameters),("case1_har22_audit",audits),("case1_training_log_likelihoods",likelihoods)):
        pd.DataFrame(rows).to_csv(OUTPUT/f"{name}.csv",index=False)
    test_frame = pd.DataFrame(tests)
    best = []
    level_frame = pd.DataFrame(levels)
    for (target,family),part in level_frame.groupby(["target","family"]):
        for metric in scoring.SCORE_NAMES:
            ascending = metric not in old.HIGHER_IS_BETTER
            selected = part.sort_values(metric,ascending=ascending).groupby("dataset").head(1)
            for model,count in selected.groupby("model").size().items():
                best.append({"target":target,"family":family,"score":metric,"model":model,"count":int(count)})
    pd.DataFrame(best).to_csv(OUTPUT/"case1_oos_best_model_counts.csv",index=False)
    audit = {"status":"complete","completed":75,"expected":75,"lr_tests":len(tests),"oos_levels":len(levels),
        "oos_comparisons":len(comparisons),"parameters":len(parameters),"invalid_lr_tests":int((~np.isfinite(test_frame.lr)).sum()),
        "negative_raw_lr_min":float(test_frame.raw_lr.min()),"source_hash":source_hash(),
        "conditioning":"fitted upstream paths held fixed; first-stage uncertainty not propagated"}
    if any(r["source_hash"]!=source_hash() for r in results):
        raise RuntimeError("mixed source fingerprints")
    dump(OUTPUT/"completion_audit.json",audit)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets",default="all")
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--summarize-only",action="store_true")
    args = parser.parse_args()
    manifest = json.loads((OUTPUT/"input_manifest.json").read_text(encoding="utf-8"))
    if not args.summarize_only:
        selected = manifest["entries"] if args.datasets=="all" else [e for e in manifest["entries"] if e["dataset"] in args.datasets.split(",")]
        if args.datasets!="all" and len(selected)!=len(args.datasets.split(",")):
            raise ValueError("unknown or duplicate dataset selection")
        failures = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_dataset,entry):entry["dataset"] for entry in selected}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    future.result()
                except Exception:
                    failure = {"dataset":key,"traceback":traceback.format_exc()}
                    failures.append(failure)
                    print("FAILED",key,failure["traceback"],flush=True)
                    dump(OUTPUT/"failed_datasets.json",failures)
        if failures:
            raise RuntimeError(f"failed datasets: {[f['dataset'] for f in failures]}")
    summarize(manifest)


if __name__ == "__main__":
    main()
