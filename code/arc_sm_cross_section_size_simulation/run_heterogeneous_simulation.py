"""Independent heterogeneous constituents with an exact DCD-SM average density.

For V_i = G_t(R_i), the constituent density is
1 + a_i (2v-1) + b_i (6v^2-6v+1).  Fixed, balanced loadings make the
cross-sectional average density exactly g_t for every selected panel size.
The singleton is necessarily neutral.  Inference uses daily-score sandwich
Wald statistics, not the iid likelihood-ratio calibration of the old run.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from functools import partial
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import chi2

import run_simulation as sim
from estimate_score_har import hac_meat


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output_heterogeneous"


@dataclass(frozen=True)
class Design:
    training: int = 1886
    oos: int = 642
    burn: int = 500
    strength: float = 0.8
    seed: int = 20261004
    sample_sizes: tuple[int, ...] = (1, 20, 50, 100)
    dgps: tuple[str, ...] = sim.DGP_NAMES


def constituent_loadings(strength: float = 0.8) -> np.ndarray:
    """Zero means and equal second moments at N=20,50,100; first row zero.

    The 19 nonneutral members of the first 20-panel are slightly inflated
    so the neutral singleton does not dilute its loading second moments.
    This is a deliberately balanced finite population, not random sampling
    of arbitrary securities.
    """
    if not 0 <= strength < np.sqrt(19 / 20):
        raise ValueError("strength must be in [0, sqrt(19/20))")
    loadings = np.zeros((sim.MASTER_N, 2))
    for start, stop, inflation in ((1, 20, np.sqrt(20 / 19)), (20, 50, 1), (50, 100, 1)):
        angles = 2 * np.pi * np.arange(stop - start) / (stop - start)
        loadings[start:stop] = strength * inflation / np.sqrt(2) * np.column_stack(
            (np.cos(angles), np.sin(angles))
        )
    if np.max(np.abs(loadings).sum(axis=1)) >= 1:
        raise ValueError("constituent density positivity is not guaranteed")
    return loadings


def constituent_pit_density(v: np.ndarray, loadings: np.ndarray) -> np.ndarray:
    return 1 + loadings[..., 0] * (2 * v - 1) + loadings[..., 1] * (6 * v * v - 6 * v + 1)


def constituent_pit_cdf(v: np.ndarray, loadings: np.ndarray) -> np.ndarray:
    return v + loadings[..., 0] * (v * v - v) + loadings[..., 1] * (2 * v**3 - 3 * v * v + v)


def constituent_pit_ppf(probability: np.ndarray, loadings: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability, dtype=float)
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("probabilities must lie in [0,1]")
    lower = np.zeros_like(probability)
    upper = np.ones_like(probability)
    for _ in range(48):
        middle = (lower + upper) / 2
        left = constituent_pit_cdf(middle, loadings) < probability
        lower = np.where(left, middle, lower)
        upper = np.where(left, upper, middle)
    return (lower + upper) / 2


def sample_heterogeneous_u(
    count, q_state, masses, helmert, component_inverse, rng, *,
    loadings, grid, component_cdf,
):
    """Independent uniforms are transformed separately for each fixed identity."""
    if count != len(loadings):
        raise ValueError("the master panel must contain all fixed constituents")
    independent = rng.random(count)
    v = constituent_pit_ppf(independent, loadings)
    logits = np.log(masses) + helmert @ q_state
    weights = np.exp(logits - logsumexp(logits))
    mixture_cdf = np.maximum.accumulate(component_cdf @ weights)
    mixture_cdf[0], mixture_cdf[-1] = 0, 1
    # Same component-CDF grid as the legacy experiment; interpolation is a
    # numerical approximation, while the average-density identity is exact.
    return np.clip(np.interp(v, mixture_cdf, grid), 1e-12, 1 - 1e-12)


def sandwich_wald(coefficients, run_filter, bandwidth=22):
    """Conditional fitted-alternative Wald with date-average score clusters.

    Hessian and meat both use sums.  No extra N or T scaling is applied.
    Nonpositive/singular matrices produce an invalid diagnostic, not a
    chi-square statistic based on a silently truncated pseudoinverse.
    """
    coefficients = np.asarray(coefficients, dtype=float)
    units = np.maximum(np.abs(coefficients), 1.0)
    theta = coefficients / units
    fitted = run_filter(coefficients)
    scores = fitted.parameter_scores * units
    if scores.ndim != 2 or scores.shape[1] != 3 or not np.all(np.isfinite(scores)):
        raise ValueError("finite daily parameter scores are required")
    step = 1e-4
    hessian = np.empty((3, 3))
    for j in range(3):
        shift = np.zeros(3)
        shift[j] = step * units[j]
        plus = run_filter(coefficients + shift).gradient * units
        minus = run_filter(coefficients - shift).gradient * units
        hessian[:, j] = -(plus - minus) / (2 * step)
    hessian = (hessian + hessian.T) / 2
    eigenvalues = np.linalg.eigvalsh(hessian)
    result = {
        "wald": float("nan"), "wald_p": float("nan"), "wald_valid": False,
        "hessian_min_eigenvalue": float(eigenvalues[0]),
        "hessian_condition": float(np.linalg.cond(hessian)),
        "scaled_gradient_per_day": float(np.max(np.abs(scores.mean(axis=0)))),
    }
    if eigenvalues[0] <= 1e-10 * max(eigenvalues[-1], 1):
        return result
    inverse = np.linalg.inv(hessian)
    covariance = inverse @ hac_meat(scores, min(bandwidth, len(scores) - 1)) @ inverse
    covariance = (covariance + covariance.T) / 2
    ev = np.linalg.eigvalsh(covariance)
    if ev[0] <= 1e-10 * max(ev[-1], np.finfo(float).tiny):
        return result
    statistic = float(theta @ np.linalg.solve(covariance, theta))
    result.update(wald=statistic, wald_p=float(chi2.sf(statistic, 3)), wald_valid=True)
    return result


def one_replication(replication: int, design: Design):
    # Each worker has its own interpreter.  Legacy defaults and outputs are
    # unchanged when run_simulation.py is used directly.
    sim.TRAINING, sim.OOS, sim.BURN, sim.SEED = design.training, design.oos, design.burn, design.seed
    knots = sim.core.make_clamped_knots(sim.base.BREAKPOINTS, sim.core.DEGREE)
    masses = sim.core.bspline_masses(knots, sim.core.DEGREE)
    grid, _, cdf, _ = sim.prepare_component_sampler(knots, masses)
    sampler = partial(
        sample_heterogeneous_u, loadings=constituent_loadings(design.strength),
        grid=grid, component_cdf=cdf,
    )
    started = time.monotonic()
    rows = sim.one_replication(
        replication, sample_uniforms=sampler, fit_inference=sandwich_wald,
        sample_sizes=design.sample_sizes, dgps=design.dgps,
    )
    for row in rows:
        # LR chi-square p-values from a mixture working likelihood are not
        # valid iid likelihood-ratio inference for these heterogeneous draws.
        for model in ("m3", "m4", "m5"):
            row.pop(f"{model}_lr")
            row.pop(f"{model}_lr_p")
            if not row[f"{model}_optimizer_success"]:
                row[f"{model}_wald_valid"] = False
                row[f"{model}_wald_p"] = float("nan")
        row["heterogeneity_strength"] = design.strength
        row["replication_seconds"] = time.monotonic() - started
    return rows


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dgp, n), block in frame.groupby(["dgp", "N"], sort=True):
        row = {"dgp": dgp, "N": int(n), "replications": len(block)}
        for model in ("m3", "m4", "m5"):
            p = block[f"{model}_wald_p"].dropna()
            rate = float((p < .05).mean()) if len(p) else float("nan")
            row[f"{model}_valid_tests"] = len(p)
            row[f"{model}_invalid_tests"] = len(block) - len(p)
            row[f"{model}_optimizer_failures"] = int((block[f"{model}_optimizer_success"] != 1).sum())
            row[f"{model}_wald_rejection_rate_5pct_valid"] = rate
            row[f"{model}_rejection_mc_se"] = np.sqrt(rate * (1 - rate) / len(p)) if len(p) else float("nan")
        for column in block.columns:
            if "state_correlation" in column or "state_rmse" in column:
                row[f"mean_{column}"] = float(block[column].mean())
                row[f"count_{column}"] = int(block[column].notna().sum())
        for better, baseline in (("M3", "M2"), ("M4", "M2"), ("M5", "M3"), ("M5", "M4")):
            for metric in ("log_score", "q_loss_0.01", "q_loss_0.05"):
                gains = block[f"{better}_{metric}"] - block[f"{baseline}_{metric}"]
                if metric != "log_score":
                    gains = -gains
                prefix = f"{metric}_gain_{better}_vs_{baseline}"
                row[f"mean_{prefix}"] = float(gains.mean())
                row[f"median_{prefix}"] = float(gains.median())
                row[f"positive_fraction_{prefix}"] = float((gains > 0).mean())
                row[f"min_{prefix}"] = float(gains.min())
                row[f"max_{prefix}"] = float(gains.max())
        rows.append(row)
    return pd.DataFrame(rows)


def manifest(design: Design):
    loadings = constituent_loadings(design.strength)
    source_hash = hashlib.sha256()
    for path in (Path(__file__), Path(sim.__file__), ROOT / "scalar_optimizer.py", sim.CALIBRATION):
        source_hash.update(path.read_bytes())
    return {
        "design": asdict(design), "source_sha256": source_hash.hexdigest(),
        "master_N": sim.MASTER_N, "conditional_independence": True,
        "individual_density": "g_i,t(r)=g_t(r)[1+a_i(2G_t(r)-1)+b_i(6G_t(r)^2-6G_t(r)+1)]",
        "inference": "conditional daily-score sandwich Wald, three coefficients, HAC(22)",
        "singleton": "neutral constituent; heterogeneity comparison applies to N>=20",
        "known_in_estimation": "static anchor and true location/scale path",
        "upstream_uncertainty": "M5 holds the fitted M3 path fixed",
        "finite_population": "balanced nested subsets, not arbitrary random subsets",
        "sampler": {"component_cdf_grid_size": sim.GRID_SIZE, "pit_inverse_bisections": 48},
        "constituent_loadings": loadings.tolist(),
        "loading_audits": {
            str(n): {"mean": loadings[:n].mean(axis=0).tolist(),
                     "second_moment": (loadings[:n].T @ loadings[:n] / n).tolist()}
            for n in design.sample_sizes
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--training", type=int, default=1886)
    parser.add_argument("--oos", type=int, default=642)
    parser.add_argument("--burn", type=int, default=500)
    parser.add_argument("--strength", type=float, default=.8)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[1, 20, 50, 100], choices=[1, 20, 50, 100])
    parser.add_argument("--dgps", nargs="+", default=list(sim.DGP_NAMES), choices=sim.DGP_NAMES)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if min(args.replications, args.workers, args.oos) < 1 or args.training <= 22 or args.burn < 0 or args.start < 0:
        parser.error("positive replications/workers/OOS, training>22, and nonnegative burn/start are required")
    design = Design(args.training, args.oos, args.burn, args.strength,
                    sample_sizes=tuple(sorted(set(args.sample_sizes))), dgps=tuple(dict.fromkeys(args.dgps)))
    metadata = manifest(design)
    output = args.output.resolve()
    if output == sim.OUTPUT.resolve():
        parser.error("the legacy iid output directory must not be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "design.json"
    serialized = json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    if metadata_path.exists() and json.loads(metadata_path.read_text(encoding="utf-8")) != json.loads(serialized):
        parser.error("output contains a different design or source version; use a new directory")
    metadata_path.write_text(serialized, encoding="utf-8")
    pd.DataFrame(constituent_loadings(design.strength), columns=["a", "b"]).to_csv(output / "constituent_loadings.csv", index_label="constituent")
    requested = list(range(args.start, args.start + args.replications))
    pending = [r for r in requested if not (output / f"replication_{r:04d}.csv").exists()]

    def write_status(state, error=None):
        completed = [r for r in requested if (output / f"replication_{r:04d}.csv").exists()]
        status = {"state": state, "complete": len(completed) == len(requested),
                  "requested_replications": requested, "completed_replications": completed,
                  "completed_count": len(completed), "requested_count": len(requested)}
        if error is not None:
            status["error"] = str(error)
        temporary = output / "run_status.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / "run_status.json")

    def save(replication, rows):
        if len(rows) != len(design.sample_sizes) * len(design.dgps):
            raise RuntimeError("incomplete replication must not be checkpointed as complete")
        temporary = output / f"replication_{replication:04d}.tmp"
        pd.DataFrame(rows).to_csv(temporary, index=False)
        temporary.replace(output / f"replication_{replication:04d}.csv")
        paths = sorted(output.glob("replication_*.csv"))
        frame = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True).sort_values(["replication", "dgp", "N"])
        frame.to_csv(output / "replications.csv", index=False)
        summarize(frame).to_csv(output / "detailed_summary.csv", index=False)
        write_status("running")
        print(f"Saved replication {replication}; {len(paths)} complete", flush=True)

    print(json.dumps({"output": str(output), "pending": len(pending), "design": asdict(design)}), flush=True)
    write_status("running")
    failures = {}

    def record_failure(replication, error):
        failures[replication] = repr(error)
        (output / f"failure_{replication:04d}.json").write_text(
            json.dumps({"replication": replication, "error": repr(error)}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"FAILED replication {replication}: {error!r}", flush=True)

    try:
        if args.workers == 1:
            for r in pending:
                try:
                    rows = one_replication(r, design)
                except Exception as error:
                    record_failure(r, error)
                else:
                    save(r, rows)
        elif pending:
            with ProcessPoolExecutor(max_workers=min(args.workers, len(pending))) as executor:
                futures = {executor.submit(one_replication, r, design): r for r in pending}
                for future in as_completed(futures):
                    try:
                        rows = future.result()
                    except Exception as error:
                        record_failure(futures[future], error)
                    else:
                        save(futures[future], rows)
    except BaseException as error:
        write_status("failed", error)
        raise
    if failures:
        write_status("completed_with_failures", failures)
        raise SystemExit(1)
    write_status("complete")


if __name__ == "__main__":
    main()
