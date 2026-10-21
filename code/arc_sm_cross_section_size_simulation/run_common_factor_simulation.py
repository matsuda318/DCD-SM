"""Identical DCD-SM marginals with a 30% common return-variance component.

No score/coefficient rescaling beyond the legacy calibration is performed.
The Gaussian copula correlation is calibrated separately at each current
marginal, not mistaken for the return correlation. All N are nested subsets
of one 100-constituent market, including the fixed first constituent for N=1.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import logsumexp, ndtr, roots_hermitenorm

import run_simulation as sim
from run_heterogeneous_simulation import sandwich_wald, summarize

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output_common_factor_30pct_optimizer_fixed"
HERMITE_DEGREE = 40
QUADRATURE_ORDERS = (96, 192, 384, 768, 1536, 3072)
SHARE_TOLERANCE = 3e-4


@dataclass(frozen=True)
class Design:
    training: int = 1886
    oos: int = 642
    burn: int = 500
    common_share: float = .30
    seed: int = 20261005
    sample_sizes: tuple[int, ...] = (1, 20, 50, 100)
    dgps: tuple[str, ...] = sim.DGP_NAMES


@lru_cache(maxsize=None)
def normal_quadrature(order):
    nodes, weights = roots_hermitenorm(order)
    weights = weights / np.sqrt(2 * np.pi)
    basis = np.empty((HERMITE_DEGREE + 1, order))
    basis[0], basis[1] = 1, nodes
    for k in range(1, HERMITE_DEGREE):
        basis[k + 1] = (nodes * basis[k] - np.sqrt(k) * basis[k - 1]) / np.sqrt(k + 1)
    return nodes, weights, basis


def hermite_moments(quantile, order):
    """For h(Z), Corr(h(Z1),h(Z2)) = sum_{k>=1} a_k^2 rho^k / Var(h)."""
    nodes, weights, basis = normal_quadrature(order)
    values = quantile(ndtr(nodes))
    mean = float(weights @ values)
    centered = values - mean
    variance = float(weights @ (centered * centered))
    if not np.isfinite(variance) or variance <= 0:
        raise FloatingPointError("invalid marginal variance in common-factor calibration")
    coefficients = basis[1:] @ (weights * centered)
    return coefficients**2 / variance, mean, variance


def correlation_from_moments(rho, energy):
    return float(energy @ np.power(rho, np.arange(1, len(energy) + 1)))


def calibrate_common_share(quantile, target=.30):
    """Solve with one quadrature and check the share at twice its order.

    Hermite truncation at degree 40 is negligible at the recorded rho values;
    a conservative rho**41 tail bound is enforced as an additional check.
    Numerical integration failures are explicit, never replaced by rho=.30.
    """
    previous = None
    for order in QUADRATURE_ORDERS:
        energy, mean, variance = hermite_moments(quantile, order)
        if previous is not None:
            rho = previous
            checked = correlation_from_moments(rho, energy)
            if abs(checked - target) <= SHARE_TOLERANCE and rho**(HERMITE_DEGREE + 1) < 1e-7:
                return rho, {
                    "latent_rho": rho, "checked_common_share": checked,
                    "share_error": checked - target, "quadrature_order": order,
                    "marginal_mean_standardized": mean,
                    "marginal_variance_standardized": variance,
                }
        if correlation_from_moments(.95, energy) <= target:
            previous = None
            continue
        previous = brentq(lambda rho: correlation_from_moments(rho, energy) - target,
                          0.0, .95, xtol=1e-11)
    raise FloatingPointError("common-share quadrature did not converge within tolerance")


class CommonFactorSampler:
    def __init__(self, grid, component_cdf, target=.30):
        self.grid, self.component_cdf, self.target = grid, component_cdf, target
        self.audit = []
        self._cache_key = None
        self._cache_calibration = None

    def __call__(self, count, q_state, masses, helmert, component_inverse, rng, *, reference_state):
        weights = np.exp(np.log(masses) + helmert @ q_state - logsumexp(np.log(masses) + helmert @ q_state))
        mixture_cdf = np.maximum.accumulate(self.component_cdf @ weights)
        mixture_cdf[0], mixture_cdf[-1] = 0, 1

        def to_reference_u(v):
            return np.clip(np.interp(v, mixture_cdf, self.grid), 1e-12, 1 - 1e-12)

        def quantile(v):
            # Location and scale do not change the common variance fraction.
            return sim.core.hansen_ppf(to_reference_u(v), reference_state[2:])

        key = (reference_state[2:].tobytes(), q_state.tobytes())
        if key == self._cache_key:
            rho, audit = self._cache_calibration
        else:
            rho, audit = calibrate_common_share(quantile, self.target)
            self._cache_key, self._cache_calibration = key, (rho, audit)
        record = dict(audit, generated_day=len(self.audit),
                      log_variance=float(reference_state[1]), eta_state=float(reference_state[2]),
                      q_norm=float(np.linalg.norm(q_state)))
        self.audit.append(record)
        common = rng.standard_normal()
        individual = rng.standard_normal(count)
        latent = np.sqrt(rho) * common + np.sqrt(1 - rho) * individual
        return to_reference_u(ndtr(latent))


def one_replication(replication, design, output):
    sim.TRAINING, sim.OOS, sim.BURN, sim.SEED = design.training, design.oos, design.burn, design.seed
    knots = sim.core.make_clamped_knots(sim.base.BREAKPOINTS, sim.core.DEGREE)
    masses = sim.core.bspline_masses(knots, sim.core.DEGREE)
    grid, _, cdf, _ = sim.prepare_component_sampler(knots, masses)
    sampler = CommonFactorSampler(grid, cdf, design.common_share)
    started = time.monotonic()
    try:
        rows = sim.one_replication(
            replication, sample_state_uniforms=sampler, fit_inference=sandwich_wald,
            sample_sizes=design.sample_sizes, dgps=design.dgps,
        )
    finally:
        audit = pd.DataFrame(sampler.audit)
        if not audit.empty:
            per_dgp = design.training + design.oos + design.burn
            audit["dgp"] = [design.dgps[min(int(i) // per_dgp, len(design.dgps) - 1)] for i in audit.generated_day]
            audit.to_csv(Path(output) / f"sampler_audit_{replication:04d}.csv", index=False)
            previous_path = ROOT / "output_common_factor_30pct" / f"sampler_audit_{replication:04d}.csv"
            if previous_path.exists() and (design.training, design.oos, design.burn, design.seed, design.dgps) == (
                    1886, 642, 500, 20261005, sim.DGP_NAMES):
                previous = pd.read_csv(previous_path).iloc[:len(audit)]
                columns = ["latent_rho", "checked_common_share", "log_variance", "eta_state", "q_norm"]
                maximum_difference = float(np.max(np.abs(audit[columns].to_numpy() - previous[columns].to_numpy())))
                if maximum_difference > 1e-12:
                    raise RuntimeError(f"DGP audit differs from previous run: {maximum_difference}")
                (Path(output) / f"generation_match_{replication:04d}.json").write_text(
                    json.dumps({"replication": replication, "days_checked": len(audit),
                                "maximum_absolute_difference": maximum_difference,
                                "same_seed_and_generation_settings": True}, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        for model in ("m3", "m4", "m5"):
            # iid chi-square LR p values are invalid for dependent constituents.
            row.pop(f"{model}_lr")
            row.pop(f"{model}_lr_p")
            if not row[f"{model}_optimizer_success"]:
                row[f"{model}_wald_valid"] = False
                row[f"{model}_wald_p"] = float("nan")
        row["common_variance_share"] = design.common_share
        row["replication_seconds"] = time.monotonic() - started
    return rows


def manifest(design):
    calibrated = sim.calibration()
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(sim.__file__), ROOT / "run_heterogeneous_simulation.py",
                 ROOT / "scalar_optimizer.py", sim.CALIBRATION):
        digest.update(path.read_bytes())
    return {
        "design": asdict(design), "source_sha256": digest.hexdigest(),
        "master_N": sim.MASTER_N, "identical_constituent_marginals": True,
        "marginal": "same DCD-SM G_t for all i; current eta and endpoint state enter inverse CDF",
        "copula": "Z_i=sqrt(rho_t) F+sqrt(1-rho_t) epsilon_i, independent standard normals",
        "target": "Var(E[R_i|F,past]|past)/Var(R_i|past)=0.30; equivalently pairwise return correlation",
        "idiosyncratic_share": "E[Var(R_i|F,past)|past]/Var(R_i|past)=0.70; conditional independence given F",
        "coefficient_calibration": "unchanged legacy empirical coefficients times sqrt(100/491); no dependence rescaling",
        "coefficients": {k: np.asarray(calibrated[k]).tolist() for k in ("scale_kappa", "eta_kappa", "even_kappa")},
        "nested_panels": "fixed first N columns; no new path when N changes",
        "inference": "daily-score sandwich Wald, 3 coefficients, HAC(22); NOT iid LR",
        "known_in_estimation": "static anchor and true location/scale path; M5 freezes fitted M3",
        "legacy_comparison": "old iid LR rejection rates are not matched-method power comparisons",
        "optimizer": "mean objective L-BFGS-B; invalid trials switch to feasible BFGS with Armijo backtracking; projected gradient <=1e-8 and coordinate-neighbor gain <=1e-10; uncertified fits raise failures",
        "numeric": {"quadrature_orders": QUADRATURE_ORDERS, "hermite_degree": HERMITE_DEGREE,
                    "common_share_absolute_tolerance": SHARE_TOLERANCE,
                    "component_cdf_grid_size": sim.GRID_SIZE, "reference_pit_clip": 1e-12},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--training", type=int, default=1886)
    parser.add_argument("--oos", type=int, default=642)
    parser.add_argument("--burn", type=int, default=500)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[1, 20, 50, 100], choices=[1, 20, 50, 100])
    parser.add_argument("--dgps", nargs="+", default=list(sim.DGP_NAMES), choices=sim.DGP_NAMES)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if min(args.replications, args.workers, args.oos) < 1 or args.training <= 22 or args.burn < 0 or args.start < 0:
        parser.error("positive replication/workers/OOS, training>22, nonnegative burn/start required")
    design = Design(training=args.training, oos=args.oos, burn=args.burn,
                    sample_sizes=tuple(sorted(set(args.sample_sizes))), dgps=tuple(dict.fromkeys(args.dgps)))
    output = args.output.resolve()
    if output in (sim.OUTPUT.resolve(), (ROOT / "output_heterogeneous").resolve()):
        parser.error("existing experiment outputs must not be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(json.dumps(manifest(design)))
    metadata_path = output / "design.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
        parser.error("different design or source fingerprint; use a new output directory")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    requested = list(range(args.start, args.start + args.replications))
    pending = [r for r in requested if not (output / f"replication_{r:04d}.csv").exists()]
    failures = {}

    def write_status(state):
        completed = [r for r in requested if (output / f"replication_{r:04d}.csv").exists()]
        data = {"state": state, "complete": len(completed) == len(requested),
                "requested_replications": requested, "completed_replications": completed,
                "requested_count": len(requested), "completed_count": len(completed), "failures": failures}
        temporary = output / "run_status.tmp"
        temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / "run_status.json")

    def save(replication, rows):
        if len(rows) != len(design.sample_sizes) * len(design.dgps):
            raise RuntimeError("incomplete replication")
        temporary = output / f"replication_{replication:04d}.tmp"
        pd.DataFrame(rows).to_csv(temporary, index=False)
        temporary.replace(output / f"replication_{replication:04d}.csv")
        frame = pd.concat([pd.read_csv(output / f"replication_{r:04d}.csv") for r in requested
                           if (output / f"replication_{r:04d}.csv").exists()], ignore_index=True)
        frame.to_csv(output / "replications.csv", index=False)
        summarize(frame).to_csv(output / "detailed_summary.csv", index=False)
        write_status("running")
        print(f"Saved replication {replication}; {frame.replication.nunique()} complete", flush=True)

    def failure(replication, error):
        failures[replication] = repr(error)
        data = {"replication": replication, "error": repr(error), "traceback": traceback.format_exc()}
        (output / f"failure_{replication:04d}.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        write_status("running")
        print(f"FAILED replication {replication}: {error!r}", flush=True)

    print(json.dumps({"output": str(output), "pending": len(pending), "design": asdict(design)}), flush=True)
    write_status("running")
    try:
        if args.workers == 1:
            for r in pending:
                try:
                    rows = one_replication(r, design, output)
                except Exception as error:
                    failure(r, error)
                else:
                    save(r, rows)
        elif pending:
            with ProcessPoolExecutor(max_workers=min(args.workers, len(pending))) as executor:
                futures = {executor.submit(one_replication, r, design, output): r for r in pending}
                for future in as_completed(futures):
                    try:
                        rows = future.result()
                    except Exception as error:
                        failure(futures[future], error)
                    else:
                        save(futures[future], rows)
    except BaseException:
        write_status("failed")
        raise
    write_status("completed_with_failures" if failures else "complete")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
