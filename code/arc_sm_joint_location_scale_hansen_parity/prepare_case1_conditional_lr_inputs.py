"""Prepare the 35 original, 19 matched raw, 19 residual and 2 index targets."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import run_case1_final_common_calendar_35_series as final
import run_case1_factor_robustness_markets as factor
import run_identity_sm_skewt_sector_five_models as sectors
import run_static_q0_single_stocks as stocks

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output_case1_conditional_lr"
INPUTS = ROOT.parent / "data" / "arc_sm_case1_conditional_lr"
CROSS_SECTIONS = ("sp500", "n225") + tuple(sectors.SECTORS)


def load_index(market):
    ticker = "^GSPC" if market == "sp500" else "^N225"
    path = INPUTS / f"yahoo_{market}_index.json"
    if not path.exists():
        INPUTS.mkdir(parents=True, exist_ok=True)
        stocks.fetch_yahoo_cache(ticker, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(payload["records"])
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.sort_values("date").drop_duplicates("date")
    frame["market_log_return"] = np.log(frame["adj_close"]).diff()
    frame = frame.loc[np.isfinite(frame.market_log_return)].copy()
    return frame[["date", "market_log_return"]], {
        "index_ticker": ticker, "index_source": payload.get("source"),
        "index_cache": str(path.relative_to(ROOT.parent)),
        "index_cache_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def write_panel(key, panel, label, family, target, parent, metadata):
    panel = final.trim_panel(panel)
    INPUTS.mkdir(parents=True, exist_ok=True)
    path = INPUTS / f"{key}.npz"
    dates = pd.DatetimeIndex(panel.daily.date)
    date_values = dates.to_numpy(dtype="datetime64[ns]")
    unchanged = False
    if path.exists():
        # NPZ ZIP timestamps would change the byte hash on every rebuild even
        # for identical observations. Preserve identical inputs for resume.
        with np.load(path) as existing:
            unchanged = (np.array_equal(existing["values"],panel.values)
                and np.array_equal(existing["offsets"],panel.offsets)
                and np.array_equal(existing["dates"],date_values))
    if not unchanged:
        np.savez_compressed(path, values=panel.values, offsets=panel.offsets, dates=date_values)
    n_train = int((dates <= final.TRAINING_END).sum())
    counts = np.diff(panel.offsets)
    assert n_train > 22 and n_train < len(dates)
    return {"dataset": key, "label": label, "family": family,
            "target": target, "parent_dataset": parent,
            "input": str(path.relative_to(ROOT.parent)),
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "actual_start": str(dates[0].date()), "training_end": str(dates[n_train-1].date()),
            "actual_oos_start": str(dates[n_train].date()), "end": str(dates[-1].date()),
            "training_days": n_train, "oos_days": len(dates)-n_train,
            "training_observations": int(panel.offsets[n_train]),
            "observations": len(panel.values),
            "N_min": int(counts.min()), "N_median": float(np.median(counts)),
            "N_max": int(counts.max()), **metadata}


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    entries, diagnostics = [], []
    for key in final.DATASETS:
        if key in sectors.SECTORS:
            panel, info = sectors.load_sector(key)
        elif key in final.base.FX:
            panel, info = final.base.fx.load_fx(key)
        else:
            panel, info = final.base.locked.load_dataset(key)
        old = json.loads((final.OUTPUT/key/"estimation.json").read_text(encoding="utf-8"))
        label = old["label"]
        entries.append(write_panel(key, panel, label, final.family(key), "raw", key, {"original_source": info}))
        print("prepared", key, flush=True)
    indices = {}
    for market in ("sp500", "n225"):
        index, source = load_index(market)
        indices[market] = (index, source)
        entries.append(write_panel(f"{market}_index", factor.to_panel(index, "market_log_return"),
                                  f"{factor.MARKETS[market]['label']} index", "2 indices", "index", market, source))
        spec = factor.MARKETS[market]
        broad = pd.read_parquet(spec["source"], columns=["date", "firm_id", "log_return"])
        broad["date"] = pd.to_datetime(broad.date).dt.normalize()
        broad["firm_id"] = broad.firm_id.astype(str)
        broad = broad.loc[np.isfinite(broad.log_return) & broad.date.between(final.START,final.END)].copy()
        assert not broad.duplicated(["date","firm_id"]).any()
        merged = broad.merge(index, on="date", how="inner", validate="many_to_one")
        residualized = factor.causal_rolling_residuals(merged)
        valid = residualized.loc[np.isfinite(residualized.idiosyncratic_return)].copy()
        for key in CROSS_SECTIONS:
            if key != market and not (key in sectors.SECTORS and sectors.SECTORS[key]["market"] == market):
                continue
            if key == market:
                matched = valid.copy()
            else:
                ss = sectors.SECTORS[key]
                path = sectors.DATA_ROOT / market / "daily_returns" / f"{ss['code']}.parquet"
                sf = pd.read_parquet(path, columns=["date","firm_id","log_return"])
                sf["date"] = pd.to_datetime(sf.date).dt.normalize()
                sf["firm_id"] = sf.firm_id.astype(str)
                sf = sf.loc[np.isfinite(sf.log_return) & sf.date.between(final.START,final.END)]
                assert not sf.duplicated(["date","firm_id"]).any()
                matched = sf.merge(valid, on=["date","firm_id"], how="inner",
                                   suffixes=("_sector",""), validate="one_to_one")
                if not np.allclose(matched.log_return_sector, matched.log_return, atol=1e-12, rtol=0):
                    raise RuntimeError(f"{key}: sector and broad-market raw returns differ")
            matched = matched.sort_values(["date","firm_id"]).reset_index(drop=True)
            raw_panel = factor.to_panel(matched, "log_return")
            residual_panel = factor.to_panel(matched, "idiosyncratic_return")
            assert np.array_equal(raw_panel.offsets,residual_panel.offsets)
            assert raw_panel.daily.date.equals(residual_panel.daily.date)
            label = next(e["label"] for e in entries if e["dataset"]==key)
            meta = {**source, "market": market, "rolling_window": 252,
                    "minimum_lagged_observations": 126,
                    "regression_history": "all prior available broad-market stock observations, including before a sector change",
                    "matched_asset_days": len(matched)}
            for target, panel in (("raw_matched",raw_panel),("idiosyncratic",residual_panel)):
                entries.append(write_panel(f"{key}_{target}",panel,f"{label} {target}",
                    final.family(key),target,key,meta))
            for split, block in (("training",matched.loc[matched.date<=final.TRAINING_END]),
                                 ("oos",matched.loc[matched.date>=final.OOS_START])):
                daily = block.groupby("date")[["log_return","idiosyncratic_return","market_log_return"]].mean()
                # Weighted mean pairwise covariance / variance ratio. This is
                # a linear-dependence diagnostic, not proof of independence.
                dep = {}
                for col, name in (("log_return","raw"),("idiosyncratic_return","residual")):
                    wide = block.pivot(index="date",columns="firm_id",values=col)
                    corr = wide.corr(min_periods=60).to_numpy()
                    mask = np.triu(np.ones(corr.shape,dtype=bool),1) & np.isfinite(corr)
                    pairs = corr[mask]
                    dep[f"{name}_pair_correlation_mean"] = float(np.mean(pairs)) if len(pairs) else None
                    dep[f"{name}_pair_abs_correlation_mean"] = float(np.mean(np.abs(pairs))) if len(pairs) else None
                    dep[f"{name}_pair_count"] = int(len(pairs))
                diagnostics.append({"dataset": key, "split": split,
                    "raw_mean_market_correlation": float(daily.log_return.corr(daily.market_log_return)),
                    "residual_mean_market_correlation": float(daily.idiosyncratic_return.corr(daily.market_log_return)),
                    **dep})
            print("prepared matched pair",key,len(matched),flush=True)
    if len(entries)!=75 or len(set(e["dataset"] for e in entries))!=75:
        raise RuntimeError("expected exactly 75 targets")
    manifest = {"analysis":"Case1 conditional LR, fixed fitted upstream paths",
                "requested_start":"2016-07-05", "training_end":"2023-12-29",
                "oos_start":"2024-01-02", "end":"2026-07-24",
                "original_targets":35,"matched_raw_targets":19,"residual_targets":19,"index_targets":2,
                "entries":entries}
    (OUTPUT/"input_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    pd.DataFrame([{k:v for k,v in e.items() if not isinstance(v,dict)} for e in entries]).to_csv(OUTPUT/"case1_samples.csv",index=False)
    pd.DataFrame(diagnostics).to_csv(OUTPUT/"factor_dependence_diagnostics.csv",index=False)
    return manifest


if __name__ == "__main__":
    prepare()
