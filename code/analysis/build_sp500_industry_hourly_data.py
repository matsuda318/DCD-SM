"""Build GICS-industry-group daily samples from S&P 500 hourly returns.

The source hourly data were already downloaded for the ADR experiment.  This
script reuses the point-in-time U.S. observations, removes overnight returns,
and assigns every ticker-date to one of the 25 standard GICS industry groups.

Historical constituent classifications are reconstructed from dated versions
of ``datasets/s-and-p-500-companies``.  For each ticker-date, the latest
available classification snapshot on or before that date is used.  If a new
constituent has no prior snapshot, its first later snapshot is used and the
fallback is recorded.

No balanced security panel is imposed: ticker entry, exit, relisting, missing
bars, and unequal observation counts are retained.  The manifest only records
whether every industry group happens to have at least one return on each date.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RETURNS = (
    REPO_ROOT
    / "data"
    / "daily_density_hourly"
    / "hourly_returns"
    / "usa.parquet"
)
DEFAULT_OUTPUT = REPO_ROOT / "data" / "sp500_industry_hourly"
GICS_URL = (
    "https://gist.githubusercontent.com/shadiakiki1986/"
    "7c873ea0050cdb7a81f597fb30e5936c/raw/"
    "06448e172709ec9945c071b86cf1048884e430bc/"
    "GICS%20Mappings%20-%20March%202023%20Update.csv"
)
MANUAL_CLASSIFICATION_OVERRIDES = (
    {
        "Symbol": "SOLS",
        "Security": "Solstice Advanced Materials",
        "GICS Sector": "Materials",
        "GICS Sub-Industry": "Specialty Chemicals",
        "snapshot_date": "2025-10-30",
        "commit_hash": "manual_sp_global_release_2025-10-27",
        "source_url": (
            "https://press.spglobal.com/2025-10-27-"
            "Solstice-Advance-Materials-and-Qnity-Electronics-"
            "Set-to-Join-S-P-500-Others-to-Join-S-P-SmallCap-600"
        ),
        "reason": (
            "S&P official addition release identifies SOLS as Materials; "
            "Materials maps uniquely to GICS industry group 1510"
        ),
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--returns", type=Path, default=DEFAULT_RETURNS)
    parser.add_argument(
        "--classification-repo",
        type=Path,
        required=True,
        help=(
            "Local clone of datasets/s-and-p-500-companies with Git history."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", type=str, default="2024-08-05")
    parser.add_argument("--end", type=str, default="2026-07-24")
    parser.add_argument(
        "--maximum-interval-minutes",
        type=float,
        default=90.0,
        help="Retain only adjacent same-day hourly closes.",
    )
    return parser.parse_args()


def _run_git(repo: Path, arguments: list[str]) -> str:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo.resolve()}",
            "-C",
            str(repo),
            *arguments,
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout


def _normal_name(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _normal_ticker(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.upper()
        .str.strip()
        .str.replace(".", "-", regex=False)
    )


def load_gics_map(output_dir: Path) -> pd.DataFrame:
    source_dir = output_dir / "classification_sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    target = source_dir / "gics_mapping_march_2023.csv"
    if not target.exists():
        request = urllib.request.Request(
            GICS_URL,
            headers={"User-Agent": "academic density-valued research"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            target.write_bytes(response.read())
    mapping = pd.read_csv(target)
    mapping["subindustry_key"] = _normal_name(mapping["Sub-Industry"])
    mapping = mapping[
        [
            "subindustry_key",
            "Industry Group Code",
            "Industry Group",
            "Sector Code",
            "Sector",
        ]
    ].drop_duplicates("subindustry_key")
    return mapping


def load_classification_snapshots(
    repo: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (repo / ".git").exists():
        raise FileNotFoundError(f"not a Git repository: {repo}")
    log_text = _run_git(
        repo,
        [
            "log",
            "--format=%H,%as",
            "--date=short",
            f"--since={(start - pd.Timedelta(days=60)).date()}",
            f"--until={(end + pd.Timedelta(days=60)).date()}",
            "--",
            "data/constituents.csv",
        ],
    )
    commit_rows = []
    for line in log_text.splitlines():
        if not line.strip():
            continue
        commit_hash, commit_date = line.split(",", maxsplit=1)
        commit_rows.append(
            {
                "commit_hash": commit_hash,
                "snapshot_date": pd.Timestamp(commit_date),
            }
        )
    prior_text = _run_git(
        repo,
        [
            "log",
            "--format=%H,%as",
            "--date=short",
            f"--before={start.date()}",
            "-1",
            "--",
            "data/constituents.csv",
        ],
    ).strip()
    if prior_text:
        commit_hash, commit_date = prior_text.split(",", maxsplit=1)
        commit_rows.append(
            {
                "commit_hash": commit_hash,
                "snapshot_date": pd.Timestamp(commit_date),
            }
        )
    commits = (
        pd.DataFrame(commit_rows)
        .drop_duplicates("commit_hash")
        .sort_values("snapshot_date")
        .reset_index(drop=True)
    )
    snapshots = []
    for row in commits.itertuples(index=False):
        content = _run_git(
            repo,
            ["show", f"{row.commit_hash}:data/constituents.csv"],
        )
        frame = pd.read_csv(io.StringIO(content))
        if "Security" not in frame and "Company" in frame:
            frame = frame.rename(columns={"Company": "Security"})
        required = {
            "Symbol",
            "Security",
            "GICS Sector",
            "GICS Sub-Industry",
        }
        if not required.issubset(frame.columns):
            raise ValueError(
                f"snapshot {row.commit_hash} lacks {required - set(frame)}"
            )
        frame = frame[
            ["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]
        ].copy()
        frame["ticker_key"] = _normal_ticker(frame["Symbol"])
        frame["subindustry_key"] = _normal_name(
            frame["GICS Sub-Industry"]
        )
        frame["snapshot_date"] = row.snapshot_date
        frame["commit_hash"] = row.commit_hash
        snapshots.append(frame)
    history = pd.concat(snapshots, ignore_index=True)
    history = history.drop_duplicates(
        ["snapshot_date", "ticker_key"], keep="last"
    ).sort_values(["ticker_key", "snapshot_date"])
    return history, commits


def append_manual_classifications(
    history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overrides = pd.DataFrame(MANUAL_CLASSIFICATION_OVERRIDES)
    if overrides.empty:
        return history, overrides
    overrides["ticker_key"] = _normal_ticker(overrides["Symbol"])
    overrides["subindustry_key"] = _normal_name(
        overrides["GICS Sub-Industry"]
    )
    overrides["snapshot_date"] = pd.to_datetime(
        overrides["snapshot_date"]
    )
    history_columns = history.columns
    combined = pd.concat(
        [history, overrides.reindex(columns=history_columns)],
        ignore_index=True,
    )
    combined = combined.drop_duplicates(
        ["snapshot_date", "ticker_key"], keep="last"
    ).sort_values(["ticker_key", "snapshot_date"])
    return combined, overrides


def attach_gics_groups(
    history: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    classified = history.merge(
        mapping,
        on="subindustry_key",
        how="left",
        validate="many_to_one",
    )
    missing = classified["Industry Group Code"].isna()
    reit = (
        missing
        & classified["subindustry_key"].str.endswith("REITs")
        & ~classified["subindustry_key"].str.startswith("Mortgage")
    )
    real_estate = mapping.loc[
        mapping["Industry Group"].str.contains(
            "Equity Real Estate Investment Trusts", regex=False
        )
    ].iloc[0]
    classified.loc[reit, "Industry Group Code"] = real_estate[
        "Industry Group Code"
    ]
    classified.loc[reit, "Industry Group"] = real_estate["Industry Group"]
    classified.loc[reit, "Sector Code"] = real_estate["Sector Code"]
    classified.loc[reit, "Sector"] = real_estate["Sector"]
    if classified["Industry Group Code"].isna().any():
        unknown = sorted(
            classified.loc[
                classified["Industry Group Code"].isna(),
                "subindustry_key",
            ].unique()
        )
        raise ValueError(f"unmapped GICS sub-industries: {unknown}")
    classified["industry_group_code"] = (
        classified["Industry Group Code"].astype(int).astype(str)
    )
    return classified


def _assign_one_ticker(
    dates: pd.Series,
    ticker_history: pd.DataFrame,
) -> pd.DataFrame:
    history = ticker_history.sort_values("snapshot_date")
    snapshot_dates = history["snapshot_date"].to_numpy(dtype="datetime64[ns]")
    query_dates = pd.to_datetime(dates).to_numpy(dtype="datetime64[ns]")
    indices = np.searchsorted(snapshot_dates, query_dates, side="right") - 1
    used_future = indices < 0
    indices = np.maximum(indices, 0)
    selected = history.iloc[indices].reset_index(drop=True)
    selected["classification_used_future_snapshot"] = used_future
    selected["classification_lag_days"] = (
        pd.to_datetime(dates).reset_index(drop=True)
        - selected["snapshot_date"]
    ).dt.days
    return selected


def assign_classifications(
    returns: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    pieces = []
    known_tickers = set(history["ticker_key"])
    missing_tickers = sorted(set(returns["ticker_key"]) - known_tickers)
    if missing_tickers:
        raise ValueError(
            "tickers absent from all classification snapshots: "
            + ", ".join(missing_tickers)
        )
    for ticker, frame in returns.groupby("ticker_key", sort=True):
        selected = _assign_one_ticker(
            frame["date"],
            history.loc[history["ticker_key"] == ticker],
        )
        output = frame.reset_index(drop=True).copy()
        for column in [
            "Security",
            "GICS Sector",
            "GICS Sub-Industry",
            "snapshot_date",
            "commit_hash",
            "industry_group_code",
            "Industry Group",
            "Sector Code",
            "Sector",
            "classification_used_future_snapshot",
            "classification_lag_days",
        ]:
            output[column] = selected[column].to_numpy()
        pieces.append(output)
    return pd.concat(pieces, ignore_index=True)


def main() -> None:
    args = parse_args()
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if start > end:
        raise ValueError("start must not exceed end")
    output_dir = args.output_dir.resolve()
    returns_dir = output_dir / "hourly_returns"
    output_dir.mkdir(parents=True, exist_ok=True)
    returns_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(args.returns)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.loc[
        (frame["date"] >= start)
        & (frame["date"] <= end)
        & ~frame["includes_overnight"].astype(bool)
        & (frame["interval_minutes"] <= args.maximum_interval_minutes)
    ].copy()
    frame["ticker_key"] = _normal_ticker(frame["ticker"])
    if frame.empty:
        raise ValueError("no eligible within-day hourly returns")

    mapping = load_gics_map(output_dir)
    history, commits = load_classification_snapshots(
        args.classification_repo.resolve(), start, end
    )
    history, manual_overrides = append_manual_classifications(history)
    history = attach_gics_groups(history, mapping)
    assigned = assign_classifications(frame, history)
    assigned["country_code"] = (
        "g" + assigned["industry_group_code"].astype(str)
    )
    assigned["country"] = assigned["Industry Group"].astype(str)
    assigned["security_type"] = (
        "point-in-time S&P 500 constituent; within-day hourly return"
    )
    assigned["week_start"] = (
        assigned["date"]
        - pd.to_timedelta(assigned["date"].dt.weekday, unit="D")
    ).dt.date

    source_columns = [
        "snapshot_date",
        "commit_hash",
        "ticker_key",
        "Security",
        "GICS Sector",
        "GICS Sub-Industry",
        "industry_group_code",
        "Industry Group",
        "Sector Code",
        "Sector",
    ]
    history[source_columns].drop_duplicates().to_parquet(
        output_dir / "gics_classification_history.parquet",
        index=False,
        compression="zstd",
    )
    commits.to_csv(
        output_dir / "classification_snapshot_commits.csv", index=False
    )
    manual_overrides.to_csv(
        output_dir / "manual_classification_overrides.csv", index=False
    )

    file_rows = []
    for code, group in assigned.groupby("country_code", sort=True):
        target = returns_dir / f"{code}.parquet"
        group = group.sort_values(["date", "timestamp", "ticker_key"])
        group.to_parquet(target, index=False, compression="zstd")
        file_rows.append(
            {
                "country_code": code,
                "country": str(group["country"].iloc[0]),
                "sector": str(group["Sector"].iloc[0]),
                "industry_group_code": str(
                    group["industry_group_code"].iloc[0]
                ),
                "n_returns": int(len(group)),
                "n_tickers": int(group["ticker_key"].nunique()),
                "n_dates": int(group["date"].nunique()),
                "date_min": str(group["date"].min().date()),
                "date_max": str(group["date"].max().date()),
                "fraction_exact_zero": float(
                    np.mean(group["log_return"] == 0.0)
                ),
                "file": target.relative_to(REPO_ROOT).as_posix(),
            }
        )
    file_summary = pd.DataFrame(file_rows)
    file_summary.to_csv(output_dir / "industry_group_summary.csv", index=False)

    group_day = (
        assigned.groupby(
            ["country_code", "country", "Sector", "date"], sort=True
        )
        .agg(
            n_returns=("log_return", "size"),
            n_tickers=("ticker_key", "nunique"),
            zero_return_fraction=(
                "log_return",
                lambda values: float(np.mean(values == 0.0)),
            ),
            return_mean=("log_return", "mean"),
            return_std=("log_return", "std"),
            return_min=("log_return", "min"),
            return_max=("log_return", "max"),
            n_future_classifications=(
                "classification_used_future_snapshot",
                "sum",
            ),
        )
        .reset_index()
    )
    group_day.to_csv(output_dir / "industry_group_day_index.csv", index=False)
    date_group_counts = group_day.groupby("date")["country_code"].nunique()
    manifest = {
        "study_start": args.start,
        "study_end": args.end,
        "source_returns": str(args.returns.resolve()),
        "return_definition": (
            "log difference of consecutive valid hourly closes within the "
            "same New York trading date"
        ),
        "overnight_returns_excluded": True,
        "maximum_interval_minutes": args.maximum_interval_minutes,
        "gics_level": "industry group",
        "n_industry_groups": int(assigned["country_code"].nunique()),
        "n_dates": int(assigned["date"].nunique()),
        "n_tickers": int(assigned["ticker_key"].nunique()),
        "n_returns": int(len(assigned)),
        "all_groups_observed_each_date": bool(
            date_group_counts.nunique() == 1
            and int(date_group_counts.iloc[0])
            == int(assigned["country_code"].nunique())
        ),
        "balanced_security_panel_required": False,
        "missing_security_observations": (
            "retained as missing; every available within-day return is used"
        ),
        "minimum_group_day_returns": int(group_day["n_returns"].min()),
        "median_group_day_returns": float(
            group_day["n_returns"].median()
        ),
        "maximum_group_day_returns": int(group_day["n_returns"].max()),
        "fraction_exact_zero_returns": float(
            np.mean(assigned["log_return"] == 0.0)
        ),
        "classification_snapshots": int(len(commits)),
        "classification_source": (
            "dated Git history of datasets/s-and-p-500-companies"
        ),
        "manual_classification_overrides": int(len(manual_overrides)),
        "gics_structure_source": GICS_URL,
        "rows_using_first_future_classification": int(
            assigned["classification_used_future_snapshot"].sum()
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(file_summary.to_string(index=False), flush=True)
    print(group_day["n_returns"].describe().to_string(), flush=True)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
