"""Build daily cross-sectional return samples for 11 S&P 500 GICS sectors.

The source is the existing point-in-time S&P 500 adjusted-close return panel.
Each ticker-date is assigned the latest available GICS snapshot on or before
that date.  Every sector-day density then uses all constituent returns
available on that date.

No balanced security panel is imposed.  Entry, exit, ticker changes, and
missing security observations remain in the data.  The output separately
reports whether all 11 sectors happen to be represented on every date.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_sp500_industry_hourly_data import (
    _normal_name,
    _normal_ticker,
    _assign_one_ticker,
    append_manual_classifications,
    load_classification_snapshots,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RETURNS = (
    REPO_ROOT / "data" / "weekly_density_em" / "daily_returns" / "usa.parquet"
)
DEFAULT_GICS_MAP = (
    REPO_ROOT
    / "data"
    / "sp500_industry_hourly"
    / "classification_sources"
    / "gics_mapping_march_2023.csv"
)
DEFAULT_OUTPUT = REPO_ROOT / "data" / "sp500_sector_daily"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--returns", type=Path, default=DEFAULT_RETURNS)
    parser.add_argument(
        "--classification-repo",
        type=Path,
        required=True,
        help="Local datasets/s-and-p-500-companies clone with Git history.",
    )
    parser.add_argument("--gics-map", type=Path, default=DEFAULT_GICS_MAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", type=str, default="2023-07-24")
    parser.add_argument("--end", type=str, default="2026-07-24")
    return parser.parse_args()


def load_existing_gics_map(path: Path) -> pd.DataFrame:
    mapping = pd.read_csv(path)
    mapping["subindustry_key"] = _normal_name(mapping["Sub-Industry"])
    return mapping[
        [
            "subindustry_key",
            "Industry Group Code",
            "Industry Group",
            "Sector Code",
            "Sector",
        ]
    ].drop_duplicates("subindustry_key")


def attach_gics_sectors(
    history: pd.DataFrame, mapping: pd.DataFrame
) -> pd.DataFrame:
    sector_map = mapping[["Sector Code", "Sector"]].drop_duplicates().copy()
    sector_map["sector_key"] = _normal_name(sector_map["Sector"])
    if sector_map["sector_key"].duplicated().any():
        raise ValueError("GICS sector names do not map uniquely to codes")
    output = history.copy()
    output["sector_key"] = _normal_name(output["GICS Sector"])
    output = output.merge(
        sector_map,
        on="sector_key",
        how="left",
        validate="many_to_one",
    )
    if output["Sector Code"].isna().any():
        unknown = sorted(
            output.loc[output["Sector Code"].isna(), "sector_key"]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(f"unmapped GICS sectors: {unknown}")
    return output


def assign_sector_classifications(
    returns: pd.DataFrame, history: pd.DataFrame
) -> pd.DataFrame:
    pieces = []
    known_tickers = set(history["ticker_key"])
    missing_tickers = sorted(set(returns["ticker_key"]) - known_tickers)
    if missing_tickers:
        raise ValueError(
            "tickers absent from all classification snapshots: "
            + ", ".join(missing_tickers)
        )
    columns = [
        "Security",
        "GICS Sector",
        "GICS Sub-Industry",
        "snapshot_date",
        "commit_hash",
        "Sector Code",
        "Sector",
        "classification_used_future_snapshot",
        "classification_lag_days",
    ]
    for ticker, frame in returns.groupby("ticker_key", sort=True):
        selected = _assign_one_ticker(
            frame["date"],
            history.loc[history["ticker_key"] == ticker],
        )
        output = frame.reset_index(drop=True).copy()
        for column in columns:
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
    returns_dir = output_dir / "daily_returns"
    output_dir.mkdir(parents=True, exist_ok=True)
    returns_dir.mkdir(parents=True, exist_ok=True)

    returns = pd.read_parquet(args.returns)
    returns["date"] = pd.to_datetime(returns["date"])
    returns = returns.loc[
        (returns["date"] >= start) & (returns["date"] <= end)
    ].copy()
    returns = returns.loc[np.isfinite(returns["log_return"])].copy()
    returns["ticker_key"] = _normal_ticker(returns["ticker"])
    if returns.empty:
        raise ValueError("no eligible daily S&P 500 returns")
    if returns.duplicated(["ticker_key", "date"]).any():
        raise ValueError("duplicate ticker-date returns")

    mapping = load_existing_gics_map(args.gics_map.resolve())
    history, commits = load_classification_snapshots(
        args.classification_repo.resolve(), start, end
    )
    history, manual_overrides = append_manual_classifications(history)
    history = attach_gics_sectors(history, mapping)
    assigned = assign_sector_classifications(returns, history)
    assigned["country_code"] = (
        "s" + assigned["Sector Code"].astype(int).astype(str)
    )
    assigned["country"] = assigned["Sector"].astype(str)
    assigned["week_start"] = (
        assigned["date"]
        - pd.to_timedelta(assigned["date"].dt.weekday, unit="D")
    ).dt.normalize()
    assigned["security_type"] = (
        "point-in-time S&P 500 constituent; daily adjusted-close return"
    )

    source_columns = [
        "snapshot_date",
        "commit_hash",
        "ticker_key",
        "Security",
        "GICS Sector",
        "GICS Sub-Industry",
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

    output_columns = [
        "country_code",
        "country",
        "week_start",
        "date",
        "firm_id",
        "ticker",
        "company",
        "security_type",
        "log_return",
        "adjusted_close",
        "volume",
        "adjusted_dollar_volume_proxy",
        "ticker_key",
        "Security",
        "GICS Sector",
        "GICS Sub-Industry",
        "Sector Code",
        "Sector",
        "snapshot_date",
        "commit_hash",
        "classification_used_future_snapshot",
        "classification_lag_days",
    ]
    file_rows: list[dict[str, object]] = []
    for code, sector in assigned.groupby("country_code", sort=True):
        target = returns_dir / f"{code}.parquet"
        sector = sector.sort_values(["date", "ticker_key"])
        sector[output_columns].to_parquet(
            target, index=False, compression="zstd"
        )
        file_rows.append(
            {
                "country_code": code,
                "country": str(sector["country"].iloc[0]),
                "sector_code": int(sector["Sector Code"].iloc[0]),
                "n_returns": int(len(sector)),
                "n_tickers": int(sector["ticker_key"].nunique()),
                "n_dates": int(sector["date"].nunique()),
                "date_min": str(sector["date"].min().date()),
                "date_max": str(sector["date"].max().date()),
                "fraction_exact_zero": float(
                    np.mean(sector["log_return"] == 0.0)
                ),
                "file": target.relative_to(REPO_ROOT).as_posix(),
            }
        )
    file_summary = pd.DataFrame(file_rows)
    file_summary.to_csv(output_dir / "sector_summary.csv", index=False)

    grouped = assigned.groupby(
        ["country_code", "country", "date"], sort=True
    )
    sector_day = grouped.agg(
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
    ).reset_index()
    sector_day.to_csv(output_dir / "sector_day_index.csv", index=False)

    date_sector_counts = sector_day.groupby("date")[
        "country_code"
    ].nunique()
    manifest = {
        "study_start": args.start,
        "study_end": args.end,
        "source_returns": str(args.returns.resolve()),
        "price_definition": (
            "Yahoo adjusted close; corporate actions incorporated"
        ),
        "return_definition": (
            "close-to-close log difference assigned to the current "
            "trading date"
        ),
        "density_interpretation": (
            "cross-sectional distribution of point-in-time constituent "
            "daily returns within each GICS sector and date"
        ),
        "gics_level": "sector",
        "n_sectors": int(assigned["country_code"].nunique()),
        "n_dates": int(assigned["date"].nunique()),
        "n_tickers": int(assigned["ticker_key"].nunique()),
        "n_returns": int(len(assigned)),
        "all_sectors_observed_each_date": bool(
            date_sector_counts.nunique() == 1
            and int(date_sector_counts.iloc[0])
            == int(assigned["country_code"].nunique())
        ),
        "balanced_security_panel_required": False,
        "missing_security_observations": (
            "retained as missing; each sector-day likelihood uses every "
            "available point-in-time constituent return"
        ),
        "minimum_sector_day_returns": int(sector_day["n_returns"].min()),
        "median_sector_day_returns": float(
            sector_day["n_returns"].median()
        ),
        "maximum_sector_day_returns": int(sector_day["n_returns"].max()),
        "fraction_exact_zero_returns": float(
            np.mean(assigned["log_return"] == 0.0)
        ),
        "classification_snapshots": int(len(commits)),
        "classification_source": (
            "dated Git history of datasets/s-and-p-500-companies"
        ),
        "manual_classification_overrides": int(len(manual_overrides)),
        "gics_structure_source": str(args.gics_map.resolve()),
        "rows_using_first_future_classification": int(
            assigned["classification_used_future_snapshot"].sum()
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(file_summary.to_string(index=False), flush=True)
    print(sector_day["n_returns"].describe().to_string(), flush=True)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
