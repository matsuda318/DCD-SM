"""Build auditable S&P 500 and Nikkei 225 sector-level daily return panels.

The S&P 500 classification is reconstructed from every dated revision of the
locally retained ``datasets/s-and-p-500-companies`` Git repository.  The
Nikkei 225 membership is the existing official point-in-time ledger.  Current
Nikkei classifications come from the official component CSV; the 43 former
constituents are assigned an explicitly audited Nikkei industry, which maps to
the six-sector system used throughout the locked sample.

No return cleaning or balanced-panel restriction is introduced.  Within each
market, sector, and date, the script retains the raw log return used by the new
identity-SM skew-t M1--M5 specification.  Realized cross-sectional mean,
population variance, and exactly standardized return are auxiliary diagnostics,
not model inputs.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
sys.path.insert(0, str(REPO / "dcd_nikkei225_daily_composite_score_arch"))

from build_sp500_industry_hourly_data import (  # noqa: E402
    _normal_ticker,
    _run_git,
    append_manual_classifications,
)
from build_sp500_sector_daily_data import assign_sector_classifications  # noqa: E402
from build_nikkei225_daily_data import (  # noqa: E402
    COMPONENT_CHANGES,
    reconstruct_membership,
)


START = pd.Timestamp("2016-07-05")
END = pd.Timestamp("2026-07-24")
TRAIN_END = pd.Timestamp("2023-12-29")

SP500_RETURNS = REPO / "data" / "weekly_density_em" / "daily_returns" / "usa.parquet"
SP500_CLASSIFICATION_REPO = REPO / "data" / "_external" / "s-and-p-500-companies"
N225_RETURNS = REPO / "data" / "nikkei225_daily" / "daily_returns.parquet"
N225_CURRENT = (
    REPO
    / "dcd_nikkei225_daily_composite_score_arch"
    / "source_cache"
    / "current_component_weights.csv"
)
N225_HISTORY = (
    REPO
    / "dcd_nikkei225_daily_composite_score_arch"
    / "source_cache"
    / "component_changes.pdf"
)
DEFAULT_OUTPUT = REPO / "data" / "sp500_n225_sector_daily_2016_2026"

SP500_SECTOR_CODES = {
    "Energy": 10,
    "Materials": 15,
    "Industrials": 20,
    "Consumer Discretionary": 25,
    "Consumer Staples": 30,
    "Health Care": 35,
    "Financials": 40,
    "Information Technology": 45,
    "Communication Services": 50,
    "Utilities": 55,
    "Real Estate": 60,
}

SP500_LEGACY_SECTOR_ALIASES = {
    "Telecommunications Services": "Communication Services",
    "Telecommunication Services": "Communication Services",
    "Technology": "Information Technology",
}

# The return panel uses the eventual ticker across its complete price history,
# while the public constituent snapshots retained the contemporaneous ticker.
SP500_RETURN_TO_CLASSIFICATION_TICKER = {"JEF": "LUK"}

N225_INDUSTRY_TO_SECTOR = {
    "Construction": "Capital Goods/Others",
    "Machinery": "Capital Goods/Others",
    "Other Manufacturing": "Capital Goods/Others",
    "Real Estate": "Capital Goods/Others",
    "Shipbuilding": "Capital Goods/Others",
    "Fishery": "Consumer Goods",
    "Foods": "Consumer Goods",
    "Retail": "Consumer Goods",
    "Services": "Consumer Goods",
    "Banking": "Financials",
    "Insurance": "Financials",
    "Other Financial Services": "Financials",
    "Securities": "Financials",
    "Chemicals": "Materials",
    "Glass & Ceramics": "Materials",
    "Mining": "Materials",
    "Nonferrous Metals": "Materials",
    "Petroleum": "Materials",
    "Pulp & Paper": "Materials",
    "Rubber": "Materials",
    "Steel": "Materials",
    "Textiles & Apparel": "Materials",
    "Trading Companies": "Materials",
    "Automobiles & Auto parts": "Technology",
    "Communications": "Technology",
    "Electric Machinery": "Technology",
    "Pharmaceuticals": "Technology",
    "Precision Instruments": "Technology",
    "Air Transport": "Transportation and Utilities",
    "Electric Power": "Transportation and Utilities",
    "Gas": "Transportation and Utilities",
    "Land Transport": "Transportation and Utilities",
    "Marine Transport": "Transportation and Utilities",
    "Railway & Bus": "Transportation and Utilities",
    "Warehousing": "Transportation and Utilities",
}

# Codes absent from the current official component CSV.  Company names are
# reconciled to Nikkei's official component-change ledger; industries use the
# Nikkei 36-industry terminology and are deliberately reviewable here.
N225_HISTORICAL_INDUSTRIES = {
    "1333": ("Maruha Nichiro", "Fishery"),
    "2531": ("Takara Holdings", "Foods"),
    "3101": ("Toyobo", "Textiles & Apparel"),
    "3103": ("Unitika", "Textiles & Apparel"),
    # Nikkei changed 3105 from Textiles & Apparel to Electric Machinery on
    # 2016-04-01, before this sample begins (Nikkei Index News, 2016-03-22).
    "3105": ("Nisshinbo Holdings", "Electric Machinery"),
    "3863": ("Nippon Paper Industries", "Pulp & Paper"),
    "3865": ("Hokuetsu Kishu Paper", "Pulp & Paper"),
    "4041": ("Nippon Soda", "Chemicals"),
    "4272": ("Nippon Kayaku", "Chemicals"),
    "4631": ("DIC", "Chemicals"),
    "5002": ("Showa Shell Sekiyu", "Petroleum"),
    "5202": ("Nippon Sheet Glass", "Glass & Ceramics"),
    "5232": ("Sumitomo Osaka Cement", "Glass & Ceramics"),
    "5413": ("Nisshin Steel", "Steel"),
    "5541": ("Pacific Metals", "Steel"),
    "5703": ("Nippon Light Metal Holdings", "Nonferrous Metals"),
    "5707": ("Toho Zinc", "Nonferrous Metals"),
    "5715": ("Furukawa", "Nonferrous Metals"),
    "5901": ("Toyo Seikan Group Holdings", "Other Manufacturing"),
    "6366": ("Chiyoda", "Construction"),
    "6502": ("Toshiba", "Electric Machinery"),
    "6508": ("Meidensha", "Electric Machinery"),
    "6594": ("Nidec", "Electric Machinery"),
    "6674": ("GS Yuasa", "Electric Machinery"),
    "6703": ("Oki Electric Industry", "Electric Machinery"),
    "6767": ("Mitsumi Electric", "Electric Machinery"),
    "6773": ("Pioneer", "Electric Machinery"),
    "6952": ("Casio Computer", "Electric Machinery"),
    "7003": ("Mitsui E&S", "Shipbuilding"),
    "7205": ("Hino Motors", "Automobiles & Auto parts"),
    "7762": ("Citizen Watch", "Precision Instruments"),
    "8028": ("FamilyMart", "Retail"),
    "8270": ("UNY Group Holdings", "Retail"),
    "8303": ("Shinsei Bank", "Banking"),
    "8355": ("Shizuoka Bank", "Banking"),
    "8628": ("Matsui Securities", "Securities"),
    "8729": ("Sony Financial Holdings", "Insurance"),
    "9062": ("Nippon Express", "Land Transport"),
    "9301": ("Mitsubishi Logistics", "Warehousing"),
    "9412": ("SKY Perfect JSAT Holdings", "Communications"),
    "9437": ("NTT Docomo", "Communications"),
    "9613": ("NTT Data Group", "Communications"),
    "9681": ("Tokyo Dome", "Services"),
}

N225_SECTOR_CODES = {
    "Technology": "n1",
    "Financials": "n2",
    "Consumer Goods": "n3",
    "Materials": "n4",
    "Capital Goods/Others": "n5",
    "Transportation and Utilities": "n6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", choices=("sp500", "n225", "both"), default="both")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def add_sector_standardization(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    groups = result.groupby(["date", "sector_code"], sort=True)["log_return"]
    result["sector_n"] = groups.transform("size").astype(int)
    result["sector_rm"] = groups.transform("mean")
    centered = result["log_return"] - result["sector_rm"]
    result["sector_rv"] = centered.pow(2).groupby(
        [result["date"], result["sector_code"]], sort=True
    ).transform("mean")
    if (result["sector_rv"] <= 0.0).any() or result["sector_rv"].isna().any():
        raise ValueError("nonpositive or missing sector-day population variance")
    result["standardized_return"] = centered / np.sqrt(result["sector_rv"])
    return result


def load_sp500_sector_snapshots(
    repo: Path, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read both legacy sector-only and modern GICS Git snapshots."""
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
    prior = _run_git(
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
    rows = []
    for line in [*log_text.splitlines(), prior]:
        if not line.strip():
            continue
        commit_hash, commit_date = line.split(",", maxsplit=1)
        rows.append({"commit_hash": commit_hash, "snapshot_date": pd.Timestamp(commit_date)})
    commits = (
        pd.DataFrame(rows)
        .drop_duplicates("commit_hash")
        .sort_values(["snapshot_date", "commit_hash"])
        .reset_index(drop=True)
    )
    broad_sector_names = set(SP500_SECTOR_CODES) | set(SP500_LEGACY_SECTOR_ALIASES)
    snapshots = []
    schemas = []
    invalid_counts = []
    for row in commits.itertuples(index=False):
        content = _run_git(repo, ["show", f"{row.commit_hash}:data/constituents.csv"])
        frame = pd.read_csv(io.StringIO(content))
        if "Security" not in frame and "Company" in frame:
            frame = frame.rename(columns={"Company": "Security"})
        if {"Symbol", "Security", "GICS Sector"}.issubset(frame.columns):
            if "GICS Sub-Industry" not in frame:
                frame["GICS Sub-Industry"] = "Not available in source snapshot"
            schema = "modern_gics"
        elif {"Symbol", "Name", "Sector"}.issubset(frame.columns):
            frame = frame.rename(columns={"Name": "Security", "Sector": "GICS Sector"})
            observed = set(frame["GICS Sector"].dropna().astype(str))
            if len(observed - broad_sector_names) <= 1:
                frame["GICS Sub-Industry"] = "Not available in legacy sector-only snapshot"
                schema = "legacy_sector_only"
            else:
                frame["GICS Sub-Industry"] = frame["GICS Sector"]
                schema = "legacy_subindustry_only"
        else:
            raise ValueError(
                f"unrecognized constituent schema at {row.commit_hash}: {frame.columns.tolist()}"
            )
        valid_symbol = frame["Symbol"].astype(str).str.fullmatch(r"[A-Za-z0-9.\-]+")
        invalid_counts.append(int((~valid_symbol).sum()))
        frame = frame.loc[valid_symbol].copy()
        frame["GICS Sector"] = frame["GICS Sector"].replace(SP500_LEGACY_SECTOR_ALIASES)
        frame = frame[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].copy()
        frame["ticker_key"] = _normal_ticker(frame["Symbol"])
        frame["snapshot_date"] = row.snapshot_date
        frame["commit_hash"] = row.commit_hash
        frame["source_schema"] = schema
        snapshots.append(frame)
        schemas.append(schema)
    history = pd.concat(snapshots, ignore_index=True)
    modern = history.loc[history["source_schema"] == "modern_gics"]
    subindustry_only = history["source_schema"] == "legacy_subindustry_only"
    for index, row in history.loc[subindustry_only].iterrows():
        same_ticker = history["ticker_key"] == row["ticker_key"]
        prior = history.loc[
            same_ticker
            & (history["source_schema"] == "legacy_sector_only")
            & (history["snapshot_date"] < row["snapshot_date"])
        ].sort_values("snapshot_date")
        future = modern.loc[
            (modern["ticker_key"] == row["ticker_key"])
            & (modern["snapshot_date"] > row["snapshot_date"])
        ].sort_values("snapshot_date")
        if not prior.empty:
            history.at[index, "GICS Sector"] = prior.iloc[-1]["GICS Sector"]
        elif not future.empty:
            history.at[index, "GICS Sector"] = future.iloc[0]["GICS Sector"]
        else:
            history.at[index, "GICS Sector"] = np.nan
    if history.loc[subindustry_only, "GICS Sector"].isna().any():
        unknown = sorted(
            history.loc[
                subindustry_only & history["GICS Sector"].isna(), "GICS Sub-Industry"
            ].unique()
        )
        raise ValueError(f"unmapped legacy sub-industries: {unknown}")
    history = (
        history
        .drop_duplicates(["snapshot_date", "ticker_key"], keep="last")
        .sort_values(["ticker_key", "snapshot_date"])
    )
    commits["source_schema"] = schemas
    commits["invalid_symbol_rows_excluded"] = invalid_counts
    return history, commits


def compressed_spells(history: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ticker, group in history.groupby("ticker_key", sort=True):
        ordered = group.sort_values("snapshot_date").drop_duplicates(
            "snapshot_date", keep="last"
        )
        changed = (
            ordered[["GICS Sector", "GICS Sub-Industry"]]
            .ne(ordered[["GICS Sector", "GICS Sub-Industry"]].shift())
            .any(axis=1)
        )
        states = ordered.loc[changed].copy()
        next_dates = states["snapshot_date"].shift(-1)
        for position, (_, row) in enumerate(states.iterrows()):
            valid_to = end if pd.isna(next_dates.iloc[position]) else next_dates.iloc[position] - pd.Timedelta(days=1)
            rows.append(
                {
                    "ticker": ticker,
                    "company": row["Security"],
                    "valid_from": row["snapshot_date"],
                    "valid_to": valid_to,
                    "sector": row["GICS Sector"],
                    "industry": row["GICS Sub-Industry"],
                    "source_date": row["snapshot_date"],
                    "source_id": row["commit_hash"],
                    "classification_method": row.get("source_schema", "dated_public_snapshot"),
                    "confidence": "dated_public_snapshot",
                }
            )
    return pd.DataFrame(rows)


def augment_sp500_history_for_sparse_snapshots(
    history: pd.DataFrame, returns: pd.DataFrame
) -> pd.DataFrame:
    """Make sparse public snapshots usable without hiding source timing.

    New constituents between public Git snapshots are backcast only to their
    first observed point-in-time return.  Three known GICS-wide implementation
    dates are inserted from the surrounding public classifications.
    """
    output = history.copy()
    additions = []
    first_returns = returns.groupby("ticker_key", sort=True)["date"].min()
    for ticker, first_return in first_returns.items():
        ticker_history = output.loc[output["ticker_key"] == ticker].sort_values(
            "snapshot_date"
        )
        if ticker_history.empty or first_return >= ticker_history.iloc[0]["snapshot_date"]:
            continue
        row = ticker_history.iloc[0].copy()
        source_date = row["snapshot_date"]
        if (
            row["GICS Sector"] == "Real Estate"
            and first_return < pd.Timestamp("2016-09-01")
        ):
            row["GICS Sector"] = "Financials"
        row["snapshot_date"] = first_return
        row["commit_hash"] = f"backcast:{row['commit_hash']}"
        row["source_schema"] = "backcast_from_first_public_snapshot"
        row["classification_source_date"] = source_date
        additions.append(row)
    if additions:
        output = pd.concat([output, pd.DataFrame(additions)], ignore_index=True)

    transitions = (
        (pd.Timestamp("2016-09-01"), "Real Estate", "official_gics_real_estate_2016"),
        (pd.Timestamp("2018-10-01"), "Communication Services", "official_gics_communication_2018"),
        (pd.Timestamp("2023-03-20"), None, "official_gics_revision_2023"),
    )
    transition_rows = []
    for effective, target_sector, source_id in transitions:
        members = set(
            returns.loc[
                returns["date"].between(effective, effective + pd.Timedelta(days=7)),
                "ticker_key",
            ]
        )
        for ticker in sorted(members):
            ticker_history = output.loc[output["ticker_key"] == ticker].sort_values(
                "snapshot_date"
            )
            prior = ticker_history.loc[ticker_history["snapshot_date"] < effective]
            future = ticker_history.loc[ticker_history["snapshot_date"] >= effective]
            if prior.empty or future.empty:
                continue
            before = prior.iloc[-1]
            if target_sector is None:
                after_candidates = future
            else:
                after_candidates = future.loc[future["GICS Sector"] == target_sector]
            if after_candidates.empty:
                continue
            after = after_candidates.iloc[0]
            if before["GICS Sector"] == after["GICS Sector"]:
                continue
            row = after.copy()
            row["snapshot_date"] = effective
            row["commit_hash"] = source_id
            row["source_schema"] = "official_transition_date_with_snapshot_inference"
            row["classification_source_date"] = after["snapshot_date"]
            transition_rows.append(row)
    if transition_rows:
        output = pd.concat([output, pd.DataFrame(transition_rows)], ignore_index=True)
    # TripAdvisor left the index before the next retained public snapshot, so
    # the general surrounding-snapshot rule cannot see its 2018 GICS move.
    trip = output.loc[
        (output["ticker_key"] == "TRIP")
        & (output["snapshot_date"] < pd.Timestamp("2018-10-01"))
    ].sort_values("snapshot_date")
    if not trip.empty:
        row = trip.iloc[-1].copy()
        row["GICS Sector"] = "Communication Services"
        row["GICS Sub-Industry"] = "Interactive Media & Services"
        row["snapshot_date"] = pd.Timestamp("2018-10-01")
        row["commit_hash"] = "official_gics_communication_2018:TRIP"
        row["source_schema"] = "official_transition_explicit_historical_constituent"
        row["classification_source_date"] = pd.Timestamp("2018-10-01")
        output = pd.concat([output, row.to_frame().T], ignore_index=True)
    if "classification_source_date" not in output:
        output["classification_source_date"] = output["snapshot_date"]
    output["snapshot_date"] = pd.to_datetime(output["snapshot_date"])
    output["classification_source_date"] = pd.to_datetime(
        output["classification_source_date"]
    )
    output["classification_source_date"] = output["classification_source_date"].fillna(
        output["snapshot_date"]
    )
    return output.sort_values(["ticker_key", "snapshot_date"]).reset_index(drop=True)


def build_sp500(output: Path) -> pd.DataFrame:
    returns = pd.read_parquet(SP500_RETURNS)
    returns["date"] = pd.to_datetime(returns["date"])
    returns = returns.loc[
        returns["date"].between(START, END) & np.isfinite(returns["log_return"])
    ].copy()
    returns["ticker_key"] = _normal_ticker(returns["ticker"])
    history, commits = load_sp500_sector_snapshots(
        SP500_CLASSIFICATION_REPO, START, END
    )
    history, overrides = append_manual_classifications(history)
    aliases = []
    for return_ticker, source_ticker in SP500_RETURN_TO_CLASSIFICATION_TICKER.items():
        copied = history.loc[history["ticker_key"] == source_ticker].copy()
        if copied.empty:
            raise ValueError(f"missing classification alias source {source_ticker}")
        copied["ticker_key"] = return_ticker
        copied["Symbol"] = return_ticker
        copied["Security"] = "Jefferies Financial Group"
        copied["commit_hash"] = copied["commit_hash"].astype(str) + f":alias-{source_ticker}-to-{return_ticker}"
        aliases.append(copied)
    if aliases:
        history = pd.concat([history, *aliases], ignore_index=True)
    history = augment_sp500_history_for_sparse_snapshots(history, returns)
    history["Sector"] = history["GICS Sector"]
    history["Sector Code"] = history["GICS Sector"].map(SP500_SECTOR_CODES)
    if history["Sector Code"].isna().any():
        raise ValueError("unrecognized GICS sector")
    assigned = assign_sector_classifications(returns, history)
    metadata = history[
        ["ticker_key", "commit_hash", "source_schema", "classification_source_date"]
    ].drop_duplicates(["ticker_key", "commit_hash"], keep="last")
    assigned = assigned.merge(
        metadata,
        on=["ticker_key", "commit_hash"],
        how="left",
        validate="many_to_one",
    )
    assigned = assigned.rename(
        columns={
            "ticker_key": "identifier",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "industry",
        }
    )
    assigned["market"] = "S&P 500"
    assigned["sector_code"] = "s" + assigned["Sector Code"].astype(int).astype(str)
    assigned["classification_method"] = assigned["source_schema"]
    assigned["classification_source_after_return"] = (
        pd.to_datetime(assigned["classification_source_date"])
        > pd.to_datetime(assigned["date"])
    )
    assigned["classification_confidence"] = assigned["source_schema"].map(
        {
            "legacy_sector_only": "dated_public_snapshot",
            "legacy_subindustry_only": "dated_public_snapshot_inferred_sector",
            "modern_gics": "dated_public_snapshot",
            "backcast_from_first_public_snapshot": "audited_backcast",
            "official_transition_date_with_snapshot_inference": "official_date_inferred_constituents",
            "official_transition_explicit_historical_constituent": "official_date_explicit_constituent",
        }
    ).fillna("manual_or_alias")
    assigned = add_sector_standardization(assigned)
    target = output / "sp500"
    target.mkdir(parents=True, exist_ok=True)
    commits.to_csv(target / "classification_snapshot_commits.csv", index=False)
    overrides.to_csv(target / "manual_classification_overrides.csv", index=False)
    compressed_spells(history, END).to_csv(target / "classification_spells.csv", index=False)
    save_market_outputs(assigned, target)
    return assigned


def n225_classification_table() -> pd.DataFrame:
    current = pd.read_csv(N225_CURRENT, dtype={"Code": str}, encoding="cp932")
    current["Code"] = current["Code"].astype(str).str.strip().str.upper()
    current = current.loc[current["Code"].str.fullmatch(r"[0-9A-Z]{4}")].copy()
    current_industry = "Nikkei Industrial Classification"
    official = pd.DataFrame(
        {
            "identifier": current["Code"],
            "company": current["Company Name"],
            "industry": current[current_industry],
            "sector": current["Sector"],
            "classification_method": "official current Nikkei component CSV",
            "classification_confidence": "official_current",
            "source_id": str(N225_CURRENT.relative_to(REPO)).replace("\\", "/"),
        }
    )
    historical = pd.DataFrame(
        [
            {
                "identifier": code,
                "company": company,
                "industry": industry,
                "sector": N225_INDUSTRY_TO_SECTOR[industry],
                "classification_method": "audited Nikkei 36-industry assignment for former constituent",
                "classification_confidence": "manual_high",
                "source_id": str(N225_HISTORY.relative_to(REPO)).replace("\\", "/"),
            }
            for code, (company, industry) in N225_HISTORICAL_INDUSTRIES.items()
        ]
    )
    table = pd.concat([official, historical], ignore_index=True)
    if table["identifier"].duplicated().any():
        raise ValueError("duplicate N225 classification")
    return table.sort_values("identifier").reset_index(drop=True)


def build_n225(output: Path) -> pd.DataFrame:
    classifications = n225_classification_table()
    current = set(
        classifications.loc[
            classifications["classification_confidence"] == "official_current",
            "identifier",
        ]
    )
    start_members, snapshots = reconstruct_membership(current)
    union = set().union(*snapshots.values())
    classified = set(classifications["identifier"])
    if union != classified:
        raise ValueError(
            f"N225 classification coverage mismatch: missing={sorted(union-classified)}, extra={sorted(classified-union)}"
        )
    returns = pd.read_parquet(N225_RETURNS)
    returns["date"] = pd.to_datetime(returns["date"])
    returns = returns.loc[
        returns["date"].between(START, END) & np.isfinite(returns["log_return"])
    ].copy()
    returns["identifier"] = returns["code"].astype(str).str.strip().str.upper()
    assigned = returns.merge(
        classifications,
        on="identifier",
        how="left",
        validate="many_to_one",
        suffixes=("", "_classification"),
    )
    if assigned["sector"].isna().any():
        raise ValueError("unclassified N225 return rows")
    assigned["market"] = "Nikkei 225"
    assigned["sector_code"] = assigned["sector"].map(N225_SECTOR_CODES)
    assigned["classification_used_future_snapshot"] = False
    assigned["classification_lag_days"] = np.nan
    assigned = add_sector_standardization(assigned)
    target = output / "n225"
    target.mkdir(parents=True, exist_ok=True)
    classifications.assign(valid_from=START, valid_to=END).to_csv(
        target / "classification_spells.csv", index=False
    )
    save_market_outputs(assigned, target)
    return assigned


def save_market_outputs(frame: pd.DataFrame, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    returns_dir = target / "daily_returns"
    returns_dir.mkdir(parents=True, exist_ok=True)
    frame = frame.sort_values(["date", "sector_code", "identifier"]).reset_index(drop=True)
    frame.to_parquet(target / "daily_returns_all_sectors.parquet", index=False, compression="zstd")
    summaries = []
    for code, group in frame.groupby("sector_code", sort=True):
        group.to_parquet(returns_dir / f"{code}.parquet", index=False, compression="zstd")
        daily_n = group.groupby("date")["identifier"].nunique()
        summaries.append(
            {
                "market": str(group["market"].iloc[0]),
                "sector_code": code,
                "sector": str(group["sector"].iloc[0]),
                "n_returns": len(group),
                "n_tickers": group["identifier"].nunique(),
                "n_dates": group["date"].nunique(),
                "date_min": group["date"].min().date(),
                "date_max": group["date"].max().date(),
                "daily_n_min": int(daily_n.min()),
                "daily_n_median": float(daily_n.median()),
                "daily_n_max": int(daily_n.max()),
                "future_snapshot_rows": int(group["classification_used_future_snapshot"].sum()),
            }
        )
    pd.DataFrame(summaries).to_csv(target / "sector_summary.csv", index=False)
    day = (
        frame.groupby(["market", "date", "sector_code", "sector"], sort=True)
        .agg(
            n=("identifier", "nunique"),
            rm=("sector_rm", "first"),
            rv=("sector_rv", "first"),
            standardized_mean=("standardized_return", "mean"),
            standardized_variance=("standardized_return", lambda x: float(np.mean(np.square(x)))),
        )
        .reset_index()
    )
    day.to_parquet(target / "sector_day_summary.parquet", index=False, compression="zstd")


def write_manifest(output: Path, frames: dict[str, pd.DataFrame]) -> None:
    payload = {
        "study_start": START.date().isoformat(),
        "study_end": END.date().isoformat(),
        "training_end": TRAIN_END.date().isoformat(),
        "return_cleaning": "none beyond pre-existing finite adjusted-close return construction",
        "primary_model_input": "raw within-sector daily log returns; no RM/RV pre-normalization",
        "auxiliary_diagnostics": "same-day within-sector realized mean, population variance, and standardized return",
        "markets": {
            name: {
                "rows": int(frame.shape[0]),
                "dates": int(frame["date"].nunique()),
                "sectors": int(frame["sector_code"].nunique()),
                "tickers": int(frame["identifier"].nunique()),
                "future_snapshot_rows": int(frame["classification_used_future_snapshot"].sum()),
                "later_public_classification_source_rows": int(
                    frame.get(
                        "classification_source_after_return",
                        pd.Series(False, index=frame.index),
                    ).sum()
                ),
            }
            for name, frame in frames.items()
        },
        "sp500_classification": (
            "dated Git revisions of a public constituent/GICS table, augmented "
            "by the official 2016, 2018, and 2023 GICS transition dates and "
            "explicitly flagged first-public-snapshot backcasts"
        ),
        "sp500_classification_caveat": (
            "the public Git history is sparse; this is an auditable public-data "
            "reconstruction rather than a licensed daily historical GICS file"
        ),
        "n225_membership": "official current set reverse-replayed through official component changes",
        "n225_classification": "official current CSV plus explicit audited industries for 43 former constituents",
        "n225_caveat": "former-constituent classifications are audited reconstructions, not licensed historical daily constituent files",
    }
    (output / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    if args.market in {"sp500", "both"}:
        frames["sp500"] = build_sp500(output)
    if args.market in {"n225", "both"}:
        frames["n225"] = build_n225(output)
    write_manifest(output, frames)
    for market, frame in frames.items():
        print(
            f"{market}: rows={len(frame):,}, dates={frame.date.nunique():,}, "
            f"sectors={frame.sector_code.nunique()}, tickers={frame.identifier.nunique()}, "
            f"future_snapshot_rows={int(frame.classification_used_future_snapshot.sum()):,}",
            flush=True,
        )


if __name__ == "__main__":
    main()
