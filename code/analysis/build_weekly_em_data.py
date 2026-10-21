"""Build the country-week daily-return dataset used by density EM.

The analysis universe is:
  * sponsored NYSE/Nasdaq ADRs for 15 countries,
  * Canadian securities directly interlisted in the United States, and
  * point-in-time S&P 500 constituents for the United States.

Indonesia is deliberately excluded.  Daily adjusted-close log returns are
stored in country-partitioned Parquet files.  Each country/week group is the
sample to which the density-mixture EM is applied.
"""

from __future__ import annotations

import bisect
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import pickle
import statistics
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from adr_weekly_universe import (
    END,
    START,
    fetch,
    load_adr_programs,
    load_canada_current,
    week_start,
    yahoo_history,
)


OUTPUT_DIR = Path("data") / "weekly_density_em"
DAILY_DIR = OUTPUT_DIR / "daily_returns"
CACHE_DIR = OUTPUT_DIR / "_yahoo_cache"
SP500_MEMBERSHIP_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20%28Updated%29.csv"
)

COLUMNS = [
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
    "universe_start",
    "universe_end",
]


def load_sp500_membership() -> tuple[
    list[dt.date], list[set[str]], pd.DataFrame
]:
    request = urllib.request.Request(
        SP500_MEMBERSHIP_URL,
        headers={"User-Agent": "academic density-valued financial research"},
    )
    raw = fetch(request.full_url)
    frame = pd.read_csv(io.BytesIO(raw))
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    frame = frame.loc[frame["date"] <= END].copy()
    if frame.empty or frame["date"].iloc[-1] < START:
        raise RuntimeError("S&P 500 membership source does not cover the study window")
    dates = frame["date"].tolist()
    members = [
        {ticker.strip().upper() for ticker in str(value).split(",") if ticker.strip()}
        for value in frame["tickers"]
    ]
    return dates, members, frame


def membership_on(
    date: dt.date,
    membership_dates: list[dt.date],
    membership_sets: list[set[str]],
) -> set[str]:
    position = bisect.bisect_right(membership_dates, date) - 1
    return membership_sets[position] if position >= 0 else set()


def quantile(series: pd.Series, probability: float) -> float:
    return float(series.quantile(probability))


def cached_yahoo_history(ticker: str) -> dict[str, object]:
    digest = hashlib.sha1(ticker.encode("utf-8")).hexdigest()[:12]
    target = CACHE_DIR / f"{digest}.pkl.gz"
    if target.exists():
        with gzip.open(target, "rb") as handle:
            return pickle.load(handle)
    result = yahoo_history(ticker)
    with gzip.open(target, "wb", compresslevel=5) as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return result


def write_country(
    code: str,
    records: dict[tuple[str, dt.date], tuple[object, ...]],
) -> dict[str, object]:
    frame = pd.DataFrame.from_records(list(records.values()), columns=COLUMNS)
    frame["week_start"] = pd.to_datetime(frame["week_start"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame["universe_start"] = pd.to_datetime(frame["universe_start"])
    frame["universe_end"] = pd.to_datetime(frame["universe_end"])
    frame = frame.sort_values(["week_start", "date", "firm_id", "ticker"])
    duplicate_count = int(
        frame.duplicated(["country_code", "firm_id", "date"]).sum()
    )
    if duplicate_count:
        raise RuntimeError(f"{code}: {duplicate_count} duplicate firm-date rows")
    target = DAILY_DIR / f"{code}.parquet"
    frame.to_parquet(target, index=False, compression="zstd")
    return {
        "country_code": code,
        "country": str(frame["country"].iloc[0]),
        "rows": len(frame),
        "firms": int(frame["firm_id"].nunique()),
        "weeks": int(frame["week_start"].nunique()),
        "date_min": frame["date"].min().date().isoformat(),
        "date_max": frame["date"].max().date().isoformat(),
        "file": target.as_posix(),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # The foreign-country analysis sample excludes Indonesia by design.
    programs = [
        row
        for row in load_adr_programs() + load_canada_current()
        if row["country_code"] != "indo"
    ]
    ticker_programs: dict[str, list[dict[str, object]]] = defaultdict(list)
    for program in programs:
        ticker = str(program["ticker"]).strip().upper()
        if ticker:
            ticker_programs[ticker].append(program)

    membership_dates, membership_sets, membership_frame = load_sp500_membership()
    window_membership_sets = [
        members
        for date, members in zip(membership_dates, membership_sets)
        if START <= date <= END
    ]
    if membership_dates[-1] < END:
        window_membership_sets.append(membership_sets[-1])
    sp500_tickers = set().union(*window_membership_sets)
    all_tickers = sorted(set(ticker_programs) | sp500_tickers)
    print(
        f"Foreign tickers={len(ticker_programs)}, "
        f"S&P500 PIT tickers={len(sp500_tickers)}, "
        f"union={len(all_tickers)}"
    )

    histories: dict[str, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(cached_yahoo_history, ticker): ticker
            for ticker in all_tickers
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            histories[str(result["ticker"])] = result
            if completed % 100 == 0:
                print(f"Yahoo histories completed: {completed}/{len(futures)}")

    records_by_country: dict[
        str, dict[tuple[str, dt.date], tuple[object, ...]]
    ] = defaultdict(dict)

    # Foreign ADR and Canadian direct-interlisting observations.
    for ticker, ticker_rows in ticker_programs.items():
        records = histories[ticker].get("records") or []
        for program in ticker_rows:
            interval_start = program["interval_start"]
            interval_end = program["interval_end"]
            for (
                previous_date,
                current_date,
                log_return,
                dollar_volume,
                adjusted_close,
                volume,
            ) in records:
                if (
                    previous_date < interval_start
                    or current_date > interval_end
                    or current_date < START
                    or current_date > END
                ):
                    continue
                code = str(program["country_code"])
                firm_id = str(program["firm_id"])
                key = (firm_id, current_date)
                records_by_country[code].setdefault(
                    key,
                    (
                        code,
                        str(program["country"]),
                        week_start(current_date),
                        current_date,
                        firm_id,
                        ticker,
                        str(program["company"]),
                        (
                            "Canadian direct interlisting"
                            if code == "can"
                            else "sponsored exchange-listed ADR"
                        ),
                        log_return,
                        adjusted_close,
                        volume,
                        dollar_volume,
                        interval_start,
                        interval_end,
                    ),
                )

    # Point-in-time S&P 500 membership.  The last published membership is
    # carried forward from its source date to the end of the study window.
    usa_records: dict[tuple[str, dt.date], tuple[object, ...]] = {}
    for ticker in sp500_tickers:
        records = histories[ticker].get("records") or []
        for (
            _,
            current_date,
            log_return,
            dollar_volume,
            adjusted_close,
            volume,
        ) in records:
            if current_date < START or current_date > END:
                continue
            if ticker not in membership_on(
                current_date, membership_dates, membership_sets
            ):
                continue
            firm_id = f"usa|sp500_ticker|{ticker}"
            usa_records[(firm_id, current_date)] = (
                "usa",
                "United States",
                week_start(current_date),
                current_date,
                firm_id,
                ticker,
                "",
                "point-in-time S&P 500 constituent",
                log_return,
                adjusted_close,
                volume,
                dollar_volume,
                None,
                None,
            )
    records_by_country["usa"] = usa_records

    summaries: list[dict[str, object]] = []
    weekly_frames: list[pd.DataFrame] = []
    for code in sorted(records_by_country):
        records = records_by_country[code]
        if not records:
            raise RuntimeError(f"No observations for country {code}")
        summary = write_country(code, records)
        summaries.append(summary)
        frame = pd.DataFrame.from_records(list(records.values()), columns=COLUMNS)
        grouped = frame.groupby(
            ["country_code", "country", "week_start"], sort=True
        )
        weekly = grouped.agg(
            n_returns=("log_return", "size"),
            n_firms=("firm_id", "nunique"),
            n_dates=("date", "nunique"),
            return_mean=("log_return", "mean"),
            return_median=("log_return", "median"),
            return_min=("log_return", "min"),
            return_max=("log_return", "max"),
        ).reset_index()
        percentiles = grouped["log_return"].agg(
            return_p05=lambda values: quantile(values, 0.05),
            return_p95=lambda values: quantile(values, 0.95),
        ).reset_index()
        weekly_frames.append(
            weekly.merge(
                percentiles,
                on=["country_code", "country", "week_start"],
                how="left",
                validate="one_to_one",
            )
        )
        print(
            f"{summary['country']:<16} rows={summary['rows']:>8,} "
            f"firms={summary['firms']:>4} weeks={summary['weeks']:>3}"
        )

    weekly_index = pd.concat(weekly_frames, ignore_index=True).sort_values(
        ["week_start", "country_code"]
    )
    weekly_index.to_csv(
        OUTPUT_DIR / "country_week_index.csv",
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )
    pd.DataFrame(summaries).sort_values("country").to_csv(
        OUTPUT_DIR / "country_file_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    status_rows = []
    for ticker in all_tickers:
        result = histories[ticker]
        status_rows.append(
            {
                "ticker": ticker,
                "in_foreign_universe": ticker in ticker_programs,
                "in_sp500_pit_universe": ticker in sp500_tickers,
                "daily_prices": result.get("daily_prices", 0),
                "price_start": result.get("price_start", ""),
                "price_end": result.get("price_end", ""),
                "return_records": len(result.get("records") or []),
                "error": result.get("error", ""),
            }
        )
    pd.DataFrame(status_rows).to_csv(
        OUTPUT_DIR / "ticker_fetch_status.csv",
        index=False,
        encoding="utf-8-sig",
    )

    membership_frame.to_csv(
        OUTPUT_DIR / "sp500_membership_source.csv.gz",
        index=False,
        compression="gzip",
        date_format="%Y-%m-%d",
    )
    with (OUTPUT_DIR / "foreign_program_universe.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = []
        for program in programs:
            row = {
                key: value.isoformat() if isinstance(value, dt.date) else value
                for key, value in program.items()
            }
            rows.append(row)
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "study_start": START.isoformat(),
        "study_end": END.isoformat(),
        "countries": len(summaries),
        "country_codes": sorted(records_by_country),
        "rows": sum(int(row["rows"]) for row in summaries),
        "sp500_membership_source_url": SP500_MEMBERSHIP_URL,
        "sp500_membership_source_last_date": membership_dates[-1].isoformat(),
        "sp500_membership_carried_forward_through": END.isoformat(),
        "foreign_universe_segments_or_current_entries": len(programs),
        "unique_requested_tickers": len(all_tickers),
        "ticker_fetch_errors": sum(
            bool(histories[ticker].get("error")) for ticker in all_tickers
        ),
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
