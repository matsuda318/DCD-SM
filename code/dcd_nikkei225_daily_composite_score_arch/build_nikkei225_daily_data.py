"""Build point-in-time Nikkei 225 daily adjusted-close returns.

The study window and return filters deliberately match the S&P 500 input used
by ``dcd_sp500_daily_composite_score_arch``.  Membership is reconstructed from
Nikkei's official current component-weight CSV by reversing the official
component changes through the start of the sample, then replaying them on
their effective dates.

The official PDF is retained locally for audit, but the small event ledger
below is transcribed explicitly.  This avoids depending on fragile PDF table
extraction and makes every membership mutation reviewable in source control.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import math
import pickle
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
SOURCE_CACHE = ROOT / "source_cache"
CURRENT_COMPONENTS = SOURCE_CACHE / "current_component_weights.csv"
HISTORY_PDF = SOURCE_CACHE / "component_changes.pdf"
OUTPUT = REPO / "data" / "nikkei225_daily"
PRICE_CACHE = OUTPUT / "_yahoo_cache"

START = dt.date(2016, 7, 4)
END = dt.date(2026, 7, 24)
QUERY_START = dt.date(2016, 6, 20)
QUERY_END_EXCLUSIVE = dt.date(2026, 7, 30)
MINIMUM_DAILY_CONSTITUENTS = 150
MAXIMUM_RETURN_GAP_DAYS = 7
OFFICIAL_CURRENT_URL = (
    "https://indexes.nikkei.co.jp/nkave/archives/file/"
    "nikkei_stock_average_weight_en.csv"
)
OFFICIAL_HISTORY_URL = (
    "https://indexes.nikkei.co.jp/nkave/archives/file/"
    "history_of_nikkei_stock_average_component_changes_en.pdf"
)
USER_AGENT = "Mozilla/5.0 (academic cross-sectional density research)"


@dataclass(frozen=True)
class ComponentChange:
    effective_date: dt.date
    deleted: tuple[str, ...]
    added: tuple[str, ...]


def _event(
    date: str, deleted: str = "", added: str = ""
) -> ComponentChange:
    return ComponentChange(
        effective_date=dt.date.fromisoformat(date),
        deleted=tuple(deleted.split()) if deleted else (),
        added=tuple(added.split()) if added else (),
    )


# Official Nikkei "History of Component Change", Update: Apr/1/2026.
# Only events needed to reconstruct the locked 2016-07-04--2026-07-24 sample
# are included.  Add-only and delete-only bridge dates are kept separate.
COMPONENT_CHANGES: tuple[ComponentChange, ...] = (
    _event("2016-08-01", "6753", "7272"),
    _event("2016-08-29", "8270", "8028"),
    _event("2016-10-03", "4041", "4755"),
    _event("2017-01-24", "6767", "4578"),
    _event("2017-08-01", "6502", "6724"),
    _event("2017-10-02", "3865 6508", "6098 6178"),
    _event("2018-10-01", "5715", "4751"),
    _event("2018-12-26", "5413", "4631"),
    _event("2019-03-18", "6773", "6645"),
    _event("2019-03-27", "5002", "5019"),
    _event("2019-08-01", "6366", "7832"),
    _event("2019-10-01", "9681", "2413"),
    _event("2020-07-29", "8729", "8697"),
    _event("2020-10-01", "4272", "9434"),
    _event("2020-10-29", "8028", "3659"),
    _event("2020-12-02", "9437", "6753"),
    _event("2021-10-01", "3105 5901 9412", "6861 6981 7974"),
    _event("2021-12-29", "9062", ""),
    _event("2022-01-05", "", "9147"),
    _event("2022-04-04", "8303", "8591"),
    _event("2022-09-29", "8355", "6594"),
    _event("2022-10-03", "3103 6703", "6273 7741"),
    _event("2022-10-04", "1333", "5831"),
    _event("2023-04-03", "3101 5703 5707", "4661 6723 9201"),
    _event("2023-10-02", "5202 7003 8628", "4385 6920 9843"),
    _event("2024-04-01", "2531 5232 5541", "3092 6146 6526"),
    _event("2024-10-01", "3863 4631", "4307 7453"),
    _event("2025-04-01", "9301", "6532"),
    _event("2025-07-04", "9613", "6963"),
    _event("2025-10-01", "7762", "3697"),
    _event("2025-11-11", "6594", "4062"),
    _event("2026-04-01", "6674 6952 7205", "285A 543A 7532"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-official", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def fetch(url: str, attempts: int = 4, timeout: int = 30) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
    )
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as exc:  # retry transient Yahoo/CDN errors
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


def refresh_official_sources() -> None:
    SOURCE_CACHE.mkdir(parents=True, exist_ok=True)
    for url, target in (
        (OFFICIAL_CURRENT_URL, CURRENT_COMPONENTS),
        (OFFICIAL_HISTORY_URL, HISTORY_PDF),
    ):
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(fetch(url))
        temporary.replace(target)


def load_current_components(
    path: Path = CURRENT_COMPONENTS,
) -> tuple[dt.date, set[str], pd.DataFrame]:
    # Nikkei's downloadable CSV is currently encoded for Japanese Windows.
    frame = pd.read_csv(path, dtype={"Code": str}, encoding="cp932")
    frame["Code"] = frame["Code"].astype(str).str.strip().str.upper()
    frame = frame.loc[frame["Code"].str.fullmatch(r"[0-9A-Z]{4}")].copy()
    frame["Date of Data"] = pd.to_datetime(frame["Date of Data"])
    if len(frame) != 225 or frame["Code"].nunique() != 225:
        raise ValueError(
            f"official current file must contain 225 unique codes, got {len(frame)}"
        )
    dates = frame["Date of Data"].dt.date.unique()
    if len(dates) != 1 or dates[0] < END:
        raise ValueError("current component snapshot does not postdate study end")
    return dates[0], set(frame["Code"]), frame


def reconstruct_membership(
    current: set[str],
    changes: tuple[ComponentChange, ...] = COMPONENT_CHANGES,
    start: dt.date = START,
) -> tuple[set[str], dict[dt.date, set[str]]]:
    if tuple(sorted(change.effective_date for change in changes)) != tuple(
        change.effective_date for change in changes
    ):
        raise ValueError("component changes are not strictly chronological")
    membership = set(current)
    for change in reversed(changes):
        if not set(change.added).issubset(membership):
            missing = set(change.added) - membership
            raise ValueError(f"reverse membership missing additions {missing}")
        if set(change.deleted) & membership:
            present = set(change.deleted) & membership
            raise ValueError(f"reverse membership already has deletions {present}")
        membership.difference_update(change.added)
        membership.update(change.deleted)

    # The ledger begins after the locked start; this is membership on START.
    start_membership = set(membership)
    snapshots: dict[dt.date, set[str]] = {start: set(membership)}
    for change in changes:
        if change.effective_date <= start:
            continue
        if not set(change.deleted).issubset(membership):
            missing = set(change.deleted) - membership
            raise ValueError(f"forward membership missing deletions {missing}")
        if set(change.added) & membership:
            present = set(change.added) & membership
            raise ValueError(f"forward membership already has additions {present}")
        membership.difference_update(change.deleted)
        membership.update(change.added)
        snapshots[change.effective_date] = set(membership)
    if membership != current:
        raise ValueError("membership replay does not recover current components")
    return start_membership, snapshots


def membership_on(
    date: dt.date, snapshots: dict[dt.date, set[str]]
) -> set[str]:
    eligible = [effective for effective in snapshots if effective <= date]
    if not eligible:
        return set()
    return snapshots[max(eligible)]


def yahoo_history(code: str) -> dict[str, object]:
    PRICE_CACHE.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(code.encode("ascii")).hexdigest()[:12]
    target = PRICE_CACHE / f"{code}_{digest}.pkl.gz"
    if target.exists():
        with gzip.open(target, "rb") as stream:
            return pickle.load(stream)

    symbol = urllib.parse.quote(f"{code}.T", safe=".")
    period1 = int(
        dt.datetime.combine(QUERY_START, dt.time(), tzinfo=dt.timezone.utc).timestamp()
    )
    period2 = int(
        dt.datetime.combine(
            QUERY_END_EXCLUSIVE, dt.time(), tzinfo=dt.timezone.utc
        ).timestamp()
    )
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={period1}&period2={period2}&interval=1d"
        "&events=history&includeAdjustedClose=true"
    )
    output: dict[str, object]
    try:
        payload = json.loads(fetch(url, attempts=3, timeout=25).decode("utf-8"))
        result = payload.get("chart", {}).get("result")
        if not result:
            output = {
                "code": code,
                "records": [],
                "error": json.dumps(payload.get("chart", {}).get("error")),
            }
        else:
            data = result[0]
            timestamps = data.get("timestamp") or []
            quote = data.get("indicators", {}).get("quote", [{}])[0]
            adjusted = data.get("indicators", {}).get("adjclose") or [{}]
            closes = adjusted[0].get("adjclose") or quote.get("close") or []
            volumes = quote.get("volume") or []
            prices: list[tuple[dt.date, float, float]] = []
            for timestamp, close, volume in zip(timestamps, closes, volumes):
                if close is None or not math.isfinite(float(close)):
                    continue
                date = dt.datetime.fromtimestamp(
                    int(timestamp), tz=dt.timezone.utc
                ).date()
                prices.append(
                    (date, float(close), float(volume) if volume is not None else 0.0)
                )
            records: list[tuple[dt.date, dt.date, float, float, float]] = []
            for previous, current_bar in zip(prices, prices[1:]):
                previous_date, previous_close, _ = previous
                current_date, current_close, current_volume = current_bar
                if (current_date - previous_date).days > MAXIMUM_RETURN_GAP_DAYS:
                    continue
                if current_volume <= 0.0 or previous_close <= 0.0 or current_close <= 0.0:
                    continue
                records.append(
                    (
                        previous_date,
                        current_date,
                        math.log(current_close / previous_close),
                        current_close,
                        current_volume,
                    )
                )
            output = {
                "code": code,
                "records": records,
                "error": "",
                "price_start": prices[0][0].isoformat() if prices else "",
                "price_end": prices[-1][0].isoformat() if prices else "",
                "daily_prices": len(prices),
            }
    except Exception as exc:
        output = {
            "code": code,
            "records": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
    with gzip.open(target, "wb", compresslevel=5) as stream:
        pickle.dump(output, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return output


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.refresh_official or not CURRENT_COMPONENTS.exists() or not HISTORY_PDF.exists():
        refresh_official_sources()
    current_date, current, current_frame = load_current_components()
    start_membership, snapshots = reconstruct_membership(current)
    event_sizes = {date: len(members) for date, members in snapshots.items()}
    if min(event_sizes.values()) < 224 or max(event_sizes.values()) > 225:
        raise ValueError(f"unexpected official membership sizes: {event_sizes}")

    union = set().union(*snapshots.values())
    print(
        f"Nikkei membership start={len(start_membership)}, current={len(current)}, "
        f"union={len(union)}",
        flush=True,
    )
    histories: dict[str, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(yahoo_history, code): code for code in sorted(union)}
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            histories[str(result["code"])] = result
            if completed % 25 == 0 or completed == len(futures):
                print(f"Yahoo histories {completed}/{len(futures)}", flush=True)

    company_by_code = dict(
        zip(current_frame["Code"], current_frame["Company Name"].fillna(""))
    )
    records: list[dict[str, object]] = []
    fetch_status: list[dict[str, object]] = []
    for code in sorted(union):
        history = histories[code]
        fetch_status.append(
            {
                "code": code,
                "ticker": f"{code}.T",
                "error": history.get("error", ""),
                "price_start": history.get("price_start", ""),
                "price_end": history.get("price_end", ""),
                "daily_prices": history.get("daily_prices", 0),
            }
        )
        for previous_date, current_day, log_return, adjusted_close, volume in history.get(
            "records", []
        ):
            if current_day < START or current_day > END:
                continue
            if code not in membership_on(current_day, snapshots):
                continue
            records.append(
                {
                    "date": current_day,
                    "firm_id": f"jpn|nikkei225_code|{code}",
                    "code": code,
                    "ticker": f"{code}.T",
                    "company": company_by_code.get(code, ""),
                    "log_return": log_return,
                    "adjusted_close": adjusted_close,
                    "volume": volume,
                    "previous_date": previous_date,
                }
            )

    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError("no Nikkei 225 daily returns were downloaded")
    frame["date"] = pd.to_datetime(frame["date"])
    frame["previous_date"] = pd.to_datetime(frame["previous_date"])
    frame = frame.sort_values(["date", "firm_id"]).reset_index(drop=True)
    if frame.duplicated(["date", "firm_id"]).any():
        raise ValueError("duplicate Nikkei constituent-date observations")
    daily_counts = frame.groupby("date")["firm_id"].nunique()
    retained_dates = daily_counts.index[
        daily_counts.to_numpy() >= MINIMUM_DAILY_CONSTITUENTS
    ]
    dropped = daily_counts.loc[~daily_counts.index.isin(retained_dates)]
    frame = frame.loc[frame["date"].isin(retained_dates)].copy()
    daily_counts = frame.groupby("date")["firm_id"].nunique()
    if daily_counts.max() > 225 or daily_counts.min() < MINIMUM_DAILY_CONSTITUENTS:
        raise ValueError("retained daily constituent count violates locked bounds")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUTPUT / "daily_returns.parquet", index=False, compression="zstd")
    pd.DataFrame(fetch_status).to_csv(OUTPUT / "ticker_fetch_status.csv", index=False)
    pd.DataFrame(
        {
            "date": daily_counts.index,
            "n_constituents": daily_counts.to_numpy(),
        }
    ).to_csv(OUTPUT / "daily_counts.csv", index=False)
    pd.DataFrame(
        [
            {
                "effective_date": change.effective_date,
                "deleted": " ".join(change.deleted),
                "added": " ".join(change.added),
                "membership_after": len(snapshots[change.effective_date]),
            }
            for change in COMPONENT_CHANGES
        ]
    ).to_csv(OUTPUT / "membership_changes.csv", index=False)

    manifest = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "study_start": START.isoformat(),
        "study_end": END.isoformat(),
        "official_current_date": current_date.isoformat(),
        "official_current_url": OFFICIAL_CURRENT_URL,
        "official_history_url": OFFICIAL_HISTORY_URL,
        "official_current_sha256": file_sha256(CURRENT_COMPONENTS),
        "official_history_sha256": file_sha256(HISTORY_PDF),
        "membership_method": "reverse current official set and replay official changes",
        "start_members": len(start_membership),
        "current_members": len(current),
        "union_codes": len(union),
        "return_source": "Yahoo chart adjusted close",
        "maximum_return_gap_days": MAXIMUM_RETURN_GAP_DAYS,
        "positive_current_volume_required": True,
        "minimum_daily_constituents_locked": MINIMUM_DAILY_CONSTITUENTS,
        "rows": len(frame),
        "dates": int(frame["date"].nunique()),
        "date_min": frame["date"].min().date().isoformat(),
        "date_max": frame["date"].max().date().isoformat(),
        "daily_count_min": int(daily_counts.min()),
        "daily_count_median": float(daily_counts.median()),
        "daily_count_max": int(daily_counts.max()),
        "dropped_dates_below_minimum": {
            str(date.date()): int(count) for date, count in dropped.items()
        },
        "ticker_fetch_errors": int(
            sum(bool(str(row["error"])) for row in fetch_status)
        ),
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
