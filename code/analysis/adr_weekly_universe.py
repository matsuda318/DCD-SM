"""Estimate weekly point-in-time ADR sample sizes from public price histories.

Inputs:
  - Citi sponsored NYSE/Nasdaq ADR program segments
  - Current Canadian NYSE/Nasdaq interlisted stocks (lower-bound universe)

For each ticker, Yahoo daily adjusted closes and volume are retrieved.  A daily
return counts only when adjacent observations are at most seven calendar days
apart, the current bar has positive volume, and both dates fall inside the
program's active interval.  Ticker/CUSIP changes are joined using underlying
ISIN where available.
"""

from __future__ import annotations

import csv
import datetime as dt
import html
import json
import math
import re
import statistics
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from adr_coverage_research import TableParser


# Use complete Monday--Friday weeks.  The surrounding query window still
# includes earlier prices so that Monday returns can use the previous close.
START = dt.date(2016, 7, 4)
END = dt.date(2026, 7, 24)
START_TS = int(dt.datetime(2016, 6, 20, tzinfo=dt.timezone.utc).timestamp())
END_TS = int(dt.datetime(2026, 7, 30, tzinfo=dt.timezone.utc).timestamp())
USER_AGENT = "Mozilla/5.0 (compatible; academic ADR weekly universe research)"
CANADA_URL = (
    "https://topforeignstocks.com/foreign-adrs-list/"
    "the-complete-list-of-canada-stocks-trading-on-us-markets/"
)


def fetch(url: str, attempts: int = 3, timeout: int = 25) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/json"},
    )
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as exc:
            last_error = exc
            time.sleep(1.25 * (attempt + 1))
    assert last_error is not None
    raise last_error


def normalize_company(value: str) -> str:
    value = html.unescape(value).lower()
    return re.sub(r"[^a-z0-9]+", "", value)


def parse_program_date(value: str, default: dt.date) -> dt.date:
    if not value:
        return default
    for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return dt.datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return default


def load_adr_programs() -> list[dict[str, str]]:
    path = (
        Path("results")
        / "adr_point_in_time"
        / "level2plus_programs_2016_2026.csv"
    )
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["source"] = "Citi DR directory"
        row["firm_id"] = (
            f"{row['country_code']}|isin|{row['underlying_isin']}"
            if row["underlying_isin"]
            else f"{row['country_code']}|company|{normalize_company(row['company'])}"
        )
        row["interval_start"] = parse_program_date(row["effective_date"], START)
        row["interval_end"] = parse_program_date(row["inactive_date"], END)
    return rows


def load_canada_current() -> list[dict[str, str]]:
    parser = TableParser()
    parser.feed(fetch(CANADA_URL).decode("utf-8", errors="replace"))
    output: list[dict[str, str]] = []
    for exchange, table in zip(("NYSE", "NASDAQ"), parser.tables[:2]):
        header = [re.sub(r"[^a-z0-9]+", " ", value.lower()).strip() for value in table[0]]
        for cells in table[1:]:
            record = dict(zip(header, cells))
            ticker = record.get("ticker", "").strip().upper()
            company = record.get("name", "").strip()
            if not ticker or not re.search(r"[A-Z0-9]", ticker):
                continue
            output.append(
                {
                    "country_code": "can",
                    "country": "Canada",
                    "company": company,
                    "ticker": ticker,
                    "exchange": exchange,
                    "underlying_isin": "",
                    "effective_date": "",
                    "inactive_date": "",
                    "source": "current Canadian interlisted list; survivor lower bound",
                    "firm_id": f"can|company|{normalize_company(company)}",
                    "interval_start": START,
                    "interval_end": END,
                }
            )
    return output


def yahoo_history(ticker: str) -> dict[str, object]:
    symbol = urllib.parse.quote(ticker.replace(".", "-"), safe="-")
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={START_TS}&period2={END_TS}&interval=1d"
        "&events=history&includeAdjustedClose=true"
    )
    try:
        payload = json.loads(fetch(url, attempts=2, timeout=18).decode("utf-8"))
        result = payload.get("chart", {}).get("result")
        if not result:
            return {
                "ticker": ticker,
                "records": [],
                "error": json.dumps(payload.get("chart", {}).get("error")),
            }
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
            date = dt.datetime.fromtimestamp(int(timestamp), tz=dt.timezone.utc).date()
            prices.append(
                (
                    date,
                    float(close),
                    float(volume) if volume is not None else 0.0,
                )
            )
        records: list[
            tuple[dt.date, dt.date, float, float, float, float]
        ] = []
        for previous, current in zip(prices, prices[1:]):
            previous_date, previous_close, _ = previous
            current_date, current_close, current_volume = current
            if (current_date - previous_date).days > 7 or current_volume <= 0:
                continue
            if previous_close <= 0 or current_close <= 0:
                continue
            log_return = math.log(current_close / previous_close)
            records.append(
                (
                    previous_date,
                    current_date,
                    log_return,
                    current_close * current_volume,
                    current_close,
                    current_volume,
                )
            )
        return {
            "ticker": ticker,
            "records": records,
            "error": "",
            "price_start": prices[0][0].isoformat() if prices else "",
            "price_end": prices[-1][0].isoformat() if prices else "",
            "daily_prices": len(prices),
        }
    except Exception as exc:
        return {
            "ticker": ticker,
            "records": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def week_start(date: dt.date) -> dt.date:
    return date - dt.timedelta(days=date.weekday())


def quantile(values: list[int], probability: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main() -> None:
    programs = load_adr_programs() + load_canada_current()
    ticker_programs: dict[str, list[dict[str, str]]] = defaultdict(list)
    for program in programs:
        ticker = program["ticker"].strip().upper()
        if ticker:
            ticker_programs[ticker].append(program)

    print(
        f"Programs={len(programs)}, unique tickers with symbols={len(ticker_programs)}"
    )

    # (country, week, firm) -> set of daily return dates.  Sets prevent double
    # counting where a ticker/CUSIP successor carries backfilled price history.
    firm_week_dates: dict[tuple[str, dt.date, str], set[dt.date]] = defaultdict(set)
    firm_week_dollar_volume: dict[
        tuple[str, dt.date, str], list[float]
    ] = defaultdict(list)
    # One return for each country/firm/date.  The composite key prevents a
    # successor ticker with backfilled prices from duplicating an observation.
    firm_daily_returns: dict[
        tuple[str, dt.date, str, dt.date], float
    ] = {}
    ticker_status: list[dict[str, str]] = []

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(yahoo_history, ticker): ticker
            for ticker in sorted(ticker_programs)
        }
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            ticker = str(result["ticker"])
            records = result.get("records") or []
            for program in ticker_programs[ticker]:
                interval_start = program["interval_start"]
                interval_end = program["interval_end"]
                for (
                    previous_date,
                    current_date,
                    log_return,
                    dollar_volume,
                    _,
                    _,
                ) in records:
                    if (
                        previous_date < interval_start
                        or current_date > interval_end
                        or current_date < START
                        or current_date > END
                    ):
                        continue
                    key = (
                        program["country_code"],
                        week_start(current_date),
                        program["firm_id"],
                    )
                    firm_week_dates[key].add(current_date)
                    firm_week_dollar_volume[key].append(float(dollar_volume))
                    firm_daily_returns.setdefault(
                        (
                            program["country_code"],
                            week_start(current_date),
                            program["firm_id"],
                            current_date,
                        ),
                        float(log_return),
                    )
            ticker_status.append(
                {
                    "ticker": ticker,
                    "program_segments": str(len(ticker_programs[ticker])),
                    "daily_prices": str(result.get("daily_prices", 0)),
                    "price_start": str(result.get("price_start", "")),
                    "price_end": str(result.get("price_end", "")),
                    "return_records": str(len(records)),
                    "error": str(result.get("error", "")),
                }
            )
            completed += 1
            if completed % 100 == 0:
                print(f"Yahoo histories completed: {completed}/{len(futures)}")

    # Official program count, independent of Yahoo availability.
    weeks: list[dt.date] = []
    cursor = week_start(START)
    while cursor <= END:
        weeks.append(cursor)
        cursor += dt.timedelta(days=7)

    countries = {
        program["country_code"]: program["country"] for program in programs
    }
    official_firms: dict[tuple[str, dt.date], set[str]] = defaultdict(set)
    for program in programs:
        first_week = week_start(max(program["interval_start"], START))
        last_week = week_start(min(program["interval_end"], END))
        cursor = first_week
        while cursor <= last_week:
            official_firms[(program["country_code"], cursor)].add(program["firm_id"])
            cursor += dt.timedelta(days=7)

    firm_keys_by_country_week: dict[
        tuple[str, dt.date], list[tuple[str, dt.date, str]]
    ] = defaultdict(list)
    for key in firm_week_dates:
        firm_keys_by_country_week[(key[0], key[1])].append(key)

    weekly_rows: list[dict[str, str | int | float]] = []
    for code, country in sorted(countries.items(), key=lambda item: item[1]):
        for week in weeks:
            keys = firm_keys_by_country_week[(code, week)]
            observed_firms = {key[2] for key in keys if firm_week_dates[key]}
            return_count = sum(len(firm_week_dates[key]) for key in keys)
            liquid_firms = {
                key[2]
                for key in keys
                if firm_week_dollar_volume[key]
                and statistics.median(firm_week_dollar_volume[key]) >= 1_000_000
            }
            liquid_return_count = sum(
                len(firm_week_dates[key])
                for key in keys
                if key[2] in liquid_firms
            )
            weekly_rows.append(
                {
                    "country_code": code,
                    "country": country,
                    "week_start": week.isoformat(),
                    "official_active_M": len(official_firms[(code, week)]),
                    "observed_M": len(observed_firms),
                    "observed_daily_returns": return_count,
                    "liquid_M_1m": len(liquid_firms),
                    "liquid_daily_returns_1m": liquid_return_count,
                }
            )

    output_dir = Path("results") / "adr_point_in_time"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "ticker_price_status.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fields = list(ticker_status[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(ticker_status, key=lambda row: row["ticker"]))

    with (output_dir / "weekly_country_coverage.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fields = list(weekly_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(weekly_rows)

    summary_rows: list[dict[str, str | int | float]] = []
    for code, country in sorted(countries.items(), key=lambda item: item[1]):
        rows = [row for row in weekly_rows if row["country_code"] == code]
        observed = [int(row["observed_M"]) for row in rows]
        returns = [int(row["observed_daily_returns"]) for row in rows]
        liquid = [int(row["liquid_M_1m"]) for row in rows]
        official = [int(row["official_active_M"]) for row in rows]
        summary_rows.append(
            {
                "country_code": code,
                "country": country,
                "weeks": len(rows),
                "official_M_min": min(official),
                "official_M_p10": f"{quantile(official, 0.10):.1f}",
                "official_M_median": f"{statistics.median(official):.1f}",
                "official_M_max": max(official),
                "observed_M_min": min(observed),
                "observed_M_p10": f"{quantile(observed, 0.10):.1f}",
                "observed_M_median": f"{statistics.median(observed):.1f}",
                "observed_M_max": max(observed),
                "weeks_M0": sum(value == 0 for value in observed),
                "weeks_M1": sum(value == 1 for value in observed),
                "weeks_M_lt5": sum(value < 5 for value in observed),
                "daily_returns_median": f"{statistics.median(returns):.1f}",
                "daily_returns_p10": f"{quantile(returns, 0.10):.1f}",
                "liquid_M_median_1m": f"{statistics.median(liquid):.1f}",
            }
        )

    with (output_dir / "country_coverage_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fields = list(summary_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    returns_by_week_country: dict[
        tuple[dt.date, str], list[float]
    ] = defaultdict(list)
    for (code, week, _, _), value in firm_daily_returns.items():
        returns_by_week_country[(week, code)].append(value)

    national_rows: list[dict[str, str | int | float]] = []
    for week in weeks:
        week_country_rows = [
            row for row in weekly_rows if row["week_start"] == week.isoformat()
        ]
        returns_by_country = {
            code: returns_by_week_country[(week, code)]
            for code in countries
            if returns_by_week_country[(week, code)]
        }
        pooled_returns = np.asarray(
            [
                value
                for values in returns_by_country.values()
                for value in values
            ],
            dtype=float,
        )
        country_means = [
            statistics.mean(values) for values in returns_by_country.values()
        ]
        national_rows.append(
            {
                "week_start": week.isoformat(),
                "official_active_M_strict_ADR16": sum(
                    int(row["official_active_M"])
                    for row in week_country_rows
                    if row["country_code"] != "can"
                ),
                "observed_M_strict_ADR16": sum(
                    int(row["observed_M"])
                    for row in week_country_rows
                    if row["country_code"] != "can"
                ),
                "observed_M_all17_including_Canada": sum(
                    int(row["observed_M"]) for row in week_country_rows
                ),
                "pooled_daily_return_n": len(pooled_returns),
                "pooled_daily_log_return_mean": (
                    float(np.mean(pooled_returns)) if len(pooled_returns) else ""
                ),
                "pooled_daily_log_return_p05": (
                    float(np.quantile(pooled_returns, 0.05))
                    if len(pooled_returns)
                    else ""
                ),
                "pooled_daily_log_return_p95": (
                    float(np.quantile(pooled_returns, 0.95))
                    if len(pooled_returns)
                    else ""
                ),
                "country_equal_log_return_mean": (
                    statistics.mean(country_means) if country_means else ""
                ),
            }
        )

    with (output_dir / "weekly_national_aggregate.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fields = list(national_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(national_rows)

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    dates = [
        dt.date.fromisoformat(str(row["week_start"])) for row in national_rows
    ]

    fig, axis = plt.subplots(figsize=(12, 5.6), constrained_layout=True)
    axis.plot(
        dates,
        [row["official_active_M_strict_ADR16"] for row in national_rows],
        color="#252525",
        linewidth=1.5,
        label="Official active programs: strict ADR, 16 countries",
    )
    axis.plot(
        dates,
        [row["observed_M_strict_ADR16"] for row in national_rows],
        color="#2171b5",
        linewidth=1.35,
        label="Return-observed firms: strict ADR, 16 countries",
    )
    axis.plot(
        dates,
        [row["observed_M_all17_including_Canada"] for row in national_rows],
        color="#e6550d",
        linewidth=1.2,
        alpha=0.9,
        label="Return-observed firms: 17 countries incl. Canada",
    )
    axis.set(
        title="Weekly number of US-traded foreign firms",
        xlabel="Week",
        ylabel="Number of firms",
    )
    axis.xaxis.set_major_locator(mdates.YearLocator(1))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.7)
    axis.legend(frameon=False, ncol=1, loc="upper left")
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    fig.savefig(plot_dir / "weekly_total_adr_count.png", dpi=180)
    fig.savefig(plot_dir / "weekly_total_adr_count.svg")
    plt.close(fig)

    mean_pct = np.asarray(
        [float(row["pooled_daily_log_return_mean"]) for row in national_rows]
    ) * 100
    p05_pct = np.asarray(
        [float(row["pooled_daily_log_return_p05"]) for row in national_rows]
    ) * 100
    p95_pct = np.asarray(
        [float(row["pooled_daily_log_return_p95"]) for row in national_rows]
    ) * 100
    fig, axis = plt.subplots(figsize=(12, 5.6), constrained_layout=True)
    axis.fill_between(
        dates,
        p05_pct,
        p95_pct,
        color="#9ecae1",
        alpha=0.55,
        linewidth=0,
        label="Pooled 5th–95th percentile",
    )
    axis.plot(
        dates,
        mean_pct,
        color="#084594",
        linewidth=1.15,
        label="Pooled weekly mean",
    )
    axis.axhline(0, color="#636363", linewidth=0.8)
    axis.set(
        title="Weekly distribution of observed daily log returns, 17 countries",
        xlabel="Week",
        ylabel="Daily log return (%)",
    )
    axis.xaxis.set_major_locator(mdates.YearLocator(1))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.7)
    axis.legend(frameon=False, loc="lower left")
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    fig.savefig(plot_dir / "weekly_pooled_return_mean_p05_p95.png", dpi=180)
    fig.savefig(plot_dir / "weekly_pooled_return_mean_p05_p95.svg")
    plt.close(fig)

    print("\nPoint-in-time weekly coverage based on Yahoo-observed returns:")
    for row in summary_rows:
        print(
            f"{row['country']:<16} "
            f"M median={row['observed_M_median']:>6} "
            f"p10={row['observed_M_p10']:>5} "
            f"min={row['observed_M_min']:>3} max={row['observed_M_max']:>3} "
            f"weeks<5={row['weeks_M_lt5']:>3}/{row['weeks']} "
            f"n_returns median={row['daily_returns_median']:>6}"
        )


if __name__ == "__main__":
    main()
