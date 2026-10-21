"""Research current exchange-listed ADR coverage for the repository's 20 countries.

Candidate lists are taken from the country pages linked by TopForeignStocks.
Current listing metadata are cross-checked against Nasdaq's public screener.
Approximate ten-year price availability is checked by asking Yahoo Finance for
daily observations in July 2016.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path


MAIN_URL = "https://topforeignstocks.com/foreign-adrs-list/"
NASDAQ_URL = (
    "https://api.nasdaq.com/api/screener/stocks"
    "?tableonly=true&limit=10000&offset=0&download=true"
)
USER_AGENT = "Mozilla/5.0 (compatible; academic ADR coverage research)"
CUTOFF_START = int(dt.datetime(2016, 7, 1, tzinfo=dt.timezone.utc).timestamp())
CUTOFF_END = int(dt.datetime(2016, 8, 1, tzinfo=dt.timezone.utc).timestamp())

COUNTRIES = {
    "arg": ("Argentina", "Argentina ADRs"),
    "aus": ("Australia", "Australia ADRs"),
    "bra": ("Brazil", "Brazil ADRs"),
    "can": ("Canada", "Canada Stocks Trading on US Markets"),
    "chi": ("China", "China ADRs"),
    "fra": ("France", "France ADRs"),
    "ger": ("Germany", "Germany ADRs"),
    "ind": ("India", "India ADRs"),
    "indo": ("Indonesia", "Indonesia ADRs"),
    "ita": ("Italy", "Italy ADRs"),
    "jpn": ("Japan", "Japan ADRs"),
    "mex": ("Mexico", "Mexico ADRs"),
    "rus": ("Russia", "Russia ADRs"),
    "sau": ("Saudi Arabia", None),
    "safr": ("South Africa", "South Africa ADRs"),
    "sko": ("South Korea", "Korea ADRs"),
    "spa": ("Spain", "Spain ADRs"),
    "tur": ("Turkey", "Turkey ADRs"),
    "uk": ("United Kingdom", "British ADRs"),
    "usa": ("United States", None),
}


def fetch(url: str, *, attempts: int = 4, timeout: int = 30) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/json",
        },
    )
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as exc:  # network endpoints occasionally rate-limit
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.current_href: str | None = None
        self.current_text: list[str] = []
        self.links: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self.current_href = dict(attrs).get("href")
            self.current_text = []

    def handle_data(self, data: str) -> None:
        if self.current_href is not None:
            self.current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self.current_href is not None:
            text = " ".join("".join(self.current_text).split())
            self.links.append((text, self.current_href))
            self.current_href = None
            self.current_text = []


class TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self.table_depth = 0
        self.current_table: list[list[str]] | None = None
        self.current_row: list[str] | None = None
        self.current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self.table_depth += 1
            if self.table_depth == 1:
                self.current_table = []
        elif self.table_depth == 1 and tag == "tr":
            self.current_row = []
        elif self.table_depth == 1 and tag in {"td", "th"}:
            self.current_cell = []

    def handle_data(self, data: str) -> None:
        if self.current_cell is not None:
            self.current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self.table_depth == 1 and tag in {"td", "th"}:
            assert self.current_row is not None
            value = " ".join("".join(self.current_cell or []).split())
            self.current_row.append(value)
            self.current_cell = None
        elif self.table_depth == 1 and tag == "tr":
            if self.current_table is not None and self.current_row:
                self.current_table.append(self.current_row)
            self.current_row = None
        elif tag == "table":
            if self.table_depth == 1 and self.current_table is not None:
                self.tables.append(self.current_table)
                self.current_table = None
            self.table_depth -= 1


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def country_page_urls() -> dict[str, str]:
    parser = LinkParser()
    parser.feed(fetch(MAIN_URL).decode("utf-8", errors="replace"))
    normalized_links = [(normalize(text), urllib.parse.urljoin(MAIN_URL, href)) for text, href in parser.links]
    output: dict[str, str] = {}
    for code, (_, label) in COUNTRIES.items():
        if label is None or code == "can":
            continue
        target = normalize(label)
        matches = [(text, href) for text, href in normalized_links if target in text]
        if not matches:
            raise RuntimeError(f"No country page found for {code}: {label}")
        output[code] = matches[0][1]
    return output


def exchange_candidates(code: str, url: str) -> list[dict[str, str]]:
    parser = TableParser()
    parser.feed(fetch(url).decode("utf-8", errors="replace"))
    selected: list[list[str]] | None = None
    for table in parser.tables:
        if not table:
            continue
        header = [normalize(cell) for cell in table[0]]
        if "ticker" in header and "exchange" in header:
            selected = table
            break
    if selected is None:
        raise RuntimeError(f"No exchange table found for {code}: {url}")

    header = [normalize(cell) for cell in selected[0]]
    records: list[dict[str, str]] = []
    for row in selected[1:]:
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        record = dict(zip(header, row))
        ticker = record.get("ticker", "").strip().upper()
        exchange = record.get("exchange", "").strip().upper()
        if ticker and exchange in {"NYSE", "NASDAQ", "AMEX", "NYSE AMERICAN"}:
            records.append(
                {
                    "country_code": code,
                    "country": COUNTRIES[code][0],
                    "company": record.get("company name", ""),
                    "ticker": ticker,
                    "exchange_source": exchange,
                    "industry_source": record.get("industry", ""),
                    "candidate_source": url,
                }
            )
    return records


def nasdaq_rows() -> dict[str, dict[str, str]]:
    payload = json.loads(fetch(NASDAQ_URL).decode("utf-8"))
    rows = payload["data"]["rows"]
    return {row["symbol"].upper(): row for row in rows}


def yahoo_has_july_2016(ticker: str) -> tuple[bool, str, int, str]:
    quoted = urllib.parse.quote(ticker.replace(".", "-"), safe="-")
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{quoted}"
        f"?period1={CUTOFF_START}&period2={CUTOFF_END}&interval=1d"
        "&events=history&includeAdjustedClose=true"
    )
    try:
        payload = json.loads(fetch(url, attempts=2, timeout=12).decode("utf-8"))
        result = payload.get("chart", {}).get("result")
        error = payload.get("chart", {}).get("error")
        if not result:
            return False, "", 0, json.dumps(error, ensure_ascii=False)
        timestamps = result[0].get("timestamp") or []
        if not timestamps:
            return False, "", 0, "no July 2016 timestamps"
        first = dt.datetime.fromtimestamp(min(timestamps), tz=dt.timezone.utc).date().isoformat()
        return True, first, len(timestamps), ""
    except Exception as exc:
        return False, "", 0, f"{type(exc).__name__}: {exc}"


def main() -> None:
    urls = country_page_urls()
    candidates: list[dict[str, str]] = []
    for code, url in urls.items():
        candidates.extend(exchange_candidates(code, url))

    listing_rows = nasdaq_rows()
    for row in candidates:
        listing = listing_rows.get(row["ticker"])
        row["currently_listed"] = "yes" if listing else "no"
        row["nasdaq_name"] = listing.get("name", "") if listing else ""
        row["nasdaq_country"] = listing.get("country", "") if listing else ""
        row["ipo_year"] = listing.get("ipoyear", "") if listing else ""
        row["sector"] = listing.get("sector", "") if listing else ""
        row["industry"] = listing.get("industry", "") if listing else ""

    current_candidates = [row for row in candidates if row["currently_listed"] == "yes"]
    history_candidates = [
        row
        for row in current_candidates
        if not row["ipo_year"].isdigit() or int(row["ipo_year"]) <= 2016
    ]
    print(
        f"Checking July 2016 history for {len(history_candidates)} of "
        f"{len(current_candidates)} currently listed candidates..."
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(yahoo_has_july_2016, row["ticker"]): row
            for row in history_candidates
        }
        for future in as_completed(futures):
            row = futures[future]
            has_history, first_date, observations, error = future.result()
            row["july_2016_data"] = "yes" if has_history else "no"
            row["july_2016_first_date"] = first_date
            row["july_2016_observations"] = str(observations)
            row["history_check_error"] = error

    for row in candidates:
        row.setdefault("july_2016_data", "no")
        row.setdefault("july_2016_first_date", "")
        row.setdefault("july_2016_observations", "0")
        row.setdefault(
            "history_check_error",
            "not currently listed"
            if row["currently_listed"] == "no"
            else "IPO year after 2016",
        )

    output_dir = Path("results") / "adr_coverage"
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "country_code",
        "country",
        "company",
        "ticker",
        "exchange_source",
        "industry_source",
        "currently_listed",
        "july_2016_data",
        "july_2016_first_date",
        "july_2016_observations",
        "ipo_year",
        "nasdaq_name",
        "nasdaq_country",
        "sector",
        "industry",
        "history_check_error",
        "candidate_source",
    ]
    with (output_dir / "adr_candidates.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(candidates, key=lambda row: (row["country"], row["ticker"])))

    summary_rows: list[dict[str, str | int]] = []
    for code, (country, _) in COUNTRIES.items():
        rows = [row for row in candidates if row["country_code"] == code]
        summary_rows.append(
            {
                "country_code": code,
                "country": country,
                "exchange_candidates": len(rows),
                "currently_listed": sum(row["currently_listed"] == "yes" for row in rows),
                "with_july_2016_data": sum(row["july_2016_data"] == "yes" for row in rows),
                "note": (
                    "US domestic stocks, not ADRs"
                    if code == "usa"
                    else "Canadian interlisted stocks, not strict ADRs"
                    if code == "can"
                    else "No country page / no exchange-listed ADR candidates"
                    if code == "sau"
                    else ""
                ),
            }
        )
    with (output_dir / "adr_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    for row in summary_rows:
        print(
            f"{row['country']:<16} candidates={row['exchange_candidates']:>3} "
            f"current={row['currently_listed']:>3} "
            f"10y~={row['with_july_2016_data']:>3} {row['note']}"
        )


if __name__ == "__main__":
    main()
