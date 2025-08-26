#!/usr/bin/env python3
"""
Scrape Minnesota ZIP codes from unitedstateszipcodes.org (panel-prefixes layout)
with Wayback fallback if the live site blocks us (403).
Writes results to a CSV.

Requires:
    pip install requests beautifulsoup4
"""

import csv
import sys
import requests
from bs4 import BeautifulSoup

LIVE_URL = "https://www.unitedstateszipcodes.org/mn/"
WAYBACK_URL = "https://web.archive.org/web/https://www.unitedstateszipcodes.org/mn/"
OUT_CSV = "mn_zip_codes.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.unitedstateszipcodes.org/",
    "Connection": "keep-alive",
}


def get_soup(url: str) -> BeautifulSoup:
    """Fetch URL and return parsed BeautifulSoup object."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code == 403:
        raise requests.HTTPError("403 Forbidden", response=resp)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def find_zip_panel(soup: BeautifulSoup):
    """Locate the main ZIP code panel."""
    panel = soup.select_one("div.panel-prefixes") or \
            soup.select_one("div.panel.panel-default.panel-prefixes")
    return panel


def extract_panel_prefixes(panel: BeautifulSoup):
    """Extract headers and rows from panel-prefixes layout."""
    header_nodes = panel.select(".panel-heading .row > [class*=prefix-col]")
    headers = [h.get_text(strip=True) for h in header_nodes]
    if not headers:
        headers = ["ZIP Code", "Type", "Common Cities", "County", "Area Codes"]

    rows = []
    for item in panel.select(".list-group .list-group-item"):
        row_container = item.select_one(".row")
        if not row_container:
            continue
        cols = []
        for i in range(1, len(headers) + 1):
            cell = row_container.select_one(f".prefix-col{i}")
            text = cell.get_text(" ", strip=True) if cell else ""
            cols.append(text)
        rows.append(cols)

    return headers, rows


def write_csv(headers, rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def main():
    try:
        # Try live site
        soup = get_soup(LIVE_URL)
    except requests.HTTPError as e:
        if getattr(e, "response", None) is not None and e.response.status_code == 403:
            print("Got 403 Forbidden. Falling back to Wayback Machine snapshot...", file=sys.stderr)
            soup = get_soup(WAYBACK_URL)
        else:
            raise
    except requests.RequestException as e:
        print(f"Network error: {e}", file=sys.stderr)
        sys.exit(2)

    panel = find_zip_panel(soup)
    if not panel:
        print("Could not find ZIP code panel.", file=sys.stderr)
        sys.exit(1)

    headers, rows = extract_panel_prefixes(panel)
    if not rows:
        print("Found panel, but no rows extracted.", file=sys.stderr)
        sys.exit(1)

    write_csv(headers, rows, OUT_CSV)
    print(f"Saved {len(rows)} ZIP codes to {OUT_CSV}")


if __name__ == "__main__":
    main()
