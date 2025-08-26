#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scraper.py  —  Target (RedSky) store + product collector for Minnesota

What it does
------------
1) Reads `mn_zip_codes.csv` (MN-only ZCTAs, preserves all columns).
2) For each ZIP, queries RedSky `nearby_stores_v1` to discover stores near that ZIP.
   - Dedupes stores on-the-fly by `store_id` so later phases don’t reprocess duplicates.
   - Appends each *new* store JSON to data/store_raw.jsonl.
3) After discovery, flattens stores → data/store.csv (dash-separated column names).
4) Reads `category.csv` (see example schema below) and for each (store × category):
   - Uses RedSky PLP search to enumerate TCINs (paginated via offset/limit).
   - Fetches PDP JSON (`pdp_client_v1`) per TCIN (per store) and appends to data/product_raw.jsonl.
   - Skips any (store_id, tcin) pair that already exists in product_raw.jsonl (resume-safe).
5) Flattens product JSON → data/product.csv (dash-separated column names).

HTTP modes
----------
• Default: pure-HTTP using `httpx` (HTTP/2, retry with jitter/backoff).
• Optional: real browser fetches with Selenium (Chrome) when `--use-selenium` is set.
  - Opens a Chrome session, sets only the User-Agent via CDP, and performs in-page `fetch()`.
  - No custom request headers are forced; `credentials: "omit"` is used to avoid CORS pitfalls.
  - `ensure_context()` briefly navigates to a matching Target page to seed cookies/referer.

Pagination limits
-----------------
• RedSky PLP enforces `offset <= 1199`. This script respects the cap:
  - Requests clamp `count` so the effective offset never exceeds 1199.
  - A 400 “offset” error is treated as end-of-pagination for that slice.
• To collect more than 1200 results from a huge category, shard it (e.g., by leaf categories
  or facet slices like brand/price) and union TCINs. (Sharding strategy not implemented here.)

Notes & assumptions
-------------------
• API KEY & visitor ID:
  - Override via env vars:
      TARGET_API_KEY         (default: a known public web key)
      TARGET_VISITOR_ID      (default: stable UUID-like string)
  - You can also override the PLP endpoint via env:
      TARGET_PLP_ENDPOINT    (default: tries v2, then v1, then legacy automatically)
  - If Target changes their APIs, adjust the endpoints/params near `RedSkyClient.plp_search()`.

• Concurrency: default is 4 (configurable via `--concurrency`). Jitter and exponential backoff included.

• Logging: human-readable INFO logs. Progress is resumable by inspecting existing JSONL files.

• Python: 3.12.3

Dependencies (install)
----------------------
Required:
    pip install httpx pandas tenacity python-dateutil
Optional (for `--use-selenium`):
    pip install selenium webdriver-manager

Input files (expected)
----------------------
mn_zip_codes.csv   # columns: ZIP Code,Type,Common Cities,County,Area Codes  (preserve all)
category.csv       # columns: name,slug,url_path,full_url,n_code,data_id,number_of_children

Output files (in ./data)
------------------------
data/store_raw.jsonl   # one raw JSON object per discovered store (first time seen)
data/store.csv         # flattened, dash-named columns
data/product_raw.jsonl # one raw JSON object per (store_id × tcin)
data/product.csv       # flattened, dash-named columns

Run examples
------------
# Everything: discover stores, export store.csv, then fetch products, export product.csv
python scraper.py --mn-zip-csv mn_zip_codes.csv --category-csv category.csv --phase all

# Only store discovery + export
python scraper.py --mn-zip-csv mn_zip_codes.csv --phase stores

# Only product phase (uses existing store_raw.jsonl and category.csv)
python scraper.py --category-csv category.csv --phase products

# Use a real browser for PLP/PDP (helpful if the site tightens checks)
python scraper.py --category-csv category.csv --phase products --use-selenium

"""
from __future__ import annotations

import os
import re
import csv
import sys
import json
import time
import math
import uuid
import asyncio
import random
import logging
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type
from dateutil import tz

from urllib.parse import urlencode
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# ----------------------------- Configuration -------------------------------- #

DEFAULT_API_KEY = os.environ.get("TARGET_API_KEY", "9f36aeafbe60771e321a7cc95a78140772ab3e96")
DEFAULT_VISITOR_ID = os.environ.get("TARGET_VISITOR_ID", "0198DD0F37E0020184A9726303478665")
DEFAULT_CHANNEL = "WEB"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:115.0) Gecko/20100101 Firefox/115.0"
)



# PLP endpoint override (if needed)
ENV_PLP_ENDPOINT = os.environ.get("TARGET_PLP_ENDPOINT")  # if not provided, we auto-try v1/v2/legacy

DATA_DIR = Path("data")
STORE_RAW_PATH = DATA_DIR / "store_raw.jsonl"
STORE_CSV_PATH = DATA_DIR / "store.csv"
PRODUCT_RAW_PATH = DATA_DIR / "product_raw.jsonl"
PRODUCT_CSV_PATH = DATA_DIR / "product.csv"

# PLP pagination defaults
PLP_PAGE_SIZE = 24  # common page size; we'll try `count` and `limit` param names
MAX_PLP_OFFSET = 1199   # RedSky rejects offset > 1199

# Concurrency defaults
DEFAULT_CONCURRENCY = 4

# ------------------------------- Logging ------------------------------------ #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scraper")


# --------------------------- Utility: JSON Flatten --------------------------- #

def _normalize_key_part(k: str) -> str:
    """
    Keep the original key text but ensure it doesn't introduce dots; we use '-' as the joiner.
    Replace any whitespace with underscores to avoid accidental CSV column parsing quirks.
    """
    if not isinstance(k, str):
        k = str(k)
    k = k.replace(".", "-")
    k = re.sub(r"\s+", "_", k)
    return k


def flatten_dict_dash(obj: Any, parent_key: str = "", sep: str = "-") -> Dict[str, Any]:
    """
    Flattens nested dicts into dash-separated keys.
    Lists are JSON-encoded strings to preserve "comprehensive" structure.
    Scalars are passed through unchanged.

    Example:
      {"a": {"b": 1}, "c": [2, 3]}  ->  {"a-b": 1, "c": "[2,3]"}
    """
    items: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = _normalize_key_part(parent_key + sep + k if parent_key else k)
            if isinstance(v, dict):
                items.update(flatten_dict_dash(v, new_key, sep=sep))
            elif isinstance(v, list):
                # Preserve list content comprehensively as minified JSON text
                try:
                    items[new_key] = json.dumps(v, separators=(",", ":"), ensure_ascii=False)
                except Exception:
                    # Fallback string repr if something cannot be serialized
                    items[new_key] = str(v)
            else:
                items[new_key] = v
    elif isinstance(obj, list):
        # If a top-level list is passed, represent as JSON string
        try:
            items[parent_key or "list"] = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
        except Exception:
            items[parent_key or "list"] = str(obj)
    else:
        items[parent_key or "value"] = obj
    return items


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line in %s", path)
    return rows


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ----------------------------- HTTP Client ---------------------------------- #

# ----- SELENIUM CLIENT ----- #
class BrowserFetcher:
    def __init__(self, start_url, headless=True):
        self.start_url = start_url
        self.headless = headless
        self.driver = None

    def start(self, url_path=None, category_data_id=None):
        opts = Options()
        if self.headless:
            opts.add_argument("--headless=new")
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.add_experimental_option("excludeSwitches", ["enable-automation"])
        opts.add_experimental_option("useAutomationExtension", False)

        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=opts
        )
        self.driver.set_page_load_timeout(30)

        # Hide webdriver flag
        self.driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"}
        )

        # Enable network and set ONLY the user agent. Do NOT set extra headers.
        self.driver.execute_cdp_cmd("Network.enable", {})
        self.driver.execute_cdp_cmd(
            "Network.setUserAgentOverride",
            {"userAgent": DEFAULT_USER_AGENT}
        )

        # Load initial page
        self.driver.get(self.start_url)

    def stop(self):
        try:
            if self.driver:
                self.driver.quit()
        finally:
            self.driver = None

    def ensure_context(self, url_path: str, wait: float = 0.2):
        if not url_path:
            return
        target_url = url_path if url_path.startswith("http") else "https://www.target.com" + url_path
        if self.driver.current_url != target_url:
            self.driver.get(target_url)
            time.sleep(wait)

    def fetch_json(self, url: str, accept: str = "application/json") -> dict:
        if not self.driver:
            raise RuntimeError("Browser not started")
        script = """
    const url = arguments[0];
    const accept = arguments[1];
    const done = arguments[2];

    fetch(url, {
    method: 'GET',
    mode: 'cors',
    credentials: 'omit',          // avoid credentialed CORS constraints
    headers: { 'Accept': accept } // simple header => no preflight
    }).then(async (res) => {
    const ct = res.headers.get('content-type') || '';
    const text = await res.text();
    if (!res.ok) return done({ ok: false, status: res.status, body: text });
    if (ct.includes('application/json')) {
        try { return done({ ok: true, json: JSON.parse(text) }); }
        catch (_) { return done({ ok: false, status: res.status, body: text }); }
    }
    return done({ ok: true, text });
    }).catch((e) => done({ ok: false, error: String(e) }));
    """
        result = self.driver.execute_async_script(script, url, accept)
        if not result or not result.get("ok"):
            raise RedSkyError(f"Browser fetch failed: {result}")
        return result.get("json") or {"text": result.get("text")}

class RedSkyError(Exception):
    pass


class RedSkyClient:
    def __init__(self, api_key: str = DEFAULT_API_KEY, visitor_id: str = DEFAULT_VISITOR_ID,
                 channel: str = DEFAULT_CHANNEL, concurrency: int = DEFAULT_CONCURRENCY,
                 timeout: float = 20.0, browser: "BrowserFetcher|None" = None) -> None:
        self.api_key = api_key
        self.visitor_id = visitor_id
        self.channel = channel
        self.semaphore = asyncio.Semaphore(concurrency)
        self.browser = browser
        self.browser_lock = asyncio.Lock() if browser else None

        self.client = httpx.AsyncClient(
            headers = {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Encoding": "gzip, deflate, br",
                "Accept-Language": "en-US,en;q=0.5",
                "Connection": "keep-alive",
                "Host": "redsky.target.com",
                "Origin": "https://www.target.com",
                "Referer": "https://www.target.com/",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "TE": "trailers",
                "Upgrade-Insecure-Requests": "1",
                "User-Agent": DEFAULT_USER_AGENT,
            },
            timeout=httpx.Timeout(timeout),
            http2=True,
            limits=httpx.Limits(max_keepalive_connections=concurrency, max_connections=concurrency),
        )

    async def aclose(self):
        await self.client.aclose()

    async def _get(self, url: str, params: dict) -> httpx.Response:
        # Shared GET with concurrency, jitter, and retry
        async with self.semaphore:

            @retry(
                reraise=True,
                stop=stop_after_attempt(6),
                wait=wait_exponential_jitter(initial=0.5, max=10.0),
                retry=retry_if_exception_type((httpx.HTTPError, RedSkyError)),
            )
            async def _attempt() -> httpx.Response:
                # Tiny jitter to avoid thundering herd
                await asyncio.sleep(random.uniform(0.05, 0.25))
                try:
                    resp = await self.client.get(url, params=params)
                except httpx.HTTPError as e:
                    raise e
                # RedSky occasionally uses 200 with error payloads; we also handle HTTP errors
                if resp.status_code == 429:
                    # Force retry with backoff
                    raise RedSkyError("HTTP 429 Too Many Requests")
                if resp.status_code >= 500:
                    raise RedSkyError(f"HTTP {resp.status_code} server error")
                return resp

            return await _attempt()

    async def nearby_stores(self, place_zip: str, within_miles: int = 100, limit: int = 20) -> dict:
        """
        GET redsky_aggregations/v1/web/nearby_stores_v1
        """
        url = "https://redsky.target.com/redsky_aggregations/v1/web/nearby_stores_v1"
        params = {
            "limit": str(limit),
            "within": str(within_miles),
            "place": place_zip,
            "key": self.api_key,
            "visitor_id": self.visitor_id,
            "channel": self.channel
        }
        resp = await self._get(url, params)
        return resp.json()


    async def plp_search_browser(
        self,
        store_id: str,
        category_data_id: str,
        url_path: str,
        offset: int,
        count: int
    ) -> dict:
        if not self.browser or not self.browser_lock:
            raise RedSkyError("Browser not attached")

        params = {
            "key": self.api_key,
            "visitor_id": self.visitor_id,
            "channel": "WEB",
            "category": category_data_id,
            "count": str(count or 24),
            "offset": str(offset),
            "default_purchasability_filter": "true",
            "include_dmc_dmr": "true",
            "include_sponsored": "true",
            "include_review_summarization": "true",
            "new_search": "false",
            "page": url_path,
            "platform": "desktop",
            "pricing_store_id": store_id,
            "scheduled_delivery_store_id": store_id,
            "store_ids": store_id,
            "useragent": DEFAULT_USER_AGENT,
            "spellcheck": "true",
        }

        url = (
            "https://redsky.target.com/redsky_aggregations/v1/web/plp_search_v2?"
            + urlencode(params)
        )

        async with self.browser_lock:
            # Ensure the browser context is aligned (cookies, referer, etc.)
            await asyncio.to_thread(
                self.browser.ensure_context,
                url_path or "/"
            )
            return await asyncio.to_thread(
                self.browser.fetch_json,
                url,
                "application/json"
            )
        
    async def pdp_browser(self, tcin: str, store_id: str) -> dict:
        if not self.browser or not self.browser_lock:
            raise RedSkyError("Browser not attached")
        params = {
            "key": self.api_key,
            "tcin": tcin,
            "is_bot": "false",
            "store_id": store_id,
            "pricing_store_id": store_id,
            "has_pricing_options": "true",
            "include_obsolete": "true",
            "visitor_id": self.visitor_id,
            "skip_personalized": "true",
            "skip_variation_hierarchy": "true",
            "channel": self.channel,
            "page": f"/p/A-{tcin}",
        }
        url = "https://redsky.target.com/redsky_aggregations/v1/web/pdp_client_v1?" + urlencode(params)
        async with self.browser_lock:
            # keep referrer on some PDP-ish page for consistency; not strictly required
            await asyncio.to_thread(self.browser.ensure_context, f"/p/A-{tcin}")
            return await asyncio.to_thread(self.browser.fetch_json, url, "application/json")

    async def plp_search(
        self,
        store_id: str,
        category_data_id: str,
        url_path: str,
        offset: int,
        count: int,
    ) -> dict:
        if self.browser:
            return await self.plp_search_browser(store_id, category_data_id, url_path, offset, count)

        endpoints: List[str]
        if ENV_PLP_ENDPOINT:
            endpoints = [ENV_PLP_ENDPOINT.strip()]
        else:
            endpoints = [
                "https://redsky.target.com/redsky_aggregations/v1/web/plp_search_v2",
                "https://redsky.target.com/redsky_aggregations/v1/web/plp_search_v1",
                "https://redsky.target.com/redsky_aggregations/v1/web/plp_search",
            ]

        count_keys = ["count", "limit"]
        category_keys = ["category", "node"]

        last_error: Optional[Exception] = None
        for ep in endpoints:
            for ck in count_keys:
                for nk in category_keys:
                    params = {
                        "key": self.api_key,
                        "visitor_id": self.visitor_id,
                        "channel": "WEB",
                        nk: category_data_id,                # <-- use nk
                        ck: str(count or 24),                # <-- use ck
                        "offset": str(offset),
                        "default_purchasability_filter": "true",
                        "include_dmc_dmr": "true",
                        "include_sponsored": "true",
                        "include_review_summarization": "true",
                        "new_search": "false",
                        "page": url_path or f"/c/{category_data_id}",
                        "platform": "desktop",
                        "pricing_store_id": store_id,
                        "scheduled_delivery_store_id": store_id,
                        "store_ids": store_id,
                        "useragent": DEFAULT_USER_AGENT,
                        "spellcheck": "true",
                    }
                    try:
                        resp = await self._get(ep, params)
                        data = resp.json()
                        tcins = extract_tcins_from_plp_json(data)
                        return data if tcins else data
                    except Exception as e:
                        last_error = e
                        continue
        if last_error:
            raise last_error
        raise RedSkyError("PLP search failed with unknown error")

    async def pdp(self, tcin: str, store_id: str) -> dict:
        """
        GET pdp_client_v1 for a TCIN scoped to a store (so we capture store-specific pricing/availability).
        """
        if self.browser:
            return await self.pdp_browser(tcin, store_id)
        
        url = "https://redsky.target.com/redsky_aggregations/v1/web/pdp_client_v1"
        params = {
            "key": self.api_key,
            "tcin": tcin,
            "is_bot": "false",
            "store_id": store_id,
            "pricing_store_id": store_id,
            "has_pricing_store_id": "true",
            "has_financing_options": "true",
            "include_obsolete": "true",
            "visitor_id": self.visitor_id,
            "skip_personalized": "true",
            "skip_variation_hierarchy": "true",
            "channel": self.channel,
            "page": f"/p/A-{tcin}",
        }
        resp = await self._get(url, params)
        return resp.json()


# ----------------------------- Data Extractors ------------------------------- #

def extract_stores_from_nearby(payload: dict) -> List[dict]:
    """
    Extracts store list safely from nearby_stores_v1 payload.
    """
    try:
        return payload["data"]["nearby_stores"]["stores"] or []
    except Exception:
        return []


def extract_tcins_from_plp_json(payload: dict) -> List[str]:
    """
    Very tolerant extractor: walk the dict and collect every value under a 'tcin' key.
    """
    tcins: Set[str] = set()
    stack: List[Any] = [payload]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                if k == "tcin" and isinstance(v, (str, int)):
                    tcins.add(str(v))
                elif isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(cur, list):
            stack.extend(cur)
    return list(tcins)


def now_iso_local() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


# ----------------------------- Store Discovery ------------------------------- #

async def discover_mn_stores(
    red: RedSkyClient,
    mn_zip_csv: Path,
    max_zip_codes: int,
    out_jsonl: Path = STORE_RAW_PATH,
    dedupe_on_store_id: bool = True
) -> List[dict]:
    """
    For each ZIP in mn_zip_codes.csv, fetch up to 20 nearby stores within 100 miles and write
    NEW stores to store_raw.jsonl. Returns the list of unique store dicts discovered
    (including those that were already present in the JSONL).
    """
    # load previous stores (for resume/dedupe)
    existing_store_rows = read_jsonl(out_jsonl)
    seen_store_ids: Set[str] = set()
    for row in existing_store_rows:
        sid = str(row.get("store_id") or row.get("store", {}).get("store_id") or "")
        if sid:
            seen_store_ids.add(sid)

    # read MN zips
    if not mn_zip_csv.exists():
        logger.error("ZIP CSV not found: %s", mn_zip_csv)
        sys.exit(1)

    with mn_zip_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        mn_zip_rows = list(reader)

    logger.info("Loaded %d MN ZIP rows", len(mn_zip_rows))

    if max_zip_codes:
        try:
            max_zip_codes = int(max_zip_codes)
        except Exception as e:
            logger.error("Argument `--max_zip_codes` must be an integer. The value %d is a %d", (type(max_zip_codes), max_zip_codes))
        mn_zip_rows = mn_zip_rows[:max_zip_codes]
        logger.info("Reduced ZIP count to %d for testing.", len(mn_zip_rows))

    # helper to process one ZIP
    async def process_zip(zip_row: dict) -> Tuple[str, int]:
        zip_code = str(zip_row.get("ZIP Code") or "").strip()
        if not zip_code:
            return ("", 0)
        try:
            payload = await red.nearby_stores(zip_code)
            stores = extract_stores_from_nearby(payload)
            new_count = 0
            for s in stores:
                sid = str(s.get("store_id") or "").strip()
                if not sid:
                    continue
                if dedupe_on_store_id and sid in seen_store_ids:
                    continue
                # enrich with discovery context
                enriched = dict(s)
                enriched["_discovery_zip"] = zip_code
                enriched["_discovered_at"] = now_iso_local()
                append_jsonl(out_jsonl, enriched)
                seen_store_ids.add(sid)
                new_count += 1
            if new_count:
                logger.info("ZIP %s → %d new stores (unique so far: %d)", zip_code, new_count, len(seen_store_ids))
            return (zip_code, new_count)
        except Exception as e:
            logger.warning("ZIP %s failed: %s", zip_code, e)
            return (zip_code, 0)

    # run with concurrency
    tasks = [asyncio.create_task(process_zip(row)) for row in mn_zip_rows]
    # Stream completions
    completed = 0
    for coro in asyncio.as_completed(tasks):
        _ = await coro
        completed += 1
        if completed % 50 == 0:
            logger.info("Processed %d/%d ZIPs", completed, len(tasks))

    # Return FINAL set of unique stores (merge already-present + new)
    final_store_rows = read_jsonl(out_jsonl)
    logger.info("Total unique stores in %s: %d", out_jsonl, len({str(r.get('store_id')) for r in final_store_rows}))
    return final_store_rows


def export_store_csv(in_jsonl: Path = STORE_RAW_PATH, out_csv: Path = STORE_CSV_PATH) -> None:
    """
    Flatten store_raw.jsonl → store.csv using dash-separated columns.
    """
    rows = read_jsonl(in_jsonl)
    if not rows:
        logger.warning("No store rows found in %s; skipping store.csv export.", in_jsonl)
        return

    flat_records: List[Dict[str, Any]] = []
    for r in rows:
        flat_records.append(flatten_dict_dash(r))

    df = pd.DataFrame(flat_records)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info("Wrote %s with %d rows and %d columns", out_csv, df.shape[0], df.shape[1])


# ----------------------------- Product Phase -------------------------------- #

def load_categories(category_csv: Path) -> List[dict]:
    if not category_csv.exists():
        logger.error("Category CSV not found: %s", category_csv)
        sys.exit(1)
    with category_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cats = []
        for row in reader:
            # Expecting columns: name,slug,url_path,full_url,n_code,data_id,number_of_children
            # We'll use url_path and data_id (e.g., "5xt4g")
            if not row.get("data_id"):
                # Sometimes data_id missing? Try 'n_code' like "N-5xt4g" -> make "5xt4g"
                n_code = row.get("n_code", "")
                m = re.search(r"N-([0-9a-z]+)", n_code or "", flags=re.I)
                row["data_id"] = m.group(1) if m else ""
            cats.append(row)
    logger.info("Loaded %d categories", len(cats))
    return cats


def load_unique_store_ids(from_jsonl: Path = STORE_RAW_PATH) -> List[str]:
    rows = read_jsonl(from_jsonl)
    store_ids: List[str] = []
    seen: Set[str] = set()
    for r in rows:
        sid = str(r.get("store_id") or "").strip()
        if sid and sid not in seen:
            seen.add(sid)
            store_ids.append(sid)
    logger.info("Unique stores available for product phase: %d", len(store_ids))
    return store_ids


def load_seen_store_tcin_pairs(product_jsonl: Path = PRODUCT_RAW_PATH) -> Set[Tuple[str, str]]:
    pairs: Set[Tuple[str, str]] = set()
    if not product_jsonl.exists():
        return pairs
    with product_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                sid = str(row.get("_store_id") or row.get("store_id") or "").strip()
                tcin = str(
                    row.get("_tcin")
                    or row.get("tcin")
                    or row.get("data", {}).get("product", {}).get("tcin")
                    or ""
                ).strip()
                if sid and tcin:
                    pairs.add((sid, tcin))
            except Exception:
                continue
    logger.info("Found %d existing (store_id, tcin) pairs in %s", len(pairs), product_jsonl)
    return pairs


async def enumerate_tcins_for_category(
    red: RedSkyClient, store_id: str, cat: dict, max_per_category: Optional[int] = None
) -> List[str]:
    category_id = (cat.get("data_id") or "").strip()
    url_path = (cat.get("url_path") or "/").strip()
    if not category_id:
        return []

    collected: List[str] = []
    seen: Set[str] = set()

    offset = 0
    page_size = PLP_PAGE_SIZE

    while True:
        # stop if we’ve reached the API’s offset cap
        if offset >= MAX_PLP_OFFSET:
            logger.info("store=%s cat=%s offset cap reached (%d) — stopping",
                        store_id, category_id, MAX_PLP_OFFSET)
            break

        # clamp count so offset never exceeds the cap on this request
        safe_count = min(page_size, MAX_PLP_OFFSET - offset + 1)

        try:
            payload = await red.plp_search(store_id, category_id, url_path, offset, safe_count)
        except RedSkyError as e:
            msg = str(e)
            # treat the offset 400 as end-of-pagination, not a hard failure
            if ("status" in msg and "400" in msg) and "offset" in msg:
                logger.info("store=%s cat=%s hit offset 400 at offset=%d — stopping",
                            store_id, category_id, offset)
                break
            # keep the previous behavior for other errors
            if "404" in msg:
                logger.info("store=%s cat=%s reached 404 at offset=%d — stopping",
                            store_id, category_id, offset)
                break
            raise

        tcins = extract_tcins_from_plp_json(payload)
        new_tcins = [t for t in tcins if t not in seen]
        if not new_tcins:
            break

        collected.extend(new_tcins)
        seen.update(new_tcins)

        logger.info("store=%s cat=%s offset=%d → +%d tcins (total %d)",
                    store_id, category_id, offset, len(new_tcins), len(collected))

        if max_per_category and len(collected) >= max_per_category:
            collected = collected[:max_per_category]
            break

        # advance by the amount we actually requested
        offset += safe_count
        await asyncio.sleep(random.uniform(0.05, 0.2))

    return collected


async def collect_products_for_store(
    red: RedSkyClient,
    store_id: str,
    categories: List[dict],
    seen_pairs: Set[Tuple[str, str]],
    out_jsonl: Path = PRODUCT_RAW_PATH,
    max_per_category: Optional[int] = None,
) -> Tuple[str, int]:
    """
    For one store, iterate categories, enumerate TCINs via PLP, then fetch PDP JSON per TCIN
    (skipping pairs that already exist in product_raw.jsonl).
    Returns (store_id, new_count).
    """
    new_count = 0

    for cat in categories:
        tcins = await enumerate_tcins_for_category(red, store_id, cat, max_per_category=max_per_category)
        if not tcins:
            continue

        # Fetch PDPs with concurrency (bounded by RedSkyClient semaphore)
        async def fetch_one(tcin: str) -> Optional[int]:
            pair = (store_id, tcin)
            if pair in seen_pairs:
                return 0
            try:
                payload = await red.pdp(tcin, store_id)
                # enrich with explicit pointers; some PDP payloads already contain tcin/store_id, but we ensure it.
                enriched = {
                    "_store_id": store_id,
                    "_tcin": tcin,
                    "_fetched_at": now_iso_local(),
                    **payload,
                }
                append_jsonl(out_jsonl, enriched)
                seen_pairs.add(pair)
                return 1
            except Exception as e:
                logger.warning("PDP failed store=%s tcin=%s: %s", store_id, tcin, e)
                return 0

        tasks = [asyncio.create_task(fetch_one(t)) for t in tcins]
        for coro in asyncio.as_completed(tasks):
            added = await coro
            if added:
                new_count += added
        # If using Selenium, tiny pause between batches helps avoid challenges
        if red.browser:
            await asyncio.sleep(0.3)


    if new_count:
        logger.info("store=%s → %d new product rows", store_id, new_count)
    return (store_id, new_count)


def export_product_csv(in_jsonl: Path = PRODUCT_RAW_PATH, out_csv: Path = PRODUCT_CSV_PATH) -> None:
    """
    Flatten product_raw.jsonl → product.csv using dash-separated columns.
    """
    rows = read_jsonl(in_jsonl)
    if not rows:
        logger.warning("No product rows found in %s; skipping product.csv export.", in_jsonl)
        return

    flat_records: List[Dict[str, Any]] = []
    for r in rows:
        flat_records.append(flatten_dict_dash(r))

    df = pd.DataFrame(flat_records)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info("Wrote %s with %d rows and %d columns", out_csv, df.shape[0], df.shape[1])


# ----------------------------- CLI Orchestration ----------------------------- #

async def phase_stores(red: RedSkyClient, args: argparse.Namespace) -> None:
    await discover_mn_stores(
        red=red,
        mn_zip_csv=Path(args.mn_zip_csv),
        max_zip_codes=args.max_zip_codes,
        out_jsonl=STORE_RAW_PATH,
        dedupe_on_store_id=True
    )
    export_store_csv(STORE_RAW_PATH, STORE_CSV_PATH)


async def phase_products(red: RedSkyClient, args: argparse.Namespace) -> None:
    store_ids = load_unique_store_ids(STORE_RAW_PATH)
    if not store_ids:
        logger.error("No stores found. Run the 'stores' phase first (or provide store_raw.jsonl).")
        sys.exit(2)

    store_ids = store_ids[:1]  # testing one store

    categories = load_categories(Path(args.category_csv))
    seen_pairs = load_seen_store_tcin_pairs(PRODUCT_RAW_PATH)

    total_new = 0

    async def do_store(sid: str) -> Tuple[str, int]:
        return await collect_products_for_store(
            red=red,
            store_id=sid,
            categories=categories,
            seen_pairs=seen_pairs,
            out_jsonl=PRODUCT_RAW_PATH,
            max_per_category=args.max_per_category,
        )

    tasks = [asyncio.create_task(do_store(sid)) for sid in store_ids]
    completed = 0
    for coro in asyncio.as_completed(tasks):
        sid, added = await coro
        total_new += added
        completed += 1
        logger.info("Completed store %s (%d/%d). Newly added so far: %d",
                    sid, completed, len(tasks), total_new)

    export_product_csv(PRODUCT_RAW_PATH, PRODUCT_CSV_PATH)


async def main_async(args: argparse.Namespace) -> None:
    browser = None
    if args.use_selenium:
        browser = BrowserFetcher(headless=not args.no_headless,
                                 start_url="https://www.target.com/")
        browser.start()
        # # Optional: preload a category page to seed category-scoped cookies
        # try:
        #     browser.driver.get("https://www.target.com/c/arts-crafts-sewing-home/-/N-5xt4g")
        #     time.sleep(2.0)
        # except Exception:
        #     pass

    red = RedSkyClient(
        api_key=DEFAULT_API_KEY,
        visitor_id=DEFAULT_VISITOR_ID,
        channel=DEFAULT_CHANNEL,
        concurrency=args.concurrency or DEFAULT_CONCURRENCY,
        browser=browser,
    )
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if args.phase in ("stores", "all"):
            logger.info("=== Phase: STORES ===")
            await phase_stores(red, args)
        if args.phase in ("products", "all"):
            logger.info("=== Phase: PRODUCTS ===")
            await phase_products(red, args)
        logger.info("All done.")
    finally:
        await red.aclose()
        if browser:
            browser.stop()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Target (RedSky) MN scraper — stores & products")
    p.add_argument("--mn-zip-csv", type=str, default="mn_zip_codes.csv",
                   help="Path to MN zip codes CSV (with a 'ZIP Code' column).")
    p.add_argument("--category-csv", type=str, default="category.csv",
                   help="Path to category CSV (expects data_id and url_path columns).")
    p.add_argument("--phase", choices=["stores", "products", "all"], default="all",
                   help="Which phase(s) to run.")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                   help="Max concurrent HTTP requests (default: 16).")
    p.add_argument("--max-zip-codes", type=int, default=None,
                   help="Optional cap on zip codes to iterate through for testing.")
    p.add_argument("--max-per-category", type=int, default=None,
                   help="Optional cap on products per (store x category) for testing.")
    p.add_argument("--use-selenium", action="store_true", help="Fetch PLP/PDP via a real browser.")
    p.add_argument("--no-headless", action="store_true", help="Run browser visible (recommended if challenged).")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")


if __name__ == "__main__":
    main()
