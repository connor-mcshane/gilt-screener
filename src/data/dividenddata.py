from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

DIVIDEND_DATA_URL = "https://www.dividenddata.co.uk/uk-gilts-prices-yields.py"
DEFAULT_CACHE_DIR = Path("data/raw/dividenddata")
DEFAULT_CACHE_MAX_AGE_DAYS = 1

OUTPUT_COLUMNS = [
    "epic",
    "name",
    "coupon",
    "redemption_date",
    "clean_price",
    "yield_to_maturity",
    "close_date",
]


def _parse_percentage(val: str) -> Optional[float]:
    val = val.strip().replace("%", "")
    if not val or val == "-":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _parse_price(val: str) -> Optional[float]:
    val = val.strip().replace("£", "").replace(",", "")
    if not val or val == "-":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def parse_dividenddata_html(html_text: str, fetch_date: dt.date) -> pd.DataFrame:
    # parse html table
    soup = BeautifulSoup(html_text, "html.parser")
    table = soup.find("table")
    if not table:
        raise ValueError("Could not find the gilt table on dividenddata.co.uk")

    records = []
    # skip header
    for row in table.find_all("tr")[1:]:
        cols = [td.text.strip() for td in row.find_all("td")]
        if len(cols) < 8:
            continue
            
        epic = cols[0]
        name = cols[1]
        coupon_str = cols[2]
        maturity_str = cols[3]
        price_str = cols[5]
        ytm_str = cols[7]

        try:
            redemption_date = dt.datetime.strptime(maturity_str, "%d-%b-%Y").date()
        except ValueError:
            logger.warning("Could not parse maturity date: %s", maturity_str)
            continue

        records.append({
            "epic": epic,
            "name": name,
            "coupon": _parse_percentage(coupon_str),
            "redemption_date": redemption_date,
            "clean_price": _parse_price(price_str),
            "yield_to_maturity": _parse_percentage(ytm_str),
            "close_date": fetch_date,
        })

    df = pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS)
    return df


def fetch_latest_prices(
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    max_age_days: int = DEFAULT_CACHE_MAX_AGE_DAYS,
    force_refresh: bool = False,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    # fetch latest prices
    
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    today = dt.date.today()

    # check cache
    target = cache_dir / f"dividenddata_{today.isoformat()}.html"
    
    if not force_refresh and target.exists():
        logger.info("Using cached dividenddata file: %s", target)
        html_text = target.read_text(encoding="utf-8")
    else:
        logger.info("Downloading gilt prices from %s", DIVIDEND_DATA_URL)
        client = session or requests
        # spoof user agent
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = client.get(DIVIDEND_DATA_URL, headers=headers, timeout=30)
        response.raise_for_status()
        html_text = response.text
        target.write_text(html_text, encoding="utf-8")
        logger.info("Cached dividenddata file at %s", target)

    return parse_dividenddata_html(html_text, today)
