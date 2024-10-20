from __future__ import annotations

import datetime as dt
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from lxml import etree

logger = logging.getLogger(__name__)

DMO_D1A_URL = "https://www.dmo.gov.uk/data/XmlDataReport?reportCode=D1A"
DEFAULT_CACHE_DIR = Path("data/raw/dmo")
DEFAULT_CACHE_MAX_AGE_DAYS = 7

# vulgar fractions in names
_VULGAR_FRACTIONS: dict[str, float] = {
    "¼": 0.25,
    "½": 0.50,
    "¾": 0.75,
    "⅛": 0.125,
    "⅜": 0.375,
    "⅝": 0.625,
    "⅞": 0.875,
    "⅓": 1 / 3,
    "⅔": 2 / 3,
}

_COUPON_RE = re.compile(
    r"""^\s*
        (?P<whole>\d+)                       # whole-number part
        (?:\s*(?P<vulgar>[¼½¾⅛⅜⅝⅞⅓⅔]))?     # optional vulgar fraction glyph
        (?:\s+(?P<num>\d+)/(?P<den>\d+))?    # or optional space-separated "1/8"
        \s*%                                 # trailing percent sign
    """,
    re.VERBOSE,
)

OUTPUT_COLUMNS = [
    "isin",
    "name",
    "coupon",
    "redemption_date",
    "first_issue_date",
    "amount_in_issue",
    "instrument_type",
    "maturity_bracket",
    "is_conventional",
    "close_of_business_date",
]


def parse_coupon(name: str) -> Optional[float]:
    # extract coupon from name

    if not name:
        return None
    match = _COUPON_RE.match(name)
    if not match:
        return None

    whole = int(match.group("whole"))
    frac = 0.0
    if match.group("vulgar"):
        frac = _VULGAR_FRACTIONS[match.group("vulgar")]
    elif match.group("num") and match.group("den"):
        frac = int(match.group("num")) / int(match.group("den"))
    return whole + frac


def _parse_date(value: Optional[str]) -> Optional[dt.date]:
    if not value:
        return None
    # strip time from dmo dates
    return dt.datetime.fromisoformat(value).date()


def parse_gilts_in_issue(xml_bytes: bytes) -> pd.DataFrame:
    # parse dmo xml

    root = etree.fromstring(xml_bytes)
    records: list[dict] = []
    for el in root.iter("View_GILTS_IN_ISSUE"):
        attrs = dict(el.attrib)
        instrument_type = (attrs.get("INSTRUMENT_TYPE") or "").strip()
        name = attrs.get("INSTRUMENT_NAME", "")
        records.append(
            {
                "isin": attrs.get("ISIN_CODE"),
                "name": name,
                "coupon": parse_coupon(name),
                "redemption_date": _parse_date(attrs.get("REDEMPTION_DATE")),
                "first_issue_date": _parse_date(attrs.get("FIRST_ISSUE_DATE")),
                "amount_in_issue": _to_float(attrs.get("TOTAL_AMOUNT_IN_ISSUE")),
                "instrument_type": instrument_type,
                "maturity_bracket": attrs.get("MATURITY_BRACKET"),
                "is_conventional": instrument_type.lower() == "conventional",
                "close_of_business_date": _parse_date(
                    attrs.get("CLOSE_OF_BUSINESS_DATE")
                ),
            }
        )

    df = pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS)
    if df.empty:
        return df

    # enforce types
    df["coupon"] = pd.to_numeric(df["coupon"], errors="coerce")
    df["amount_in_issue"] = pd.to_numeric(df["amount_in_issue"], errors="coerce")
    df["is_conventional"] = df["is_conventional"].astype(bool)

    missing_coupons = df[df["coupon"].isna() & df["is_conventional"]]
    if not missing_coupons.empty:
        logger.warning(
            "Could not parse coupons for %d conventional gilts: %s",
            len(missing_coupons),
            missing_coupons["name"].tolist(),
        )

    return df


def _to_float(value: Optional[str]) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _latest_cached(cache_dir: Path) -> Optional[Path]:
    if not cache_dir.exists():
        return None
    candidates = sorted(cache_dir.glob("gilts_in_issue_*.xml"))
    return candidates[-1] if candidates else None


def _cache_age_days(path: Path, today: dt.date) -> int:
    mtime = dt.date.fromtimestamp(path.stat().st_mtime)
    return (today - mtime).days


def fetch_gilts_in_issue(
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    max_age_days: int = DEFAULT_CACHE_MAX_AGE_DAYS,
    force_refresh: bool = False,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    # fetch and parse dmo report with caching

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    today = dt.date.today()

    latest = _latest_cached(cache_dir)
    if (
        not force_refresh
        and latest is not None
        and _cache_age_days(latest, today) <= max_age_days
    ):
        logger.info("Using cached DMO D1A file: %s", latest)
        xml_bytes = latest.read_bytes()
    else:
        logger.info("Downloading DMO D1A report from %s", DMO_D1A_URL)
        client = session or requests
        response = client.get(DMO_D1A_URL, timeout=30)
        response.raise_for_status()
        xml_bytes = response.content
        target = cache_dir / f"gilts_in_issue_{today.isoformat()}.xml"
        target.write_bytes(xml_bytes)
        logger.info("Cached DMO D1A file at %s", target)

    return parse_gilts_in_issue(xml_bytes)
