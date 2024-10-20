from __future__ import annotations

import datetime as dt
import logging
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path("data/raw/tradeweb")

OUTPUT_COLUMNS = [
    "isin",
    "clean_price",
    "dirty_price",
    "accrued_interest",
    "yield_to_maturity",
    "close_date",
]

# map headers to our schema
_COLUMN_ALIASES: dict[str, str] = {
    "isin": "isin",
    "isin_code": "isin",
    "close_of_business_date": "close_date",
    "close_date": "close_date",
    "cob_date": "close_date",
    "date": "close_date",
    "clean_price": "clean_price",
    "closing_clean_price": "clean_price",
    "dirty_price": "dirty_price",
    "closing_dirty_price": "dirty_price",
    "accrued_interest": "accrued_interest",
    "accrued": "accrued_interest",
    "yield": "yield_to_maturity",
    "ytm": "yield_to_maturity",
    "yield_to_maturity": "yield_to_maturity",
    "gross_redemption_yield": "yield_to_maturity",
    "grr": "yield_to_maturity",
    # optional columns
    "type": "instrument_type",
    "instrument_type": "instrument_type",
}

_ISIN_RE = re.compile(r"^GB[0-9A-Z]{10}$")


def _normalise_header(header: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", header.strip().lower()).strip("_")
    return slug


def _build_rename_map(columns: Iterable[str]) -> dict[str, str]:
    rename: dict[str, str] = {}
    for col in columns:
        key = _normalise_header(col)
        target = _COLUMN_ALIASES.get(key)
        if target and target not in rename.values():
            rename[col] = target
    return rename


def _find_latest_file(raw_dir: Path) -> Path:
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Tradeweb raw directory does not exist: {raw_dir}. "
            "Download a gilt closing-prices CSV from "
            "https://reports.tradeweb.com/closing-prices/gilts/ and save it there."
        )
    candidates = [p for p in raw_dir.glob("*.csv") if not p.name.startswith(".")]
    if not candidates:
        raise FileNotFoundError(
            f"No Tradeweb CSVs found in {raw_dir}. Export a gilt closing-prices "
            "CSV from Tradeweb Insite and drop it in this folder."
        )
    # get most recently modified file
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _infer_close_date_from_name(path: Path) -> Optional[dt.date]:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", path.stem)
    if match:
        try:
            return dt.date.fromisoformat(match.group(1))
        except ValueError:
            return None
    return None


def load_tradeweb_csv(path: Path) -> pd.DataFrame:
    # parse tradeweb csv

    raw = pd.read_csv(path)
    rename = _build_rename_map(raw.columns)
    missing_required = {"isin", "clean_price", "yield_to_maturity"} - set(rename.values())
    if missing_required:
        raise ValueError(
            f"Tradeweb CSV {path.name} is missing required columns "
            f"{sorted(missing_required)}. Saw columns: {list(raw.columns)}."
        )

    df = raw.rename(columns=rename)[list(rename.values())].copy()

    if "instrument_type" in df.columns:
        conventional_mask = (
            df["instrument_type"].astype(str).str.strip().str.lower()
            == "conventional"
        )
        df = df.loc[conventional_mask].drop(columns=["instrument_type"])

    df["isin"] = df["isin"].astype(str).str.strip().str.upper()
    valid_isin = df["isin"].str.match(_ISIN_RE)
    dropped = int((~valid_isin).sum())
    if dropped:
        logger.info("Dropping %d rows with non-gilt ISINs (strips/T-bills/etc.)", dropped)
    df = df.loc[valid_isin].copy()

    for col in ("clean_price", "dirty_price", "accrued_interest", "yield_to_maturity"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "close_date" in df.columns:
        # handle date parsing
        sample = df["close_date"].dropna().astype(str).iloc[0] if len(df) else ""
        dayfirst = "/" in sample
        df["close_date"] = pd.to_datetime(
            df["close_date"], errors="coerce", dayfirst=dayfirst
        ).dt.date
    else:
        inferred = _infer_close_date_from_name(path)
        if inferred is None:
            raise ValueError(
                f"Could not determine close-of-business date for {path.name}. "
                "Include a 'Close of Business Date' column or use a filename "
                "like 'gilts_2025-05-02.csv'."
            )
        df["close_date"] = inferred

    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    return df[OUTPUT_COLUMNS].reset_index(drop=True)


def load_latest(raw_dir: Path | str = DEFAULT_RAW_DIR) -> pd.DataFrame:
    # load newest tradeweb csv

    raw_dir = Path(raw_dir)
    path = _find_latest_file(raw_dir)
    logger.info("Loading Tradeweb snapshot from %s", path)
    return load_tradeweb_csv(path)
