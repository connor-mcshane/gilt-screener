from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from analytics.metrics import (
    cgt_attractiveness_flag,
    post_tax_yield,
    running_yield,
    time_to_maturity,
    tax_free_return_pct,
)
from data import dmo, tradeweb, dividenddata

logger = logging.getLogger(__name__)

DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_PROCESSED_FILE = DEFAULT_PROCESSED_DIR / "gilts_latest.parquet"

DEFAULT_TAX_BANDS = (0.0, 0.20, 0.40, 0.45)


def build_latest_snapshot(
    tradeweb_dir: Path | str = tradeweb.DEFAULT_RAW_DIR,
    dmo_cache_dir: Path | str = dmo.DEFAULT_CACHE_DIR,
    output_path: Path | str | None = DEFAULT_PROCESSED_FILE,
    tax_bands: tuple[float, ...] = DEFAULT_TAX_BANDS,
    as_of: Optional[dt.date] = None,
    force_refresh_dmo: bool = False,
    use_dividenddata: bool = False,
) -> pd.DataFrame:
    # build joined snapshot

    ref = dmo.fetch_gilts_in_issue(
        cache_dir=dmo_cache_dir, force_refresh=force_refresh_dmo
    )
    ref = ref.loc[ref["is_conventional"]].copy()

    if use_dividenddata:
        prices = dividenddata.fetch_latest_prices()
        # join on maturity and coupon since no isins
        
        # drop name to avoid conflict
        prices = prices.drop(columns=["name"])
        
        # round coupons for join
        ref["_join_coupon"] = ref["coupon"].round(3)
        prices["_join_coupon"] = prices["coupon"].round(3)
        
        merged = ref.merge(
            prices, 
            left_on=["redemption_date", "_join_coupon"], 
            right_on=["redemption_date", "_join_coupon"],
            how="inner", 
            validate="one_to_one"
        )
        merged = merged.drop(columns=["_join_coupon", "coupon_y"])
        merged = merged.rename(columns={"coupon_x": "coupon"})
        
        # fill missing cols
        merged["dirty_price"] = pd.NA
        merged["accrued_interest"] = pd.NA
        
        logger.info("Joined DMO with DividendData on maturity and coupon.")
    else:
        prices = tradeweb.load_latest(tradeweb_dir)
        merged = ref.merge(prices, on="isin", how="inner", validate="one_to_one")
        
        extra_in_tradeweb = set(prices["isin"]) - set(ref["isin"])
        if extra_in_tradeweb:
            logger.info(
                "%d Tradeweb ISINs not in DMO D1A list (possible new issues / strips)",
                len(extra_in_tradeweb),
            )

    dropped_no_price = len(ref) - len(merged)
    if dropped_no_price:
        logger.info(
            "%d conventional gilts in DMO list had no matching price",
            dropped_no_price,
        )

    resolved_as_of = as_of or merged["close_date"].dropna().max()
    if resolved_as_of is None or pd.isna(resolved_as_of):
        resolved_as_of = dt.date.today()

    merged["snapshot_date"] = resolved_as_of
    merged["time_to_maturity"] = merged["redemption_date"].apply(
        lambda d: time_to_maturity(d, resolved_as_of) if pd.notna(d) else None
    )
    merged["running_yield"] = merged.apply(
        lambda row: running_yield(row["coupon"], row["clean_price"]),
        axis=1,
    )
    merged["tax_free_return_pct"] = merged.apply(
        lambda row: tax_free_return_pct(
            row["coupon"], row["clean_price"], row["time_to_maturity"]
        ),
        axis=1,
    )
    merged["cgt_attractive"] = merged.apply(
        lambda row: cgt_attractiveness_flag(row["coupon"], row["clean_price"]),
        axis=1,
    )
    for tax_rate in tax_bands:
        col = f"post_tax_yield_{int(round(tax_rate * 100))}"
        merged[col] = merged.apply(
            lambda row, t=tax_rate: post_tax_yield(
                coupon=row["coupon"],
                clean_price=row["clean_price"],
                time_to_maturity_years=row["time_to_maturity"],
                tax_rate=t,
            ),
            axis=1,
        )

    merged = merged.sort_values("redemption_date").reset_index(drop=True)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(output_path, index=False)
        logger.info("Wrote processed snapshot to %s", output_path)

    return merged


def load_latest_snapshot(
    path: Path | str = DEFAULT_PROCESSED_FILE,
) -> pd.DataFrame:
    # load latest snapshot

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No processed snapshot at {path}. Run build_latest_snapshot() first."
        )
    return pd.read_parquet(path)
