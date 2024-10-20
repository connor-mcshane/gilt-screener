from __future__ import annotations

import datetime as dt
import math
from typing import Optional, Union

Number = Union[int, float]


def _is_number(value) -> bool:
    if value is None:
        return False
    try:
        return not math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def time_to_maturity(
    redemption_date: dt.date | dt.datetime,
    as_of: dt.date | dt.datetime,
) -> Optional[float]:
    # years to maturity act/365f
    if isinstance(redemption_date, dt.datetime):
        redemption_date = redemption_date.date()
    if isinstance(as_of, dt.datetime):
        as_of = as_of.date()
    days = (redemption_date - as_of).days
    if days <= 0:
        return None
    return days / 365.0


def running_yield(coupon: Optional[Number], clean_price: Optional[Number]) -> Optional[float]:
    # running yield
    if not (_is_number(coupon) and _is_number(clean_price)) or float(clean_price) == 0:
        return None
    return 100.0 * float(coupon) / float(clean_price)


def ytm_check(
    coupon: Number,
    clean_price: Number,
    time_to_maturity_years: Number,
    coupons_per_year: int = 2,
) -> Optional[float]:
    # sanity check ytm using newton raphson
    if not all(_is_number(x) for x in (coupon, clean_price, time_to_maturity_years)):
        return None
    n = max(int(round(float(time_to_maturity_years) * coupons_per_year)), 1)
    c = float(coupon) / coupons_per_year
    price = float(clean_price)

    y = 0.04 / coupons_per_year
    for _ in range(100):
        pv = sum(c / (1 + y) ** k for k in range(1, n + 1)) + 100 / (1 + y) ** n
        dpv = -sum(k * c / (1 + y) ** (k + 1) for k in range(1, n + 1)) - n * 100 / (1 + y) ** (n + 1)
        diff = pv - price
        if abs(diff) < 1e-8:
            break
        if dpv == 0:
            return None
        y -= diff / dpv
        if y <= -1:
            return None

    return 100.0 * y * coupons_per_year


def post_tax_yield(
    coupon: Optional[Number],
    clean_price: Optional[Number],
    time_to_maturity_years: Optional[Number],
    tax_rate: float,
) -> Optional[float]:
    # post tax yield assuming tax free capital gain
    if not all(
        _is_number(x) for x in (coupon, clean_price, time_to_maturity_years)
    ) or float(clean_price) == 0 or float(time_to_maturity_years) == 0:
        return None

    after_tax_coupon = float(coupon) * (1.0 - tax_rate)
    pull_to_par = 100.0 - float(clean_price)
    annualised_gain = pull_to_par / float(time_to_maturity_years)
    return 100.0 * (after_tax_coupon + annualised_gain) / float(clean_price)


def tax_free_return_pct(
    coupon: Optional[Number],
    clean_price: Optional[Number],
    time_to_maturity_years: Optional[Number],
) -> Optional[float]:
    # % of return from capital gain
    if not all(
        _is_number(x) for x in (coupon, clean_price, time_to_maturity_years)
    ) or float(time_to_maturity_years) <= 0:
        return None

    if float(clean_price) >= 100.0:
        return 0.0

    annualised_gain = (100.0 - float(clean_price)) / float(time_to_maturity_years)
    total_return = float(coupon) + annualised_gain

    if total_return <= 0:
        return 0.0

    return 100.0 * (annualised_gain / total_return)


def cgt_attractiveness_flag(
    coupon: Optional[Number],
    clean_price: Optional[Number],
    max_coupon: float = 2.0,
    max_price: float = 100.0,
) -> bool:
    # flag cgt efficient gilts
    if not (_is_number(coupon) and _is_number(clean_price)):
        return False
    return float(coupon) <= max_coupon and float(clean_price) < max_price
