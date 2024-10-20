from __future__ import annotations

import datetime as dt

import pytest

from analytics.metrics import (
    cgt_attractiveness_flag,
    post_tax_yield,
    running_yield,
    time_to_maturity,
    ytm_check,
    tax_free_return_pct,
)


def test_time_to_maturity_actual_365f():
    result = time_to_maturity(dt.date(2030, 1, 1), dt.date(2025, 1, 1))
    # 2028 is leap year
    assert result == pytest.approx(1826 / 365.0, rel=1e-9)


def test_time_to_maturity_past_redemption_returns_none():
    assert time_to_maturity(dt.date(2020, 1, 1), dt.date(2025, 1, 1)) is None


def test_running_yield_basic():
    assert running_yield(1.5, 99.5) == pytest.approx(100 * 1.5 / 99.5)


def test_running_yield_handles_missing_inputs():
    assert running_yield(None, 99.5) is None
    assert running_yield(1.5, None) is None
    assert running_yield(1.5, 0) is None


def test_ytm_check_reproduces_par_bond():
    # par bond ytm equals coupon
    result = ytm_check(coupon=4.0, clean_price=100.0, time_to_maturity_years=5.0)
    assert result == pytest.approx(4.0, abs=1e-3)


def test_ytm_check_discount_bond_yield_above_coupon():
    # subpar bond ytm > coupon
    result = ytm_check(coupon=1.0, clean_price=90.0, time_to_maturity_years=5.0)
    assert result is not None
    assert result > 1.0


@pytest.mark.parametrize(
    "tax_rate,expected",
    [
        (0.00, 100 * (0.375 + 9.0 / 5) / 91.0),
        (0.20, 100 * (0.375 * 0.80 + 9.0 / 5) / 91.0),
        (0.40, 100 * (0.375 * 0.60 + 9.0 / 5) / 91.0),
        (0.45, 100 * (0.375 * 0.55 + 9.0 / 5) / 91.0),
    ],
)
def test_post_tax_yield_low_coupon_subpar(tax_rate, expected):
    # low coupon subpar gilt
    result = post_tax_yield(
        coupon=0.375,
        clean_price=91.0,
        time_to_maturity_years=5.0,
        tax_rate=tax_rate,
    )
    assert result == pytest.approx(expected, rel=1e-9)


def test_post_tax_yield_higher_tax_lowers_yield():
    basic = post_tax_yield(4.0, 95.0, 5.0, 0.20)
    additional = post_tax_yield(4.0, 95.0, 5.0, 0.45)
    assert basic > additional


def test_cgt_flag_only_low_coupon_subpar():
    assert cgt_attractiveness_flag(0.375, 91.0)
    assert cgt_attractiveness_flag(2.0, 99.99)
    assert not cgt_attractiveness_flag(4.25, 101.0)
    assert not cgt_attractiveness_flag(0.375, 100.0)
    assert not cgt_attractiveness_flag(5.0, 95.0)
    assert not cgt_attractiveness_flag(None, 95.0)


def test_tax_free_return_pct():
    # 0% coupon, 90 price, 5 years -> 100% tax free
    assert tax_free_return_pct(0.0, 90.0, 5.0) == 100.0
    # 5% coupon, 105 price, 5 years -> 0% tax free (no capital gain)
    assert tax_free_return_pct(5.0, 105.0, 5.0) == 0.0
    # 2% coupon, 90 price, 5 years -> gain is 2/yr, coupon is 2/yr -> 50% tax free
    assert tax_free_return_pct(2.0, 90.0, 5.0) == 50.0
