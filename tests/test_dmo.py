from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from data.dmo import OUTPUT_COLUMNS, parse_coupon, parse_gilts_in_issue

FIXTURE = Path(__file__).parent / "fixtures" / "gilts_in_issue_sample.xml"


@pytest.fixture(scope="module")
def df():
    return parse_gilts_in_issue(FIXTURE.read_bytes())


def test_columns_match_contract(df):
    assert list(df.columns) == OUTPUT_COLUMNS


def test_parses_all_rows(df):
    assert len(df) == 4


def test_conventional_vs_index_linked_flag(df):
    assert df["is_conventional"].sum() == 3
    assert not df.loc[df["isin"] == "GB00BYY5F581", "is_conventional"].iloc[0]


def test_known_gilt_row(df):
    row = df.loc[df["isin"] == "GB00BYZW3G56"].iloc[0]
    assert row["name"] == "1½% Treasury Gilt 2025"
    assert row["coupon"] == pytest.approx(1.5)
    assert row["redemption_date"] == dt.date(2025, 7, 22)
    assert row["is_conventional"] is True or row["is_conventional"] == 1
    assert row["amount_in_issue"] == pytest.approx(44673.738, rel=1e-6)
    assert row["close_of_business_date"] == dt.date(2024, 10, 20)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("1½% Treasury Gilt 2025", 1.5),
        ("0 3/8% Treasury Gilt 2025", 0.375),
        ("4 1/8% Treasury Gilt 2027", 4.125),
        ("6% Treasury Stock 2028", 6.0),
        ("0½% Treasury Gilt 2029", 0.5),
        ("4¼% Treasury Gilt 2027", 4.25),
        ("4 5/8% Treasury Gilt 2034", 4.625),
        ("", None),
        ("Treasury Gilt (no coupon)", None),
    ],
)
def test_parse_coupon_variants(name, expected):
    result = parse_coupon(name)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)
