from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from data.tradeweb import OUTPUT_COLUMNS, load_tradeweb_csv

FIXTURE = Path(__file__).parent / "fixtures" / "tradeweb_sample.csv"


def test_schema_contract():
    df = load_tradeweb_csv(FIXTURE)
    assert list(df.columns) == OUTPUT_COLUMNS


def test_filters_non_conventional():
    df = load_tradeweb_csv(FIXTURE)
    # should drop ilg
    assert len(df) == 3
    assert "GB00BYY5F581" not in df["isin"].values


def test_isin_normalised_and_validated():
    df = load_tradeweb_csv(FIXTURE)
    assert all(isin.startswith("GB") and len(isin) == 12 for isin in df["isin"])


def test_numeric_columns_parsed():
    df = load_tradeweb_csv(FIXTURE)
    row = df.loc[df["isin"] == "GB00BYZW3G56"].iloc[0]
    assert row["clean_price"] == pytest.approx(99.50)
    assert row["dirty_price"] == pytest.approx(99.87)
    assert row["accrued_interest"] == pytest.approx(0.37)
    assert row["yield_to_maturity"] == pytest.approx(4.45)
    assert row["close_date"] == dt.date(2024, 10, 20)



