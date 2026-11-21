# Data tests: is cars_clean.csv safe to train on?
import pandas as pd

data = pd.read_csv("cars_clean.csv")

def test_columns_are_correct():
    assert list(data.columns) == ["year", "cc", "mileage", "imported", "automatic", "price_lakh"]

def test_no_empty_cells():
    assert data.notna().all().all()

def test_main_values_are_positive():
    # imported and automatic are 0/1 flags, so they are checked separately
    assert (data[["year", "cc", "mileage", "price_lakh"]] > 0).all().all()

def test_flags_are_0_or_1():
    assert data["imported"].isin([0, 1]).all()
    assert data["automatic"].isin([0, 1]).all()

def test_enough_cars():
    assert len(data) >= 1000