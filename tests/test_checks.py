"""Tests for src.checks, using synthetic data where the right answer is known."""
import numpy as np
import pandas as pd
import pytest

from src.checks import check_prices, format_report


def make_clean(n=300, seed=0):
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * 1.005
    low = np.minimum(open_, close) * 0.995
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                         "Volume": 1000.0}, index=idx)


def frame_from_dates(dates):
    idx = pd.to_datetime(dates)
    return pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0}, index=idx)


def test_clean_data_passes():
    r = check_prices(make_clean(), name="clean")
    assert r.passed and r.hard_failures == []
    assert r.rows == 300
    assert r.duplicate_dates == 0 and r.nan_cells == 0 and r.non_positive == 0
    assert r.ohlc_inconsistent == 0 and r.long_gaps == []


def test_duplicate_dates_detected():
    df = make_clean()
    df = pd.concat([df, df.iloc[[10]]]).sort_index()
    r = check_prices(df)
    assert r.duplicate_dates == 1 and not r.passed


def test_nan_detected():
    df = make_clean()
    df.iloc[5, df.columns.get_loc("Close")] = np.nan
    df.iloc[6, df.columns.get_loc("Open")] = np.nan
    r = check_prices(df)
    assert r.nan_cells == 2 and not r.passed


def test_non_positive_detected():
    df = make_clean()
    df.iloc[7, df.columns.get_loc("Low")] = 0.0
    df.iloc[8, df.columns.get_loc("Close")] = -3.0
    r = check_prices(df)
    assert r.non_positive == 2 and not r.passed


def test_unsorted_detected():
    df = make_clean().iloc[::-1]
    r = check_prices(df)
    assert r.unsorted and not r.passed


def test_long_gap_detected():
    df = make_clean().drop(make_clean().index[100:108])  # 8 business days missing
    r = check_prices(df)
    assert len(r.long_gaps) == 1
    a, b, days = r.long_gaps[0]
    assert days > 5 and a < b
    assert r.passed  # a gap is informational, not a hard failure


def test_gap_threshold_is_more_than_5_calendar_days():
    assert check_prices(frame_from_dates(["2020-01-01", "2020-01-06"])).long_gaps == []   # 5 days
    assert len(check_prices(frame_from_dates(["2020-01-01", "2020-01-07"])).long_gaps) == 1  # 6 days
    assert check_prices(frame_from_dates(["2020-01-03", "2020-01-07"])).long_gaps == []   # Fri -> Tue


def test_biggest_moves_finds_injected_crash():
    df = make_clean()
    df.iloc[50, df.columns.get_loc("Close")] *= 0.8   # -20% one day, rebound the next
    r = check_prices(df)
    top2 = {d for d, _ in r.biggest_moves[:2]}
    assert top2 == {df.index[50].date(), df.index[51].date()}
    assert len(r.biggest_moves) <= 10
    mags = [abs(v) for _, v in r.biggest_moves]
    assert mags == sorted(mags, reverse=True)


def test_biggest_moves_exact_value():
    df = frame_from_dates(["2020-01-01", "2020-01-02", "2020-01-03"])
    df["Close"] = [100.0, 110.0, 99.0]
    r = check_prices(df)
    assert r.biggest_moves[0][1] == pytest.approx(0.10)
    assert r.biggest_moves[1][1] == pytest.approx(-0.10)


def test_ohlc_inconsistency_counted():
    df = make_clean()
    df.iloc[3, df.columns.get_loc("High")] = df.iloc[3]["Low"] - 1   # high below low
    r = check_prices(df)
    assert r.ohlc_inconsistent >= 1
    assert r.passed  # informational only


def test_zero_volume_counted():
    df = make_clean()
    df["Volume"] = 0.0
    assert check_prices(df).zero_volume_days == 300


def test_missing_close_column_fails():
    df = make_clean().drop(columns=["Close"])
    r = check_prices(df)
    assert not r.has_close and not r.passed


def test_empty_frame_does_not_crash():
    r = check_prices(pd.DataFrame())
    assert not r.passed
    assert "FAIL" in format_report(r)


def test_rows_per_year():
    r = check_prices(frame_from_dates(["2019-12-30", "2019-12-31", "2020-01-02"]))
    assert r.rows_per_year == {2019: 2, 2020: 1}


def test_format_report_is_plain_ascii_and_states_result():
    text = format_report(check_prices(make_clean(), name="demo"))
    text.encode("ascii")  # raises if any non-ASCII character slipped in
    assert "RESULT: PASS" in text and "=== demo ===" in text
