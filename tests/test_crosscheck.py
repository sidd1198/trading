"""Tests for src.crosscheck and checks.bad_rows, on synthetic data with known answers."""
import numpy as np
import pandas as pd
import pytest

from src.checks import bad_rows
from src.crosscheck import align_returns, crosscheck, format_crosscheck


def make_pair(n=400, seed=0):
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)
    a_close = 10000 * np.cumprod(1 + rng.normal(0.0004, 0.01, n))
    # the "ETF" follows the index with tiny tracking noise
    b_close = a_close / 100 * np.cumprod(1 + rng.normal(0, 0.0003, n))
    a = pd.DataFrame({"Close": a_close}, index=idx)
    b = pd.DataFrame({"Close": b_close, "Volume": 1000.0}, index=idx)
    return a, b


def test_tracking_series_have_high_correlation_and_no_flags():
    a, b = make_pair()
    res = crosscheck(a, b)
    assert res["correlation"] > 0.99
    assert res["n_over_threshold"] == 0
    assert res["median_abs_diff"] < 0.001


def test_injected_glitch_is_flagged_on_both_days():
    a, b = make_pair()
    k = 200
    b.iloc[k, b.columns.get_loc("Close")] *= 0.1   # bad price for one day: -90%, then +900%
    res = crosscheck(a, b)
    assert res["n_over_threshold"] == 2
    flagged = set(res["worst"].index)
    assert flagged == {b.index[k], b.index[k + 1]}
    assert res["worst"]["diff"].abs().is_monotonic_decreasing


def test_only_common_dates_are_compared():
    a, b = make_pair()
    dropped = b.index[50]
    b = b.drop(dropped)                        # b is missing one day that a has
    al = align_returns(a, b)
    assert dropped not in al.index
    assert len(al) == len(a) - 2               # 1 day missing, and the first day has no return
    assert crosscheck(a, b)["n_over_threshold"] == 0   # the 2-day return is still compared fairly


def test_duplicate_dates_rejected():
    a, b = make_pair()
    b = pd.concat([b, b.iloc[[3]]]).sort_index()
    with pytest.raises(ValueError):
        align_returns(a, b)


def test_format_is_ascii_and_mentions_counts():
    a, b = make_pair()
    b.iloc[100, b.columns.get_loc("Close")] *= 0.1
    text = format_crosscheck("NSEI", "ETF", crosscheck(a, b))
    text.encode("ascii")
    assert "days with |difference| > 1.0%: 2" in text


def test_bad_rows_finds_each_kind_of_problem():
    idx = pd.bdate_range("2020-01-01", periods=6)
    df = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0}, index=idx)
    df.iloc[1, df.columns.get_loc("Close")] = 0.0        # non-positive
    df.iloc[2, df.columns.get_loc("Open")] = np.nan      # NaN
    df.iloc[3, df.columns.get_loc("High")] = 98.0        # high below low
    out = bad_rows(df)
    assert list(out.index) == [idx[1], idx[2], idx[3]]


def test_bad_rows_empty_on_clean_data_and_handles_duplicate_dates():
    idx = pd.bdate_range("2020-01-01", periods=4)
    df = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0}, index=idx)
    assert len(bad_rows(df)) == 0
    dup = pd.concat([df, df.iloc[[0]]])
    assert len(bad_rows(dup)) == 0
