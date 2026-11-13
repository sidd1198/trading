"""Tests for src.metrics against hand-computed values."""
import numpy as np
import pandas as pd
import pytest

from src.metrics import (annualized_vol, cagr, drawdown_series, max_drawdown, sharpe,
                         summarize, total_return)


def series(values, start="2020-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def test_total_return_compounds():
    assert total_return(series([0.0, 0.10, 0.10])) == pytest.approx(0.21)
    assert total_return(series([0.0, 0.10, -0.10])) == pytest.approx(-0.01)


def test_cagr_exactly_four_years_doubling_twice():
    # 2020-01-01 -> 2024-01-01 is exactly 1461 days = 4.0 years of 365.25 days.
    r = pd.Series([0.0, 15.0], index=pd.to_datetime(["2020-01-01", "2024-01-01"]))  # equity 1 -> 16
    assert cagr(r) == pytest.approx(1.0)               # 16 ** (1/4) - 1 = 100% a year


def test_cagr_total_loss_and_too_short():
    r = pd.Series([0.0, -1.0], index=pd.to_datetime(["2020-01-01", "2021-01-01"]))
    assert cagr(r) == -1.0
    assert np.isnan(cagr(series([0.0])))


def test_annualized_vol_alternating_returns():
    r = series([0.0] + [0.01, -0.01] * 50)               # 100 real returns of +/-1%
    expected = 0.01 * np.sqrt(100 / 99) * np.sqrt(252)   # sample std of +/-1%, annualized
    assert annualized_vol(r) == pytest.approx(expected)


def test_vol_ignores_first_placeholder_bar():
    a = series([0.0, 0.01, -0.01, 0.02])
    b = series([0.5, 0.01, -0.01, 0.02])                 # different first bar, same real returns
    assert annualized_vol(a) == pytest.approx(annualized_vol(b))


def test_max_drawdown_hand_example():
    # equity path 1, 2, 1.2, 3, 1.5  ->  falls: 2->1.2 is -40%, 3->1.5 is -50%
    eq = np.array([1.0, 2.0, 1.2, 3.0, 1.5])
    r = series(np.r_[0.0, eq[1:] / eq[:-1] - 1.0])
    assert max_drawdown(r) == pytest.approx(-0.5)
    assert drawdown_series(r).iloc[1] == 0.0             # new high -> no drawdown


def test_drawdown_counts_loss_from_starting_capital():
    assert max_drawdown(series([-0.2, 0.05])) == pytest.approx(-0.2)   # never got back above 1.0


def test_max_drawdown_zero_when_always_rising():
    assert max_drawdown(series([0.0, 0.01, 0.02, 0.03])) == 0.0


def test_sharpe_formula_and_leverage_invariance():
    r = series([0.0] + [0.01, 0.03] * 50)
    expected = 0.02 * 252 / (0.01 * np.sqrt(100 / 99) * np.sqrt(252))
    assert sharpe(r) == pytest.approx(expected)
    doubled = r.copy()
    doubled.iloc[1:] *= 2                                # 2x leverage should not change Sharpe
    assert sharpe(doubled) == pytest.approx(sharpe(r))


def test_sharpe_with_risk_free_rate_equal_to_mean_return_is_zero():
    r = series([0.0] + [0.01, 0.03] * 50)
    assert sharpe(r, rf=0.02 * 252) == pytest.approx(0.0, abs=1e-9)


def test_sharpe_is_nan_when_there_is_no_volatility():
    assert np.isnan(sharpe(series([0.0] + [0.01] * 20)))


def test_summarize_has_all_keys():
    s = summarize(series([0.0, 0.01, -0.02, 0.03]))
    assert set(s) == {"start", "end", "years", "bars", "total_return", "cagr",
                      "ann_vol", "max_drawdown", "sharpe"}
    assert s["bars"] == 4
