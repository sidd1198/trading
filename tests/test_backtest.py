"""Tests for src.backtest. Prices grow exactly 1% a day, so every answer is hand-computable."""
import numpy as np
import pandas as pd
import pytest

from src.backtest import buy_and_hold, run_backtest


def growing_close(n=20, g=0.01):
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(100.0 * (1 + g) ** np.arange(n), index=idx)


def step_signal(close, on_from, on_to=None):
    s = pd.Series(0.0, index=close.index)
    s.iloc[on_from:on_to] = 1.0
    return s


def test_buy_and_hold_without_costs_equals_price_ratio():
    c = growing_close()
    bt = buy_and_hold(c)
    assert bt["equity"].iloc[-1] == pytest.approx(c.iloc[-1] / c.iloc[0])


def test_buy_and_hold_pays_entry_and_exit_cost():
    c = growing_close()
    bp = 10
    bt = buy_and_hold(c, cost_bps=bp)
    expected = (1 - bp / 10_000) ** 2 * c.iloc[-1] / c.iloc[0]
    assert bt["equity"].iloc[-1] == pytest.approx(expected)


def test_without_liquidation_only_the_entry_cost_is_paid():
    c = growing_close()
    bt = run_backtest(c, pd.Series(1.0, index=c.index), cost_bps=10, lag=1,
                      initial_position=1.0, liquidate_at_end=False)
    assert bt["equity"].iloc[-1] == pytest.approx((1 - 0.001) * c.iloc[-1] / c.iloc[0])


def test_flat_signal_stays_in_cash_and_pays_nothing():
    c = growing_close()
    bt = run_backtest(c, pd.Series(0.0, index=c.index), cost_bps=50)
    assert (bt["equity"] == 1.0).all() and (bt["turnover"] == 0).all()


def test_default_lag_trades_at_next_close():
    c = growing_close()
    k = 5                                        # signal turns on at the close of bar 5
    bt2 = run_backtest(c, step_signal(c, k), lag=2)
    assert bt2["pos"].iloc[k + 1] == 0.0 and bt2["pos"].iloc[k + 2] == 1.0
    bt1 = run_backtest(c, step_signal(c, k), lag=1)
    assert bt1["pos"].iloc[k] == 0.0 and bt1["pos"].iloc[k + 1] == 1.0


def test_future_signals_cannot_change_past_returns():
    """The look-ahead guard: changing the signal from bar k on must not touch
    strategy returns before bar k + lag."""
    c = growing_close(40)
    rng = np.random.default_rng(3)
    base = pd.Series(rng.integers(0, 2, len(c)).astype(float), index=c.index)
    k = 20
    altered = base.copy()
    altered.iloc[k:] = 1.0 - altered.iloc[k:]
    for lag in (1, 2, 3):
        a = run_backtest(c, base, cost_bps=10, lag=lag)["strat_ret"]
        b = run_backtest(c, altered, cost_bps=10, lag=lag)["strat_ret"]
        pd.testing.assert_series_equal(a.iloc[:k + lag], b.iloc[:k + lag])


def test_single_entry_hand_example():
    c = growing_close(20)
    bp = 10
    bt = run_backtest(c, step_signal(c, 5), cost_bps=bp, lag=1, liquidate_at_end=False)
    # signal on from bar 5 -> position from bar 6, earning 1% on bars 6..19 = 14 bars, one entry cost
    assert bt["equity"].iloc[5] == 1.0
    assert bt["equity"].iloc[-1] == pytest.approx((1 - 0.001) * 1.01 ** 14)


def test_round_trip_pays_two_costs():
    c = growing_close(20)
    bp = 10
    bt = run_backtest(c, step_signal(c, 5, 10), cost_bps=bp, lag=1)   # signal on for bars 5..9
    # position held on bars 6..10 (5 bars), entry on bar 6 and exit on bar 11
    assert bt["pos"].iloc[6:11].eq(1.0).all() and bt["pos"].iloc[11:].eq(0.0).all()
    assert bt["equity"].iloc[-1] == pytest.approx((1 - 0.001) ** 2 * 1.01 ** 5)
    assert bt["turnover"].sum() == 2.0


def test_costs_reduce_final_equity_for_an_active_signal():
    c = growing_close(60)
    s = pd.Series(([1.0, 0.0] * 30), index=c.index)      # flips every day
    free = run_backtest(c, s, cost_bps=0)["equity"].iloc[-1]
    costly = run_backtest(c, s, cost_bps=20)["equity"].iloc[-1]
    assert costly < free


def test_inputs_are_validated():
    c = growing_close()
    ones = pd.Series(1.0, index=c.index)
    with pytest.raises(ValueError):
        run_backtest(c, ones, lag=0)
    with pytest.raises(ValueError):
        run_backtest(c, ones * 2)                         # leverage not allowed
    with pytest.raises(ValueError):
        run_backtest(c, ones.iloc[:-1])                   # signal does not cover every bar
    bad = c.copy()
    bad.iloc[3] = -1.0
    with pytest.raises(ValueError):
        run_backtest(bad, ones)
    with pytest.raises(ValueError):
        run_backtest(c.iloc[::-1], ones)                  # unsorted
    with pytest.raises(ValueError):
        run_backtest(c, ones, cost_bps=-1)
