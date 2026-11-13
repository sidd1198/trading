"""A deliberately small backtest engine: ONE asset, long or flat, daily bars.

Read these conventions. They are exactly where backtests quietly lie.

  * signal[t] is decided using data up to and INCLUDING the close of bar t.
  * lag=2 (default, conservative): the trade executes at the NEXT bar's close,
    so the new position first earns the return of bar t+2.
  * lag=1 (optimistic): the trade executes at the same close the signal was
    computed from. Only realistic if you can act on the closing price itself.
  * cost_bps is the ALL-IN cost of ONE side (buy or sell) in basis points:
    brokerage + taxes + exchange fees + slippage. 1 bp = 0.01%. It is charged on
    turnover, i.e. on every change of position.
  * Positions are between 0 (flat) and 1 (fully invested). No leverage, no shorting.
  * Returns per bar:  (1 + pos * asset_return) * (1 - cost_rate * turnover) - 1
"""
from __future__ import annotations

import pandas as pd


def run_backtest(close: pd.Series, signal: pd.Series, cost_bps: float = 0.0, lag: int = 2,
                 initial_position: float = 0.0, liquidate_at_end: bool = True) -> pd.DataFrame:
    """Run a long/flat backtest. Returns a DataFrame with columns:
    ret (asset), pos, turnover, strat_ret, equity (starts from 1.0)."""
    if lag < 1:
        raise ValueError("lag must be at least 1 (a signal cannot trade before it exists)")
    if not 0 <= cost_bps < 10_000:
        raise ValueError("cost_bps must be in [0, 10000)")
    close = close.astype(float)
    if close.index.has_duplicates or not close.index.is_monotonic_increasing:
        raise ValueError("close must have a sorted index without duplicate dates")
    if close.isna().any() or (close <= 0).any():
        raise ValueError("close must be positive with no NaN")
    signal = signal.reindex(close.index)
    if signal.isna().any():
        raise ValueError("signal must cover every bar and contain no NaN")
    if ((signal < 0) | (signal > 1)).any():
        raise ValueError("signal must be between 0 and 1 (long/flat only)")
    if not 0 <= initial_position <= 1:
        raise ValueError("initial_position must be between 0 and 1")

    ret = close.pct_change().fillna(0.0)                 # bar 0 has no previous close
    pos = signal.shift(lag).fillna(initial_position)     # position actually held during each bar
    turnover = pos.diff().abs()
    turnover.iloc[0] = abs(pos.iloc[0])                  # buying the initial position costs too
    if liquidate_at_end:
        turnover.iloc[-1] += pos.iloc[-1]                # sell everything on the last bar
    strat_ret = (1.0 + pos * ret) * (1.0 - cost_bps / 10_000 * turnover) - 1.0
    return pd.DataFrame({"ret": ret, "pos": pos, "turnover": turnover,
                         "strat_ret": strat_ret, "equity": (1.0 + strat_ret).cumprod()})


def buy_and_hold(close: pd.Series, cost_bps: float = 0.0) -> pd.DataFrame:
    """Buy at the first close, sell at the last close, pay costs on both."""
    signal = pd.Series(1.0, index=close.index)
    return run_backtest(close, signal, cost_bps=cost_bps, lag=1,
                        initial_position=1.0, liquidate_at_end=True)
