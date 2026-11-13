"""Performance metrics computed from a series of DAILY simple returns.

Conventions:
  * Input is a pandas Series of daily returns with a DatetimeIndex. The first
    bar's return is a placeholder (there is no previous close), so volatility
    and Sharpe skip it. Total return, CAGR and drawdown include everything.
  * Capital starts at 1.0 before the first bar.
  * 252 trading days per year; CAGR uses calendar time (first to last date).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def equity_curve(r: pd.Series) -> pd.Series:
    return (1.0 + r).cumprod()


def total_return(r: pd.Series) -> float:
    return float((1.0 + r).prod() - 1.0)


def years_spanned(index: pd.DatetimeIndex) -> float:
    return (index[-1] - index[0]).days / 365.25


def cagr(r: pd.Series) -> float:
    """Compound annual growth rate: the constant yearly return that would give
    the same total growth over the same calendar time."""
    if len(r) < 2:
        return float("nan")
    years = years_spanned(r.index)
    growth = 1.0 + total_return(r)
    if years <= 0:
        return float("nan")
    if growth <= 0:
        return -1.0
    return float(growth ** (1.0 / years) - 1.0)


def annualized_vol(r: pd.Series) -> float:
    """Standard deviation of daily returns, scaled by sqrt(252)."""
    x = r.iloc[1:]
    if len(x) < 2:
        return float("nan")
    return float(x.std(ddof=1) * np.sqrt(TRADING_DAYS))


def drawdown_series(r: pd.Series) -> pd.Series:
    """Fall from the highest equity so far (the starting 1.0 counts as a peak)."""
    eq = equity_curve(r)
    peak = np.maximum(eq.cummax(), 1.0)
    return eq / peak - 1.0


def max_drawdown(r: pd.Series) -> float:
    """Worst peak-to-trough fall, as a negative number (-0.30 = -30%)."""
    return float(drawdown_series(r).min())


def sharpe(r: pd.Series, rf: float = 0.0) -> float:
    """Sharpe ratio: average excess return per unit of volatility, annualized.
    rf is the ANNUAL risk-free rate as a decimal (0.06 = 6%); default 0."""
    x = r.iloc[1:] - rf / TRADING_DAYS
    if len(x) < 2:
        return float("nan")
    sd = x.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:   # constant returns leave ~1e-18 of float noise, not exactly 0
        return float("nan")
    return float(x.mean() / sd * np.sqrt(TRADING_DAYS))


def summarize(r: pd.Series, rf: float = 0.0) -> dict:
    return {
        "start": r.index[0].date(),
        "end": r.index[-1].date(),
        "years": years_spanned(r.index) if len(r) > 1 else float("nan"),
        "bars": len(r),
        "total_return": total_return(r),
        "cagr": cagr(r),
        "ann_vol": annualized_vol(r),
        "max_drawdown": max_drawdown(r),
        "sharpe": sharpe(r, rf),
    }
