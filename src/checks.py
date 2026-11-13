"""Data sanity checks. Run these BEFORE trusting any backtest number.

Hard failures (the data must not be used until fixed):
  no rows, missing Close, duplicate dates, unsorted dates, NaN prices,
  non-positive prices.
Informational (you look at them and decide): OHLC inconsistencies,
zero-volume days, long gaps, the largest daily moves, rows per year.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PRICE_COLS = ["Open", "High", "Low", "Close"]


@dataclass
class PriceReport:
    name: str
    rows: int
    first_date: object = None
    last_date: object = None
    has_close: bool = True
    duplicate_dates: int = 0
    unsorted: bool = False
    nan_cells: int = 0
    non_positive: int = 0
    ohlc_inconsistent: int = 0
    zero_volume_days: int | None = None
    long_gaps: list = field(default_factory=list)      # (from_date, to_date, calendar_days)
    biggest_moves: list = field(default_factory=list)  # (date, close-to-close return)
    rows_per_year: dict = field(default_factory=dict)

    @property
    def hard_failures(self) -> list:
        out = []
        if self.rows == 0:
            return ["no rows"]
        if not self.has_close:
            out.append("missing Close column")
        if self.duplicate_dates:
            out.append(f"{self.duplicate_dates} duplicate dates")
        if self.unsorted:
            out.append("dates not sorted")
        if self.nan_cells:
            out.append(f"{self.nan_cells} NaN price cells")
        if self.non_positive:
            out.append(f"{self.non_positive} non-positive prices")
        return out

    @property
    def passed(self) -> bool:
        return not self.hard_failures


def ohlc_bad_mask(df: pd.DataFrame) -> pd.Series:
    """True where High < Low, or Open/Close fall outside [Low, High] (tiny tolerance)."""
    tol = 1e-6 * df["Close"].abs()
    return ((df["High"] < df["Low"] - tol)
            | (df["Close"] > df["High"] + tol) | (df["Close"] < df["Low"] - tol)
            | (df["Open"] > df["High"] + tol) | (df["Open"] < df["Low"] - tol))


def bad_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a NaN or non-positive price, or inconsistent Open/High/Low/Close."""
    mask = np.zeros(len(df), dtype=bool)
    cols = [c for c in PRICE_COLS if c in df.columns]
    if cols:
        mask |= (df[cols].isna().any(axis=1) | (df[cols] <= 0).any(axis=1)).to_numpy()
    if all(c in df.columns for c in PRICE_COLS):
        mask |= ohlc_bad_mask(df).to_numpy()
    return df[mask]


def check_prices(df: pd.DataFrame, name: str = "", max_gap_days: int = 5,
                 top_moves: int = 10) -> PriceReport:
    rep = PriceReport(name=name, rows=len(df))
    if len(df) == 0:
        return rep

    idx = pd.DatetimeIndex(df.index)
    rep.first_date, rep.last_date = idx.min().date(), idx.max().date()
    rep.duplicate_dates = int(idx.duplicated().sum())
    rep.unsorted = not idx.is_monotonic_increasing
    rep.rows_per_year = {int(y): int(n) for y, n in pd.Series(1, index=idx).groupby(idx.year).sum().items()}

    cols = [c for c in PRICE_COLS if c in df.columns]
    rep.has_close = "Close" in df.columns
    if cols:
        rep.nan_cells = int(df[cols].isna().sum().sum())
        rep.non_positive = int((df[cols] <= 0).sum().sum())

    if all(c in df.columns for c in PRICE_COLS):
        rep.ohlc_inconsistent = int(ohlc_bad_mask(df).sum())

    if "Volume" in df.columns:
        rep.zero_volume_days = int((df["Volume"] == 0).sum())

    days = idx.unique().sort_values()
    if len(days) > 1:
        step = (days[1:] - days[:-1]).days
        rep.long_gaps = [(days[i].date(), days[i + 1].date(), int(d))
                         for i, d in enumerate(step) if d > max_gap_days]

    if rep.has_close:
        ret = df["Close"].sort_index().pct_change()
        vals, dates = ret.to_numpy(dtype=float), ret.index
        ok = np.isfinite(vals)
        order = np.argsort(-np.abs(np.where(ok, vals, 0.0)))
        rep.biggest_moves = [(dates[i].date(), float(vals[i])) for i in order[:top_moves] if ok[i]]
    return rep


def format_report(r: PriceReport) -> str:
    """Plain ASCII text (safe for the Windows console)."""
    L = [f"=== {r.name} ==="]
    if r.rows == 0:
        return "\n".join(L + ["rows: 0", "RESULT: FAIL (no rows)"])

    def flag(ok):
        return "OK" if ok else "FAIL"

    L.append(f"rows: {r.rows}   first: {r.first_date}   last: {r.last_date}")
    L.append("hard checks:")
    L.append(f"  Close column present ....... {flag(r.has_close)}")
    L.append(f"  duplicate dates ............ {r.duplicate_dates:<6} {flag(r.duplicate_dates == 0)}")
    L.append(f"  dates sorted ............... {'yes' if not r.unsorted else 'no':<6} {flag(not r.unsorted)}")
    L.append(f"  NaN price cells ............ {r.nan_cells:<6} {flag(r.nan_cells == 0)}")
    L.append(f"  non-positive prices ........ {r.non_positive:<6} {flag(r.non_positive == 0)}")
    L.append("informational (you judge these):")
    L.append(f"  OHLC inconsistent rows ..... {r.ohlc_inconsistent}")
    if r.zero_volume_days is not None:
        L.append(f"  zero-volume days ........... {r.zero_volume_days} (index data often has no volume)")
    L.append(f"  gaps over 5 calendar days .. {len(r.long_gaps)}")
    for a, b, d in r.long_gaps[:15]:
        L.append(f"      {a} -> {b}  ({d} days)")
    if len(r.long_gaps) > 15:
        L.append(f"      ... and {len(r.long_gaps) - 15} more")
    L.append("  rows per year: " + " ".join(f"{y}:{n}" for y, n in r.rows_per_year.items()))
    if r.biggest_moves:
        L.append(f"  {len(r.biggest_moves)} largest daily moves (close to close):")
        for d, v in r.biggest_moves:
            L.append(f"      {d}  {v * 100:+7.2f}%")
    L.append("RESULT: " + ("PASS" if r.passed else "FAIL (" + "; ".join(r.hard_failures) + ")"))
    return "\n".join(L)
