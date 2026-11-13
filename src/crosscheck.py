"""Cross-check two price series that SHOULD move together (an ETF and its index).

An ETF tracking the Nifty 50 should have daily returns very close to the index's.
A large gap on one day is either a real event (circuit limits, illiquid close)
or bad data (a wrong price, a missing split adjustment). Either way, look at it
before trusting any backtest that uses the series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def align_returns(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Daily close-to-close returns of a and b on their COMMON dates only, so
    both returns always cover the same time interval."""
    for name, d in (("a", a), ("b", b)):
        if d.index.has_duplicates:
            raise ValueError(f"series {name} has duplicate dates; fix the data first")
    common = a.index.intersection(b.index).sort_values()
    ca, cb = a.loc[common, "Close"], b.loc[common, "Close"]
    out = pd.DataFrame({"ret_a": ca.pct_change(), "ret_b": cb.pct_change()}).iloc[1:]
    out["diff"] = out["ret_b"] - out["ret_a"]
    if "Volume" in b.columns:
        out["vol_b"] = b.loc[common, "Volume"].iloc[1:]
    return out


def crosscheck(a: pd.DataFrame, b: pd.DataFrame, threshold: float = 0.01, top: int = 15) -> dict:
    al = align_returns(a, b).replace([np.inf, -np.inf], np.nan).dropna(subset=["ret_a", "ret_b"])
    absdiff = al["diff"].abs()
    big = al.loc[absdiff[absdiff > threshold].sort_values(ascending=False).index]
    return {
        "n_days": len(al),
        "correlation": float(al["ret_a"].corr(al["ret_b"])) if len(al) > 2 else float("nan"),
        "median_abs_diff": float(absdiff.median()) if len(al) else float("nan"),
        "threshold": threshold,
        "n_over_threshold": len(big),
        "worst": big.head(top),
    }


def format_crosscheck(name_a: str, name_b: str, res: dict) -> str:
    """Plain ASCII text (safe for the Windows console)."""
    L = [f"=== CROSS-CHECK: {name_b} vs {name_a} (daily close-to-close returns) ==="]
    L.append(f"common days: {res['n_days']}   return correlation: {res['correlation']:.4f}")
    L.append(f"median |difference|: {res['median_abs_diff'] * 100:.3f}%   "
             f"days with |difference| > {res['threshold'] * 100:.1f}%: {res['n_over_threshold']}")
    w = res["worst"]
    if len(w):
        L.append(f"largest differences (showing {len(w)}):")
        L.append(f"  {'date':<12}{name_a[:10]:>11}{name_b[:10]:>11}{'diff':>10}{'vol_b':>12}")
        for d, row in w.iterrows():
            vol = f"{row['vol_b']:.0f}" if "vol_b" in w.columns and pd.notna(row["vol_b"]) else "-"
            L.append(f"  {d.date()!s:<12}{row['ret_a'] * 100:>10.2f}%{row['ret_b'] * 100:>10.2f}%"
                     f"{row['diff'] * 100:>9.2f}%{vol:>12}")
    return "\n".join(L)
