"""Stage 1: buy-and-hold baseline and metrics, on data that passed the Stage 0 checks.

Usage (PowerShell, from C:\\dev\\trading):
    python scripts\\01_buy_and_hold.py
    python scripts\\01_buy_and_hold.py --tickers '^NSEI' --cost-bps 10 --rf 0.0
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.backtest import buy_and_hold  # noqa: E402
from src.checks import check_prices  # noqa: E402
from src.data import get_prices, safe_name  # noqa: E402
from src.metrics import cagr, drawdown_series, summarize, years_spanned  # noqa: E402

REPORTS = ROOT / "reports"

# Windows fixed in advance (calendar blocks), NOT picked after looking at results.
PERIODS = [
    ("Full period", None, None),
    ("2008-2012", "2008-01-01", "2012-12-31"),
    ("2013-2017", "2013-01-01", "2017-12-31"),
    ("2018-2022", "2018-01-01", "2022-12-31"),
    ("2023-now", "2023-01-01", None),
]
COSTS_BPS = [0, 5, 10, 20, 30]


def pct(x):
    return f"{x * 100:8.2f}%"


def build_report(name, close, cost_bps, rf, is_index):
    L = [f"=== BUY-AND-HOLD: {name} ===",
         f"assumptions: all-in cost {cost_bps:g} bp per side (buy and sell), risk-free rate {rf * 100:.1f}% for Sharpe",
         "windows are fixed calendar blocks chosen in advance; each starts from its own first close"]
    if is_index:
        L.append("note: a price index excludes dividends, so an investable ETF would earn somewhat more")
    L += ["", f"{'period':<12}{'from':<12}{'to':<12}{'total':>10}{'CAGR':>10}{'vol':>10}{'maxDD':>10}{'Sharpe':>8}"]
    for label, start, end in PERIODS:
        c = close.loc[start:end]
        if len(c) < 60:
            L.append(f"{label:<12}(fewer than 60 bars, skipped)")
            continue
        s = summarize(buy_and_hold(c, cost_bps)["strat_ret"], rf)
        L.append(f"{label:<12}{s['start']!s:<12}{s['end']!s:<12}{pct(s['total_return'])}{pct(s['cagr'])}"
                 f"{pct(s['ann_vol'])}{pct(s['max_drawdown'])}{s['sharpe']:>8.2f}")

    # independent hand check of CAGR with zero costs
    first, last = float(close.iloc[0]), float(close.iloc[-1])
    yrs = years_spanned(close.index)
    hand = (last / first) ** (1 / yrs) - 1
    engine = cagr(buy_and_hold(close, 0.0)["strat_ret"])
    ok = abs(hand - engine) < 1e-9
    L += ["", "hand check of full-period CAGR at zero cost:",
          f"  first close {first:,.4f} on {close.index[0].date()}, last close {last:,.4f} on {close.index[-1].date()}, "
          f"{yrs:.4f} years",
          f"  by hand (last/first)^(1/years)-1 = {hand * 100:.4f}%   engine = {engine * 100:.4f}%   "
          + ("MATCH" if ok else "MISMATCH"),
          "", "cost sensitivity, full period (cost per side, buy and sell):",
          f"{'cost':>8}{'CAGR':>10}{'100 grows to':>16}"]
    for bp in COSTS_BPS:
        bt = buy_and_hold(close, bp)
        L.append(f"{bp:>5} bp{pct(cagr(bt['strat_ret']))}{100 * bt['equity'].iloc[-1]:>16,.2f}")
    L.append("buy-and-hold trades only twice, so costs barely matter here. They will matter a lot for active strategies.")
    return "\n".join(L), ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", nargs="+", default=["^NSEI"])
    p.add_argument("--cost-bps", type=float, default=10.0, help="ASSUMED all-in cost per side in basis points")
    p.add_argument("--rf", type=float, default=0.0, help="ASSUMED annual risk-free rate as a decimal")
    args = p.parse_args()

    REPORTS.mkdir(exist_ok=True)
    status = 0
    for t in args.tickers:
        name = safe_name(t)
        df = get_prices(t)
        rep = check_prices(df, name=name)
        if not rep.passed:
            print(f"=== {name} ===\nSKIPPED: data failed the Stage 0 hard checks ({'; '.join(rep.hard_failures)}).\n"
                  "Fix or replace the data first. A baseline on bad data is worse than no baseline.\n")
            status = 1
            continue
        close = df["Close"]
        text, ok = build_report(name, close, args.cost_bps, args.rf, is_index=t.startswith("^"))
        print(text + "\n")
        (REPORTS / f"bh_{name}.txt").write_text(text + "\n", encoding="utf-8")
        status |= 0 if ok else 1

        bt = buy_and_hold(close, args.cost_bps)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        ax1.plot(bt.index, 100 * bt["equity"], linewidth=1)
        ax1.set_yscale("log")
        ax1.set_title(f"{name}: buy-and-hold, 100 invested at the first close (log scale, {args.cost_bps:g} bp per side)")
        ax1.grid(alpha=0.3)
        ax2.fill_between(bt.index, 100 * drawdown_series(bt["strat_ret"]), 0, alpha=0.5)
        ax2.set_ylabel("drawdown %")
        ax2.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(REPORTS / f"bh_{name}.png", dpi=110)
        plt.close(fig)
    print("Charts and text reports are in:", REPORTS)
    return status


if __name__ == "__main__":
    sys.exit(main())
