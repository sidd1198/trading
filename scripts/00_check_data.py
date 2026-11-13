"""Stage 0: download (or load) daily prices, run sanity checks, save a chart.

Usage (PowerShell, from C:\\dev\\trading):
    python scripts\\00_check_data.py
    python scripts\\00_check_data.py --refresh
    python scripts\\00_check_data.py --tickers '^NSEI' 'RELIANCE.NS'
    python scripts\\00_check_data.py --csv C:\\path\\to\\file.csv
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt  # noqa: E402

from src.checks import check_prices, format_report  # noqa: E402
from src.data import get_prices, load_csv, safe_name  # noqa: E402

REPORTS = ROOT / "reports"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", nargs="+", default=["^NSEI", "NIFTYBEES.NS"])
    p.add_argument("--start", default="2000-01-01")
    p.add_argument("--refresh", action="store_true", help="re-download even if cached")
    p.add_argument("--csv", nargs="+", help="check CSV file(s) instead of downloading")
    p.add_argument("--dayfirst", action="store_true", help="dates in CSV are day/month/year")
    args = p.parse_args()

    jobs = []  # (name, loader)
    if args.csv:
        for f in args.csv:
            jobs.append((Path(f).stem, lambda f=f: load_csv(f, dayfirst=args.dayfirst)))
    else:
        for t in args.tickers:
            jobs.append((safe_name(t), lambda t=t: get_prices(t, start=args.start, refresh=args.refresh)))

    REPORTS.mkdir(exist_ok=True)
    all_ok = True
    for name, load in jobs:
        try:
            df = load()
        except Exception as e:  # keep going so one bad ticker does not hide the others
            print(f"=== {name} ===\nCOULD NOT LOAD: {type(e).__name__}: {e}\n")
            all_ok = False
            continue
        rep = check_prices(df, name=name)
        text = format_report(rep)
        print(text + "\n")
        (REPORTS / f"data_check_{name}.txt").write_text(text + "\n", encoding="utf-8")
        all_ok &= rep.passed

        if rep.has_close and len(df):
            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.plot(df.index, df["Close"], linewidth=1)
            ax.set_yscale("log")
            ax.set_title(f"{name}: adjusted close (log scale)")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(REPORTS / f"prices_{name}.png", dpi=110)
            plt.close(fig)

    print("Charts and text reports are in:", REPORTS)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
