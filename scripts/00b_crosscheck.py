"""Stage 0b: compare an ETF with its index, and show the suspicious raw rows.

Usage (PowerShell, from C:\\dev\\trading, after 00_check_data.py has cached the data):
    python scripts\\00b_crosscheck.py
    python scripts\\00b_crosscheck.py --a '^NSEI' --b NIFTYBEES.NS --threshold 0.01
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.checks import bad_rows  # noqa: E402
from src.crosscheck import crosscheck, format_crosscheck  # noqa: E402
from src.data import get_prices, safe_name  # noqa: E402

REPORTS = ROOT / "reports"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", default="^NSEI", help="reference series (the index)")
    p.add_argument("--b", default="NIFTYBEES.NS", help="series to test (the ETF)")
    p.add_argument("--threshold", type=float, default=0.01, help="flag days where returns differ by more than this")
    p.add_argument("--top", type=int, default=15)
    args = p.parse_args()

    a, b = get_prices(args.a), get_prices(args.b)
    name_a, name_b = safe_name(args.a), safe_name(args.b)
    res = crosscheck(a, b, threshold=args.threshold, top=args.top)

    out = [format_crosscheck(name_a, name_b, res), ""]
    out.append(f"--- raw rows of {name_b} around the 3 largest differences (2 rows either side) ---")
    for d in list(res["worst"].index[:3]):
        i = b.index.get_loc(d)
        out.append(f"\n[around {d.date()}]")
        out.append(b.iloc[max(i - 2, 0): i + 3].to_string(float_format=lambda x: f"{x:,.4f}"))

    bad = bad_rows(b)
    out.append(f"\n--- rows of {name_b} with NaN/non-positive prices or inconsistent OHLC: {len(bad)} ---")
    for d in list(bad.index[:5]):
        i = b.index.get_loc(d)
        out.append(f"\n[around {d.date()}]")
        out.append(b.iloc[max(i - 1, 0): i + 2].to_string(float_format=lambda x: f"{x:,.4f}"))

    text = "\n".join(out)
    print(text)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"crosscheck_{name_b}_vs_{name_a}.txt").write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
