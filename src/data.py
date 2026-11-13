"""Loading and caching daily price data.

Convention used everywhere in this project:
  * DataFrame indexed by date (tz-naive DatetimeIndex, sorted ascending)
  * columns: Open, High, Low, Close, and optionally Volume
  * Close is the ADJUSTED close (splits and dividends included). Open, High
    and Low are scaled by the same factor so the four prices stay consistent.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PRICE_COLS = ["Open", "High", "Low", "Close"]


def safe_name(ticker: str) -> str:
    """'^NSEI' -> 'NSEI', 'NIFTYBEES.NS' -> 'NIFTYBEES_NS' (safe as a file name)."""
    return re.sub(r"[^A-Za-z0-9]+", "_", ticker.lstrip("^")).strip("_")


def _to_num(s: pd.Series) -> pd.Series:
    """Numbers -> float. Text like '21,000.50' is cleaned; junk becomes NaN."""
    if not pd.api.types.is_numeric_dtype(s):  # text (object or pandas 3 'str' dtype)
        s = s.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn a raw price table (yfinance, Yahoo website CSV, NSE CSV) into the
    project format. Sorting is lossless so it is done here; duplicates, NaNs
    and bad prices are NOT fixed here, the checks report them."""
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # some yfinance versions return (field, ticker) columns
        df.columns = df.columns.get_level_values(0)
    lookup = {str(c).strip().lower(): c for c in df.columns}
    if "close" not in lookup:
        raise ValueError(f"No 'Close' column found. Columns were: {list(df.columns)}")

    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)  # keep the local calendar date
    out = pd.DataFrame(index=idx.normalize())
    out.index.name = "Date"

    close = _to_num(df[lookup["close"]]).to_numpy()
    adj_key = next((k for k in ("adj close", "adj_close") if k in lookup), None)
    if adj_key is not None:
        adj = _to_num(df[lookup[adj_key]]).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = np.where(close > 0, adj / close, np.nan)
        close = adj
    else:
        factor = np.ones(len(df))

    for col in ("Open", "High", "Low"):
        if col.lower() in lookup:
            out[col] = _to_num(df[lookup[col.lower()]]).to_numpy() * factor
    out["Close"] = close
    for key in ("volume", "shares traded"):
        if key in lookup:
            out["Volume"] = _to_num(df[lookup[key]]).to_numpy()
            break
    return out.sort_index(kind="stable")


def load_csv(path, dayfirst: bool = False) -> pd.DataFrame:
    """Load a CSV with a Date column (Yahoo website, NSE, or our own cache)."""
    raw = pd.read_csv(path, skipinitialspace=True)
    raw.columns = [str(c).strip() for c in raw.columns]
    date_col = next((c for c in raw.columns if c.lower() in ("date", "datetime", "timestamp")), None)
    if date_col is None:
        raise ValueError(f"No Date column in {path}. Columns were: {list(raw.columns)}")
    raw.index = pd.to_datetime(raw[date_col], dayfirst=dayfirst)
    return normalize(raw.drop(columns=[date_col]))


def save_csv(df: pd.DataFrame, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label="Date")


def _download_raw(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    import yfinance as yf  # imported here so tests do not need the network

    return yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True, actions=False)


def download(ticker: str, start: str = "2000-01-01", end: str | None = None) -> pd.DataFrame:
    raw = _download_raw(ticker, start, end)
    if raw is None or len(raw) == 0:
        raise RuntimeError(
            f"No data returned for {ticker!r}. Check the ticker and your internet "
            "connection, or download a CSV by hand and use --csv."
        )
    return normalize(raw)


def get_prices(ticker: str, start: str = "2000-01-01", end: str | None = None,
               refresh: bool = False, raw_dir=None) -> pd.DataFrame:
    """Prices from the CSV cache in data/raw/, downloading on first use.
    Note: the cache ignores start/end changes; use refresh=True to re-download."""
    folder = Path(raw_dir) if raw_dir else RAW_DIR
    path = folder / f"{safe_name(ticker)}.csv"
    if path.exists() and not refresh:
        return load_csv(path)
    df = download(ticker, start, end)
    save_csv(df, path)
    return df
