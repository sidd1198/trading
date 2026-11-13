"""Tests for src.data. No network: the download function is replaced by a fake."""
import pandas as pd
import pytest

import src.data as data
from src.data import get_prices, load_csv, normalize, safe_name


def fake_raw(n=5, tz="Asia/Kolkata"):
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz=tz)
    return pd.DataFrame({"Open": 100.0, "High": 102.0, "Low": 99.0, "Close": 101.0,
                         "Volume": 5.0}, index=idx)


def test_safe_name():
    assert safe_name("^NSEI") == "NSEI"
    assert safe_name("NIFTYBEES.NS") == "NIFTYBEES_NS"


def test_normalize_drops_timezone_and_keeps_calendar_date():
    out = normalize(fake_raw())
    assert out.index.tz is None
    assert out.index[0] == pd.Timestamp("2024-01-01")
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_normalize_flattens_multiindex_columns():
    raw = fake_raw(tz=None)
    raw.columns = pd.MultiIndex.from_product([raw.columns, ["^NSEI"]])
    out = normalize(raw)
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert out["Close"].iloc[0] == 101.0


def test_normalize_applies_adj_close_to_all_prices():
    raw = fake_raw(n=2, tz=None)
    raw["Adj Close"] = 50.5            # half of Close (101)
    out = normalize(raw)
    assert out["Close"].tolist() == [50.5, 50.5]
    assert out["Open"].tolist() == [50.0, 50.0]    # 100 * 0.5
    assert out["High"].tolist() == [51.0, 51.0]    # 102 * 0.5


def test_normalize_requires_close():
    with pytest.raises(ValueError):
        normalize(fake_raw().drop(columns=["Close"]))


def test_normalize_sorts_but_keeps_duplicates_for_the_checks_to_report():
    raw = fake_raw(n=3, tz=None)
    raw = pd.concat([raw, raw.iloc[[0]]]).iloc[::-1]
    out = normalize(raw)
    assert out.index.is_monotonic_increasing
    assert out.index.duplicated().sum() == 1


def test_load_csv_handles_commas_descending_order_and_nse_dates(tmp_path):
    f = tmp_path / "nse.csv"
    f.write_text(
        'Date,Open,High,Low,Close,Shares Traded\n'
        '03-Jan-2024,"1,010.0","1,020.0","1,000.0","1,015.5",300\n'
        '02-Jan-2024,"1,000.0","1,012.0","990.0","1,010.0",200\n')
    df = load_csv(f)
    assert df.index.is_monotonic_increasing
    assert df["Close"].tolist() == [1010.0, 1015.5]
    assert df["Volume"].tolist() == [200.0, 300.0]


def test_load_csv_dayfirst(tmp_path):
    f = tmp_path / "d.csv"
    f.write_text("Date,Close\n02/03/2024,10\n01/03/2024,9\n")
    df = load_csv(f, dayfirst=True)
    assert df.index[0] == pd.Timestamp("2024-03-01")


def test_load_csv_without_date_column_raises(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("A,Close\n1,2\n")
    with pytest.raises(ValueError):
        load_csv(f)


def test_get_prices_downloads_once_then_uses_cache(tmp_path, monkeypatch):
    calls = []

    def fake_download(ticker, start, end):
        calls.append(ticker)
        return fake_raw()

    monkeypatch.setattr(data, "_download_raw", fake_download)
    first = get_prices("^NSEI", raw_dir=tmp_path)
    second = get_prices("^NSEI", raw_dir=tmp_path)
    assert calls == ["^NSEI"]                      # second call came from the CSV cache
    assert (tmp_path / "NSEI.csv").exists()
    pd.testing.assert_frame_equal(first, second, check_freq=False, check_exact=False)
    get_prices("^NSEI", raw_dir=tmp_path, refresh=True)
    assert len(calls) == 2                         # refresh forces a new download


def test_empty_download_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "_download_raw", lambda t, s, e: pd.DataFrame())
    with pytest.raises(RuntimeError, match="No data returned"):
        get_prices("BADTICKER", raw_dir=tmp_path)
