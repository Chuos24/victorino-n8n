"""Tests for the data layer.

Most tests are mocked, but ``test_github_fallback_real_network`` does a real
HTTPS fetch against ``raw.githubusercontent.com`` so we have at least one
end-to-end check that the live-data path works. It is auto-skipped when the
network is unavailable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_trader.data.fetcher import DataFetcher


def test_standardize_columns():
    df = pd.DataFrame(
        {
            "Open": [1, 2],
            "High": [2, 3],
            "Low": [0.5, 1.5],
            "Close": [1.5, 2.5],
            "Volume": [100, 200],
        },
        index=pd.date_range("2024-01-01", periods=2, freq="1h"),
    )
    out = DataFetcher._standardize(df)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert not out.isna().any().any()


def test_cache_round_trip(tmp_path: Path, monkeypatch):
    df = pd.DataFrame(
        {
            "open": np.linspace(100, 110, 50),
            "high": np.linspace(101, 111, 50),
            "low": np.linspace(99, 109, 50),
            "close": np.linspace(100.5, 110.5, 50),
            "volume": np.linspace(1000, 2000, 50),
        },
        index=pd.date_range("2024-01-01", periods=50, freq="1h", tz="UTC"),
    )

    fetcher = DataFetcher(cache_dir=tmp_path)

    # Pretend yfinance returns our synthetic DataFrame the first time.
    def fake_yf(self, symbol, timeframe, start, end):
        return df

    def fake_ccxt(self, symbol, timeframe, start, end):
        return pd.DataFrame()

    monkeypatch.setattr(DataFetcher, "_fetch_yfinance", fake_yf)
    monkeypatch.setattr(DataFetcher, "_fetch_ccxt", fake_ccxt)

    out = fetcher.get_bars("FAKE", "1h", "2024-01-01", "2024-01-03")
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert not out.empty
    assert not out.isna().any().any()

    cache_file = tmp_path / "FAKE_1h.parquet"
    assert cache_file.exists()


def _have_network(host: str = "raw.githubusercontent.com") -> bool:
    import socket

    try:
        socket.create_connection((host, 443), timeout=3).close()
        return True
    except OSError:
        return False


@pytest.mark.parametrize("symbol", ["BTC-USD", "SPY"])
def test_github_fallback_real_network(tmp_path: Path, symbol: str):
    """Hit the real GitHub mirrors and verify we get a valid OHLCV frame."""
    if not _have_network():
        pytest.skip("no network access to raw.githubusercontent.com")
    fetcher = DataFetcher(cache_dir=tmp_path)
    df = fetcher._fetch_github_csv(
        symbol,
        "1d",
        datetime(2023, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    assert not df.empty, f"GitHub fallback returned empty frame for {symbol}"
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert (df["high"] >= df["low"]).all()
    assert (df["close"] > 0).all()
