"""Shared pytest fixtures."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make sure the repo root is on sys.path when running pytest from any dir.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def synthetic_ohlcv() -> pd.DataFrame:
    """Realistic-shaped 30 days of hourly OHLCV bars for tests."""
    np.random.seed(7)
    n = 24 * 30
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    rets = np.random.normal(0, 0.005, n)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(np.random.normal(0, 0.002, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.002, n)))
    open_ = np.r_[close[0], close[:-1]]
    volume = np.random.uniform(1_000, 10_000, n)
    df = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum.reduce([open_, close, high]),
            "low": np.minimum.reduce([open_, close, low]),
            "close": close,
            "volume": volume,
        },
        index=idx,
    )
    return df
