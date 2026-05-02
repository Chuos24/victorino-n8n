"""DataPipeline orchestrates fetching for the full universe."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd

from .fetcher import DataFetcher

logger = logging.getLogger(__name__)


class DataPipeline:
    """Fetch and cache OHLCV bars for every symbol in the universe."""

    def __init__(self, fetcher: DataFetcher | None = None):
        self.fetcher = fetcher or DataFetcher()

    def fetch_all(
        self,
        universe: Iterable[str],
        timeframe: str,
        start: str | datetime,
        end: str | datetime | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetch bars for all symbols and return a dict keyed by symbol."""
        results: dict[str, pd.DataFrame] = {}
        for sym in universe:
            try:
                df = self.fetcher.get_bars(sym, timeframe, start, end)
                results[sym] = df
                logger.info("Fetched %s: %d bars", sym, len(df))
            except Exception as e:
                logger.warning("Failed to fetch %s: %s", sym, e)
                results[sym] = pd.DataFrame(
                    columns=["open", "high", "low", "close", "volume"]
                )
        return results
