"""Data fetcher with yfinance + ccxt adapters and a parquet cache."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Map our canonical symbols -> ccxt Binance symbols (used as fallback only).
CCXT_SYMBOL_MAP = {
    "BTC-USD": "BTC/USDT",
    "ETH-USD": "ETH/USDT",
    "SOL-USD": "SOL/USDT",
}

# Map our timeframes -> yfinance / ccxt strings.
YF_INTERVAL_MAP = {"1h": "60m", "1d": "1d", "1m": "1m", "5m": "5m", "15m": "15m"}
CCXT_INTERVAL_MAP = {"1h": "1h", "1d": "1d", "1m": "1m", "5m": "5m", "15m": "15m"}

# yfinance imposes ~730d cap on intraday data.
YF_INTRADAY_LIMIT_DAYS = 720


class DataFetcher:
    """Fetch OHLCV bars with on-disk parquet caching.

    The fetcher tries yfinance first (works for both equities and the *-USD
    crypto tickers). If yfinance returns empty for a known crypto symbol it
    falls back to ccxt's public Binance endpoint.
    """

    def __init__(self, cache_dir: Path | str = CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, symbol: str, timeframe: str) -> Path:
        safe = symbol.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{safe}_{timeframe}.parquet"

    def _load_cache(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(symbol, timeframe)
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True)
            return df
        except Exception as e:
            logger.warning("Failed to read cache %s: %s", path, e)
            return None

    def _save_cache(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        path = self._cache_path(symbol, timeframe)
        df.to_parquet(path)

    @staticmethod
    def _standardize(df: pd.DataFrame) -> pd.DataFrame:
        """Force columns to lowercase open/high/low/close/volume."""
        if df is None or df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = df.copy()
        # Flatten MultiIndex columns yfinance can produce.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df.columns = [str(c).lower() for c in df.columns]
        rename = {"adj close": "adj_close"}
        df.rename(columns=rename, inplace=True)
        keep = ["open", "high", "low", "close", "volume"]
        for c in keep:
            if c not in df.columns:
                df[c] = pd.NA
        df = df[keep].copy()
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(how="any")
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        else:
            df.index = df.index.tz_convert("UTC") if df.index.tz else df.index.tz_localize("UTC")
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df

    def _fetch_yfinance(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
        except Exception as e:
            logger.warning("yfinance unavailable: %s", e)
            return pd.DataFrame()

        interval = YF_INTERVAL_MAP.get(timeframe, timeframe)

        # yfinance can't return >730 days of intraday history.
        if interval not in {"1d", "1wk", "1mo"}:
            min_start = datetime.now(timezone.utc) - timedelta(days=YF_INTRADAY_LIMIT_DAYS)
            start = max(start, min_start)

        try:
            df = yf.download(
                tickers=symbol,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval=interval,
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            return self._standardize(df)
        except Exception as e:
            logger.warning("yfinance fetch failed for %s: %s", symbol, e)
            return pd.DataFrame()

    def _fetch_ccxt(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        ccxt_sym = CCXT_SYMBOL_MAP.get(symbol)
        if not ccxt_sym:
            return pd.DataFrame()
        try:
            import ccxt  # type: ignore
        except Exception as e:
            logger.warning("ccxt unavailable: %s", e)
            return pd.DataFrame()
        try:
            ex = ccxt.binance({"enableRateLimit": True})
            tf = CCXT_INTERVAL_MAP.get(timeframe, timeframe)
            since = int(start.timestamp() * 1000)
            end_ms = int(end.timestamp() * 1000)
            all_rows: list[list[float]] = []
            while since < end_ms:
                rows = ex.fetch_ohlcv(ccxt_sym, timeframe=tf, since=since, limit=1000)
                if not rows:
                    break
                all_rows.extend(rows)
                last_ts = rows[-1][0]
                if last_ts <= since:
                    break
                since = last_ts + 1
                if len(rows) < 1000:
                    break
            if not all_rows:
                return pd.DataFrame()
            df = pd.DataFrame(
                all_rows, columns=["ts", "open", "high", "low", "close", "volume"]
            )
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            df = df.set_index("ts")
            return self._standardize(df)
        except Exception as e:
            logger.warning("ccxt fetch failed for %s: %s", symbol, e)
            return pd.DataFrame()

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: str | datetime,
        end: str | datetime | None = None,
    ) -> pd.DataFrame:
        """Return OHLCV bars for symbol over [start, end].

        Tries cache first. On cache miss, fetches via yfinance, then ccxt.
        Saves the merged result back to cache.
        """
        if isinstance(start, str):
            start_dt = pd.Timestamp(start, tz="UTC").to_pydatetime()
        else:
            start_dt = start.replace(tzinfo=start.tzinfo or timezone.utc)
        if end is None:
            end_dt = datetime.now(timezone.utc)
        elif isinstance(end, str):
            end_dt = pd.Timestamp(end, tz="UTC").to_pydatetime()
        else:
            end_dt = end.replace(tzinfo=end.tzinfo or timezone.utc)

        cached = self._load_cache(symbol, timeframe)
        if cached is not None and not cached.empty:
            cmin = cached.index.min()
            cmax = cached.index.max()
            # If the cache fully covers the requested window, return slice.
            if cmin <= pd.Timestamp(start_dt) and cmax >= pd.Timestamp(end_dt) - pd.Timedelta(days=1):
                return cached.loc[
                    pd.Timestamp(start_dt) : pd.Timestamp(end_dt)
                ].copy()
            # Otherwise extend forward from the cache tail.
            fetch_start = max(cmax.to_pydatetime(), start_dt)
        else:
            fetch_start = start_dt

        new_df = self._fetch_yfinance(symbol, timeframe, fetch_start, end_dt)
        if new_df.empty and symbol in CCXT_SYMBOL_MAP:
            new_df = self._fetch_ccxt(symbol, timeframe, fetch_start, end_dt)

        if cached is not None and not cached.empty:
            merged = pd.concat([cached, new_df]) if not new_df.empty else cached
        else:
            merged = new_df

        merged = self._standardize(merged) if not merged.empty else merged
        if not merged.empty:
            self._save_cache(symbol, timeframe, merged)

        if merged.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        return merged.loc[pd.Timestamp(start_dt) : pd.Timestamp(end_dt)].copy()
