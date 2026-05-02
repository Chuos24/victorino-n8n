"""Feature engineering for OHLCV data.

Produces a NaN-free DataFrame of derived features suitable for ML models.
Uses the `ta` library where available with manual fallbacks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:  # pragma: no cover - import-time dependency
    from ta.momentum import RSIIndicator
    from ta.trend import MACD, EMAIndicator
    from ta.volatility import AverageTrueRange, BollingerBands

    _HAS_TA = True
except Exception:  # pragma: no cover
    _HAS_TA = False


def _rsi_manual(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr_manual(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()


class FeatureEngine:
    """Compute a fixed set of features from an OHLCV DataFrame.

    The returned DataFrame is NaN-free (warmup rows are dropped) and contains
    every column the downstream models expect.
    """

    FEATURE_COLUMNS = [
        "log_ret_1h",
        "log_ret_4h",
        "log_ret_24h",
        "log_ret_7d",
        "vol_20",
        "vol_50",
        "zscore_20",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_pct_b",
        "bb_width",
        "atr_14",
        "volume_ratio_20",
        "ema_cross",
        "hl_range_pct",
        "body_wick_ratio",
    ]

    def __init__(self, drop_warmup: bool = True):
        self.drop_warmup = drop_warmup

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return df augmented with feature columns and free of NaNs."""
        if df is None or df.empty:
            return pd.DataFrame(columns=list(df.columns) + self.FEATURE_COLUMNS)

        out = df.copy()
        # Make sure required columns exist + numeric
        for c in ["open", "high", "low", "close", "volume"]:
            if c not in out.columns:
                raise ValueError(f"missing column {c}")
        close = out["close"].astype(float)
        high = out["high"].astype(float)
        low = out["low"].astype(float)
        vol = out["volume"].astype(float)

        log_close = np.log(close.replace(0, np.nan))
        log_ret = log_close.diff()

        out["log_ret_1h"] = log_ret
        out["log_ret_4h"] = log_close.diff(4)
        out["log_ret_24h"] = log_close.diff(24)
        out["log_ret_7d"] = log_close.diff(24 * 7)

        out["vol_20"] = log_ret.rolling(20).std()
        out["vol_50"] = log_ret.rolling(50).std()

        mean20 = close.rolling(20).mean()
        std20 = close.rolling(20).std().replace(0, np.nan)
        out["zscore_20"] = (close - mean20) / std20

        # RSI
        if _HAS_TA:
            out["rsi_14"] = RSIIndicator(close=close, window=14).rsi()
        else:
            out["rsi_14"] = _rsi_manual(close, 14)

        # MACD
        if _HAS_TA:
            macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
            out["macd"] = macd.macd()
            out["macd_signal"] = macd.macd_signal()
            out["macd_hist"] = macd.macd_diff()
        else:
            ema12 = _ema(close, 12)
            ema26 = _ema(close, 26)
            out["macd"] = ema12 - ema26
            out["macd_signal"] = _ema(out["macd"], 9)
            out["macd_hist"] = out["macd"] - out["macd_signal"]

        # Bollinger
        if _HAS_TA:
            bb = BollingerBands(close=close, window=20, window_dev=2)
            mavg = bb.bollinger_mavg()
            hband = bb.bollinger_hband()
            lband = bb.bollinger_lband()
        else:
            mavg = mean20
            std = close.rolling(20).std()
            hband = mavg + 2 * std
            lband = mavg - 2 * std
        bb_range = (hband - lband).replace(0, np.nan)
        out["bb_pct_b"] = (close - lband) / bb_range
        out["bb_width"] = bb_range / mavg.replace(0, np.nan)

        # ATR
        if _HAS_TA:
            out["atr_14"] = AverageTrueRange(
                high=high, low=low, close=close, window=14
            ).average_true_range()
        else:
            out["atr_14"] = _atr_manual(out, 14)

        # Volume ratio
        avg_vol = vol.rolling(20).mean().replace(0, np.nan)
        out["volume_ratio_20"] = vol / avg_vol

        # EMA crossover signal: +1 if 50 EMA above 200 EMA, -1 otherwise.
        if _HAS_TA:
            ema50 = EMAIndicator(close=close, window=50).ema_indicator()
            ema200 = EMAIndicator(close=close, window=200).ema_indicator()
        else:
            ema50 = _ema(close, 50)
            ema200 = _ema(close, 200)
        out["ema_cross"] = np.where(ema50 > ema200, 1.0, -1.0)
        # Mark warmup rows as NaN so they get dropped together with the rest.
        out.loc[ema200.isna(), "ema_cross"] = np.nan

        # Range / candle features
        out["hl_range_pct"] = (high - low) / close.replace(0, np.nan)
        body = (close - out["open"].astype(float)).abs()
        wick = (high - low) - body
        out["body_wick_ratio"] = body / wick.replace(0, np.nan)

        # Replace ±inf and drop warmup NaN rows.
        out = out.replace([np.inf, -np.inf], np.nan)
        if self.drop_warmup:
            out = out.dropna(subset=self.FEATURE_COLUMNS).copy()
        return out
