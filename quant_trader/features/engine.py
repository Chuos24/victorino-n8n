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
        "ema_200_slope",
        "atr_pct_252",
        "hl_range_pct",
        "body_wick_ratio",
        "obv_z_20",
        "price_vs_52w_high",
        "cross_btc_ret_7d",
    ]

    def __init__(
        self,
        drop_warmup: bool = True,
        cross_assets: dict[str, pd.DataFrame] | None = None,
    ):
        self.drop_warmup = drop_warmup
        # `cross_assets` maps a symbol -> its raw OHLCV frame. Used to bake
        # cross-asset features (e.g. BTC's 7-day return as a regime signal
        # for ETH and SPY rows). Set via DataPipeline before training.
        self.cross_assets = cross_assets or {}

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

        # 200-period EMA slope, normalised by price. Positive => trending
        # market, the regime where momentum trades have positive expectancy.
        # Use a 20-bar lookback so daily values aren't dominated by noise.
        ema200_lag = ema200.shift(20)
        out["ema_200_slope"] = (ema200 - ema200_lag) / ema200_lag.replace(0, np.nan)

        # ATR rolling percentile over the last 252 bars (~ a year of daily
        # data). Used as a volatility-regime filter: very low ATR ⇒ chop and
        # whipsaw; very high ATR ⇒ regime breaks where trend signals
        # misfire. Sweet spot is the 30-70 percentile band.
        atr = out["atr_14"]
        out["atr_pct_252"] = atr.rolling(252, min_periods=60).rank(pct=True)

        # Range / candle features
        out["hl_range_pct"] = (high - low) / close.replace(0, np.nan)
        body = (close - out["open"].astype(float)).abs()
        wick = (high - low) - body
        out["body_wick_ratio"] = body / wick.replace(0, np.nan)

        # On-Balance Volume z-score (20 bars). OBV is the cumulative
        # volume signed by daily direction; the z-score normalises away
        # the absolute level so the model sees "is volume confirming /
        # diverging from price right now".
        direction = np.sign(close.diff().fillna(0.0))
        obv = (direction * vol).cumsum()
        obv_mean = obv.rolling(20).mean()
        obv_std = obv.rolling(20).std().replace(0, np.nan)
        out["obv_z_20"] = (obv - obv_mean) / obv_std

        # Price vs 52-week (252-bar) high. Range (0, 1]. Above ~0.97 is
        # "near a fresh high" → momentum anchor. Below ~0.7 is deep
        # drawdown territory.
        roll_high_252 = close.rolling(252, min_periods=60).max().replace(0, np.nan)
        out["price_vs_52w_high"] = close / roll_high_252

        # Cross-asset momentum: BTC's 7-day log return as a regime feature
        # for everything else. BTC is the highest-vol macro proxy in the
        # universe, so its multi-day move carries information about risk
        # appetite relevant to ETH (correlated) and SPY (anti-correlated
        # in stress regimes). For BTC's own rows the feature is its own
        # 7-day return — equivalent to log_ret_7d on a daily timeframe,
        # but with a different scale weighting that the model can learn.
        # When no cross-asset frame is available (e.g. unit-test
        # fixtures) we fall back to the target frame's own 7-day return
        # so the feature is defined and the warmup-drop step doesn't
        # wipe every row.
        btc_ret = self._cross_asset_btc_7d(out.index)
        if btc_ret.isna().all():
            btc_ret = log_close.diff(7)
        out["cross_btc_ret_7d"] = btc_ret

        # Replace ±inf and drop warmup NaN rows.
        out = out.replace([np.inf, -np.inf], np.nan)
        if self.drop_warmup:
            out = out.dropna(subset=self.FEATURE_COLUMNS).copy()
        return out

    def _cross_asset_btc_7d(self, target_index: pd.Index) -> pd.Series:
        """7-day BTC log return aligned to the target frame's index."""
        btc_df = self.cross_assets.get("BTC-USD")
        if btc_df is None or btc_df.empty or "close" not in btc_df.columns:
            return pd.Series(np.nan, index=target_index)
        btc_close = btc_df["close"].astype(float)
        btc_ret = np.log(btc_close.replace(0, np.nan)).diff(7)
        # Reindex onto the target's timestamps with forward-fill: BTC
        # trades 7 days a week so its index is a superset of SPY's, but
        # this also handles the rare case where bars don't line up.
        return btc_ret.reindex(target_index, method="ffill")
