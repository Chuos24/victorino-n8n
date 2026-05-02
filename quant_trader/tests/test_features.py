"""Tests for FeatureEngine."""

from __future__ import annotations

from quant_trader.features import FeatureEngine


def test_features_no_nans(synthetic_ohlcv):
    feat = FeatureEngine().transform(synthetic_ohlcv)
    assert not feat.empty
    for col in FeatureEngine.FEATURE_COLUMNS:
        assert col in feat.columns
    assert not feat[FeatureEngine.FEATURE_COLUMNS].isna().any().any()


def test_features_preserve_ohlcv(synthetic_ohlcv):
    feat = FeatureEngine().transform(synthetic_ohlcv)
    for col in ["open", "high", "low", "close", "volume"]:
        assert col in feat.columns
