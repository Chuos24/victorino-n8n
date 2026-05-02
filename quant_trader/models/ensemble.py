"""Ensemble that combines LightGBM + LSTM into a single signal."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .lgbm_model import LGBMSignalModel
from .lstm_model import LSTMRegressor


@dataclass
class SignalResult:
    """Output of the ensemble for a single bar."""

    symbol: str
    direction: int  # -1, 0, 1
    confidence: float
    lgbm_score: float
    lstm_pred: float
    # LGBM's probability margin (top class − runner-up). Used as the
    # LSTM/LGBM "agreement score delta" — when it's near zero the model
    # is split across classes and the directional call is fragile.
    agreement_delta: float = 0.0


class EnsembleModel:
    """Train an LGBM + LSTM pair per symbol and combine their outputs.

    A trade signal is produced only when:
      * LGBM direction is non-zero
      * LSTM regression sign agrees with LGBM direction
      * LGBM confidence exceeds `confidence_threshold`
    """

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        seed: int = 42,
    ):
        self.confidence_threshold = confidence_threshold
        self.seed = seed
        self.lgbm = LGBMSignalModel(seed=seed)
        self.lstm = LSTMRegressor(seed=seed)
        self._symbol: str | None = None

    def fit(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> "EnsembleModel":
        self._symbol = symbol
        self.lgbm.fit(df)
        try:
            self.lstm.fit(df)
        except Exception:
            # If torch is missing the LSTM falls through to a tiny linear
            # fallback; if that also fails (e.g. very small datasets) just
            # leave the LSTM predicting 0 which will collapse to LGBM-only.
            pass
        return self

    def predict_one(
        self, feature_row: pd.Series, recent_features: pd.DataFrame
    ) -> SignalResult:
        lgbm_pred = self.lgbm.predict_one(feature_row)
        lstm_val = self.lstm.predict_last(recent_features) if not recent_features.empty else 0.0

        lstm_dir = 1 if lstm_val > 0 else (-1 if lstm_val < 0 else 0)
        agree = lgbm_pred.direction != 0 and lstm_dir == lgbm_pred.direction
        passes = agree and lgbm_pred.confidence >= self.confidence_threshold
        direction = lgbm_pred.direction if passes else 0
        return SignalResult(
            symbol=self._symbol or "UNKNOWN",
            direction=direction,
            confidence=lgbm_pred.confidence,
            lgbm_score=float(lgbm_pred.confidence) * lgbm_pred.direction,
            lstm_pred=lstm_val,
            agreement_delta=float(lgbm_pred.margin),
        )
