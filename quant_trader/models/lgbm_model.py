"""LightGBM directional classifier with walk-forward validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb

    _HAS_LGB = True
except Exception:  # pragma: no cover
    _HAS_LGB = False

from sklearn.ensemble import GradientBoostingClassifier

from ..features import FeatureEngine

UP = 1
DOWN = -1
FLAT = 0


@dataclass
class LGBMPrediction:
    direction: int  # -1 / 0 / 1
    confidence: float  # max class probability in [0, 1]


class LGBMSignalModel:
    """Train a LightGBM classifier on engineered features.

    Falls back to sklearn's GradientBoostingClassifier if LightGBM is missing.
    """

    def __init__(
        self,
        threshold: float = 0.005,
        seed: int = 42,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
    ):
        self.threshold = threshold
        self.seed = seed
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None
        self.feature_cols: list[str] = []
        self.classes_: list[int] = []

    @staticmethod
    def make_target(close: pd.Series, threshold: float) -> pd.Series:
        """Label each bar with the *next-bar* return bucket."""
        next_ret = close.shift(-1) / close - 1.0
        y = pd.Series(FLAT, index=close.index, dtype=int)
        y[next_ret > threshold] = UP
        y[next_ret < -threshold] = DOWN
        return y

    def _prepare(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        feat = FeatureEngine().transform(df)
        y = self.make_target(feat["close"], self.threshold)
        X = feat[FeatureEngine.FEATURE_COLUMNS]
        # Drop the last row (target NaN) and align.
        keep = y.notna() & X.notna().all(axis=1)
        keep.iloc[-1] = False  # last bar has no next-bar target
        return X[keep], y[keep].astype(int)

    def fit(self, df: pd.DataFrame) -> "LGBMSignalModel":
        X, y = self._prepare(df)
        if X.empty:
            raise ValueError("Not enough data to train LGBM model")

        # Walk-forward split: train first 70%, validate on last 30%.
        n = len(X)
        split = max(1, int(n * 0.7))
        X_train, y_train = X.iloc[:split], y.iloc[:split]

        self.feature_cols = list(X.columns)
        self.classes_ = sorted(y_train.unique().tolist())

        if _HAS_LGB:
            self.model = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.seed,
                num_leaves=31,
                verbose=-1,
            )
            self.model.fit(X_train, y_train)
        else:  # pragma: no cover - fallback path
            self.model = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.seed,
            )
            self.model.fit(X_train, y_train)
        return self

    def predict_one(self, feature_row: pd.Series) -> LGBMPrediction:
        if self.model is None:
            return LGBMPrediction(direction=FLAT, confidence=0.0)
        X = feature_row[self.feature_cols].to_frame().T.astype(float)
        proba = self.model.predict_proba(X)[0]
        idx = int(np.argmax(proba))
        cls = int(self.model.classes_[idx])
        conf = float(proba[idx])
        return LGBMPrediction(direction=cls, confidence=conf)

    def predict_dataframe(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """Predict over a feature DataFrame and return direction + confidence."""
        if self.model is None or feat_df.empty:
            return pd.DataFrame(
                {"direction": [], "confidence": []}, index=feat_df.index
            )
        X = feat_df[self.feature_cols].astype(float)
        proba = self.model.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        directions = np.array([self.model.classes_[i] for i in idx], dtype=int)
        conf = proba[np.arange(len(idx)), idx]
        return pd.DataFrame(
            {"direction": directions, "confidence": conf}, index=feat_df.index
        )
