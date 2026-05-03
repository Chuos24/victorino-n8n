"""LightGBM directional classifier with walk-forward validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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

# Importance dump path. One row per (symbol, feature) per fit call so the
# CSV survives across the whole universe and can be re-read after the run.
IMPORTANCE_PATH = Path(__file__).resolve().parents[1] / "models" / "lgbm_importance.csv"


@dataclass
class LGBMPrediction:
    direction: int  # -1 / 0 / 1
    confidence: float  # max class probability in [0, 1]
    margin: float = 0.0  # top class probability minus second-place probability


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
        early_stopping_rounds: int = 50,
        # On a small / noisy validation block the multi-class error can
        # bottom out at iter 1-3 even though the model would keep
        # improving with more trees. When best_iteration_ falls below
        # this floor we ignore the early-stopped model and refit
        # without early stopping using `min_useful_estimators` trees.
        min_useful_iterations: int = 20,
        min_useful_estimators: int = 100,
        importance_floor: float = 0.01,
        importance_path: Path | str | None = IMPORTANCE_PATH,
    ):
        self.threshold = threshold
        self.seed = seed
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.min_useful_iterations = min_useful_iterations
        self.min_useful_estimators = min_useful_estimators
        # `importance_floor` keeps features whose gain importance is at
        # least this fraction of the max-importance feature. e.g. 0.01 →
        # keep anything with ≥1 % of the leader's gain. Features below
        # the floor are pruned from the second-pass refit and excluded
        # from prediction.
        self.importance_floor = importance_floor
        self.importance_path = Path(importance_path) if importance_path else None
        self.model = None
        self.feature_cols: list[str] = []
        self.kept_features: list[str] = []
        self.dropped_features: list[str] = []
        self.classes_: list[int] = []

    @staticmethod
    def make_target(close: pd.Series, threshold: float) -> pd.Series:
        """Label each bar with the *next-bar* return bucket."""
        next_ret = close.shift(-1) / close - 1.0
        y = pd.Series(FLAT, index=close.index, dtype=int)
        y[next_ret > threshold] = UP
        y[next_ret < -threshold] = DOWN
        return y

    def _prepare(
        self,
        df: pd.DataFrame,
        cross_assets: dict[str, pd.DataFrame] | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        feat = FeatureEngine(cross_assets=cross_assets).transform(df)
        y = self.make_target(feat["close"], self.threshold)
        X = feat[FeatureEngine.FEATURE_COLUMNS]
        # Drop the last row (target NaN) and align.
        keep = y.notna() & X.notna().all(axis=1)
        keep.iloc[-1] = False  # last bar has no next-bar target
        return X[keep], y[keep].astype(int)

    def fit(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        cross_assets: dict[str, pd.DataFrame] | None = None,
    ) -> "LGBMSignalModel":
        X, y = self._prepare(df, cross_assets=cross_assets)
        if X.empty:
            raise ValueError("Not enough data to train LGBM model")

        # Walk-forward split: train first 70%, validate on last 30%.
        # The validation block doubles as the early-stopping holdout.
        n = len(X)
        split = max(1, int(n * 0.7))
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        X_val, y_val = X.iloc[split:], y.iloc[split:]

        self.feature_cols = list(X.columns)
        self.classes_ = sorted(y_train.unique().tolist())

        if _HAS_LGB:
            # First pass: full feature set, with early stopping on the
            # held-out validation block. The default `multi_logloss`
            # eval metric is too brittle on noisy financial data — it
            # diverges within a handful of trees and the run stops
            # before the model can learn anything. Switch to
            # `multi_error` (classification error) which is monotone in
            # the directional accuracy we actually care about.
            self.model = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.seed,
                num_leaves=31,
                metric="multi_error",
                verbose=-1,
            )
            fit_kwargs = {}
            if not X_val.empty:
                fit_kwargs["eval_set"] = [(X_val, y_val)]
                fit_kwargs["eval_metric"] = "multi_error"
                fit_kwargs["callbacks"] = [
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(0),
                ]
            self.model.fit(X_train, y_train, **fit_kwargs)

            # Compute importance, persist it, prune.
            kept, dropped, importance_rows = self._select_features(symbol=symbol)
            self.kept_features = kept
            self.dropped_features = dropped
            self._write_importance(importance_rows)

            # Second pass: refit on the kept features only so prediction
            # never uses pruned columns.
            if dropped and kept:
                self.model = lgb.LGBMClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    random_state=self.seed,
                    num_leaves=31,
                    metric="multi_error",
                    verbose=-1,
                )
                refit_kwargs = {}
                if not X_val.empty:
                    refit_kwargs["eval_set"] = [(X_val[kept], y_val)]
                    refit_kwargs["eval_metric"] = "multi_error"
                    refit_kwargs["callbacks"] = [
                        lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                        lgb.log_evaluation(0),
                    ]
                self.model.fit(X_train[kept], y_train, **refit_kwargs)
                self.feature_cols = kept

            # Safeguard: when the validation `multi_error` peaks at iter
            # 1-3 we end up with a stump that produces near-uniform
            # class probabilities (max conf ~1/n_classes). In that case
            # the model has not learned anything useful and the
            # confidence threshold filters every signal. Refit without
            # early stopping using a fixed budget so the model gets a
            # chance to train. We log it via `self.fallback_used` so
            # downstream code (and DECISIONS.md) can see when it kicks
            # in.
            best_iter = getattr(self.model, "best_iteration_", None) or 0
            self.fallback_used = False
            if best_iter < self.min_useful_iterations:
                logger_features = self.feature_cols
                self.model = lgb.LGBMClassifier(
                    n_estimators=self.min_useful_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    random_state=self.seed,
                    num_leaves=31,
                    metric="multi_error",
                    verbose=-1,
                )
                self.model.fit(X_train[logger_features], y_train)
                self.fallback_used = True
        else:  # pragma: no cover - fallback path
            self.model = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.seed,
            )
            self.model.fit(X_train, y_train)
            self.kept_features = list(X.columns)
            self.dropped_features = []
        return self

    def _select_features(self, symbol: str) -> tuple[list[str], list[str], list[dict]]:
        """Return (kept, dropped, importance_rows). Uses gain importance
        from the first-pass model. A feature is kept if its gain is at
        least `importance_floor * max_gain`."""
        importances = np.asarray(self.model.booster_.feature_importance(importance_type="gain"))
        cols = list(self.feature_cols)
        max_gain = float(importances.max()) if importances.size else 0.0
        floor = max_gain * self.importance_floor
        kept, dropped, rows = [], [], []
        for col, imp in zip(cols, importances):
            imp_f = float(imp)
            normalised = imp_f / max_gain if max_gain > 0 else 0.0
            keep = max_gain == 0 or imp_f >= floor
            (kept if keep else dropped).append(col)
            rows.append(
                {
                    "symbol": symbol,
                    "feature": col,
                    "gain": imp_f,
                    "normalised_gain": normalised,
                    "kept": bool(keep),
                }
            )
        return kept, dropped, rows

    def _write_importance(self, rows: list[dict]) -> None:
        if not rows or self.importance_path is None:
            return
        path = Path(self.importance_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        # Append per fit so callers see the full per-symbol breakdown.
        if path.exists():
            df.to_csv(path, mode="a", header=False, index=False)
        else:
            df.to_csv(path, index=False)

    def predict_one(self, feature_row: pd.Series) -> LGBMPrediction:
        if self.model is None:
            return LGBMPrediction(direction=FLAT, confidence=0.0, margin=0.0)
        cols = self.feature_cols
        X = feature_row[cols].to_frame().T.astype(float)
        proba = self.model.predict_proba(X)[0]
        idx = int(np.argmax(proba))
        cls = int(self.model.classes_[idx])
        conf = float(proba[idx])
        sorted_proba = np.sort(proba)[::-1]
        margin = float(sorted_proba[0] - sorted_proba[1]) if len(sorted_proba) >= 2 else float(sorted_proba[0])
        return LGBMPrediction(direction=cls, confidence=conf, margin=margin)

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
