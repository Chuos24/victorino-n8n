"""Lightweight LSTM regressor for next-bar return prediction.

Falls back to a numpy-based linear regressor on the same features if
PyTorch is unavailable so the rest of the pipeline still runs.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

from ..features import FeatureEngine

KEY_FEATURES = [
    "log_ret_1h",
    "rsi_14",
    "macd_hist",
    "atr_14",
    "volume_ratio_20",
]
INPUT_COLS = ["open", "high", "low", "close", "volume"] + KEY_FEATURES


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)


class _LSTMNet(nn.Module if _HAS_TORCH else object):  # type: ignore[misc]
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class LSTMRegressor:
    """Predict next-bar log return from a fixed-length sequence."""

    def __init__(
        self,
        seq_len: int = 24,
        hidden_size: int = 64,
        num_layers: int = 2,
        epochs: int = 10,
        batch_size: int = 64,
        lr: float = 1e-3,
        seed: int = 42,
    ):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.model = None
        self.feature_means: np.ndarray | None = None
        self.feature_stds: np.ndarray | None = None
        self._fallback_w: np.ndarray | None = None  # used if torch missing
        _set_seed(seed)

    def _build_sequences(
        self, feat_df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        cols = INPUT_COLS
        arr = feat_df[cols].astype(float).values
        log_close = np.log(feat_df["close"].astype(float).values)
        target = np.diff(log_close, prepend=log_close[0])
        # Standardize features
        if self.feature_means is None:
            self.feature_means = arr.mean(axis=0)
            self.feature_stds = arr.std(axis=0) + 1e-8
        arr_n = (arr - self.feature_means) / self.feature_stds

        xs, ys = [], []
        for i in range(self.seq_len, len(arr_n) - 1):
            xs.append(arr_n[i - self.seq_len : i])
            ys.append(target[i + 1])  # next-bar log return
        if not xs:
            return np.empty((0, self.seq_len, len(cols))), np.empty((0,))
        return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.float32)

    def fit(
        self,
        df: pd.DataFrame,
        cross_assets: dict[str, pd.DataFrame] | None = None,
    ) -> "LSTMRegressor":
        feat = FeatureEngine(cross_assets=cross_assets).transform(df)
        if feat.empty or len(feat) < self.seq_len + 10:
            raise ValueError("Not enough data to train LSTM")

        X, y = self._build_sequences(feat)
        if X.shape[0] == 0:
            raise ValueError("No training sequences could be built")

        n = len(X)
        split = max(1, int(n * 0.7))
        X_train, y_train = X[:split], y[:split]

        if not _HAS_TORCH:  # pragma: no cover
            X_flat = X_train.reshape(X_train.shape[0], -1)
            X_aug = np.hstack([X_flat, np.ones((len(X_flat), 1))])
            self._fallback_w, *_ = np.linalg.lstsq(X_aug, y_train, rcond=None)
            return self

        device = torch.device("cpu")
        net = _LSTMNet(
            input_size=X.shape[2],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        ).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        net.train()
        for epoch in range(self.epochs):
            for xb, yb in loader:
                opt.zero_grad()
                pred = net(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
        net.eval()
        self.model = net
        return self

    def _predict_array(self, X: np.ndarray) -> np.ndarray:
        if X.shape[0] == 0:
            return np.empty((0,), dtype=float)
        if not _HAS_TORCH:  # pragma: no cover
            X_flat = X.reshape(X.shape[0], -1)
            X_aug = np.hstack([X_flat, np.ones((len(X_flat), 1))])
            assert self._fallback_w is not None
            return X_aug @ self._fallback_w
        with torch.no_grad():
            return self.model(torch.from_numpy(X.astype(np.float32))).numpy()

    def predict_last(self, feat_df: pd.DataFrame) -> float:
        """Predict next-bar log return from the most recent sequence."""
        if self.model is None and self._fallback_w is None:
            return 0.0
        cols = INPUT_COLS
        if len(feat_df) < self.seq_len:
            return 0.0
        arr = feat_df[cols].astype(float).values[-self.seq_len :]
        arr_n = (arr - self.feature_means) / self.feature_stds
        seq = arr_n.reshape(1, self.seq_len, len(cols)).astype(np.float32)
        return float(self._predict_array(seq)[0])

    def predict_dataframe(self, feat_df: pd.DataFrame) -> pd.Series:
        """Return predicted next-bar log return aligned to feat_df index."""
        if self.model is None and self._fallback_w is None:
            return pd.Series(0.0, index=feat_df.index)
        cols = INPUT_COLS
        arr = feat_df[cols].astype(float).values
        arr_n = (arr - self.feature_means) / self.feature_stds

        out = np.full(len(feat_df), np.nan)
        if len(arr_n) <= self.seq_len:
            return pd.Series(out, index=feat_df.index).fillna(0.0)
        seqs = np.stack(
            [arr_n[i - self.seq_len : i] for i in range(self.seq_len, len(arr_n))]
        ).astype(np.float32)
        preds = self._predict_array(seqs)
        out[self.seq_len :] = preds
        return pd.Series(out, index=feat_df.index).fillna(0.0)
