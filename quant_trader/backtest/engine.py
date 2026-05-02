"""Event-driven backtester with strict no-lookahead semantics."""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Dict, List

import pandas as pd

from ..features import FeatureEngine
from ..models import EnsembleModel, SignalResult
from ..risk import KellyPositionSizer, RiskManager
from ..risk.sizer import TradeStats
from ..strategies import MomentumMeanReversion, StrategyAction

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    symbol: str
    entry_time: str
    exit_time: str
    direction: int
    entry_price: float
    exit_price: float
    units: float
    pnl: float
    return_pct: float
    reason: str
    entry_confidence: float = 0.0
    entry_regime_slope: float = 0.0
    bars_held: int = 0


class Backtester:
    """Walk bar-by-bar across one or more symbols, never looking ahead."""

    def __init__(
        self,
        initial_capital: float = 100_000,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        max_position_pct: float = 0.02,
        max_drawdown_kill: float = 0.10,
        max_concurrent_positions: int = 5,
        confidence_threshold: float = 0.6,
        stop_atr_mult: float = 1.5,
        take_profit_atr_mult: float = 2.5,
        max_holding_bars: int = 48,
        seed: int = 42,
        train_fraction: float = 0.5,
        regime_filter: bool = False,
        regime_slope_threshold: float = 0.0,
        long_only: bool = False,
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.max_position_pct = max_position_pct
        self.max_drawdown_kill = max_drawdown_kill
        self.max_concurrent_positions = max_concurrent_positions
        self.confidence_threshold = confidence_threshold
        self.stop_atr_mult = stop_atr_mult
        self.take_profit_atr_mult = take_profit_atr_mult
        self.max_holding_bars = max_holding_bars
        self.seed = seed
        self.train_fraction = train_fraction
        self.regime_filter = regime_filter
        self.regime_slope_threshold = regime_slope_threshold
        self.long_only = long_only

        # Per-open-position metadata so closes can fill entry_time / confidence
        # without scanning history.
        self._open_meta: Dict[str, dict] = {}

        self.risk = RiskManager(
            initial_capital=initial_capital,
            max_position_pct=max_position_pct,
            max_drawdown_kill=max_drawdown_kill,
            max_concurrent_positions=max_concurrent_positions,
        )
        self.sizer = KellyPositionSizer(max_position_pct=max_position_pct)

        self.equity_curve: list[tuple[pd.Timestamp, float]] = []
        self.trades: list[Trade] = []
        self.signals: list[SignalResult] = []

    def _train_models(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, EnsembleModel]:
        models: Dict[str, EnsembleModel] = {}
        for sym, df in data.items():
            if df is None or df.empty or len(df) < 250:
                logger.warning("Skipping %s: insufficient bars (%d)", sym, len(df))
                continue
            split = max(1, int(len(df) * self.train_fraction))
            train = df.iloc[:split]
            try:
                model = EnsembleModel(
                    confidence_threshold=self.confidence_threshold,
                    seed=self.seed,
                ).fit(train, symbol=sym)
                models[sym] = model
            except Exception as e:
                logger.warning("Failed to train ensemble for %s: %s", sym, e)
        return models

    def _aligned_test_index(
        self, data: Dict[str, pd.DataFrame]
    ) -> pd.DatetimeIndex:
        """Union of test-period timestamps across symbols."""
        idx = pd.DatetimeIndex([])
        for df in data.values():
            split = max(1, int(len(df) * self.train_fraction))
            test = df.iloc[split:]
            idx = idx.union(test.index)
        return idx.sort_values()

    def _trade_stats_from_history(self) -> TradeStats:
        if not self.trades:
            return TradeStats()
        wins = [t.return_pct for t in self.trades if t.pnl > 0]
        losses = [-t.return_pct for t in self.trades if t.pnl <= 0]
        win_rate = len(wins) / max(1, len(self.trades))
        avg_win = sum(wins) / max(1, len(wins)) if wins else 0.01
        avg_loss = sum(losses) / max(1, len(losses)) if losses else 0.01
        return TradeStats(win_rate=win_rate, avg_win=avg_win, avg_loss=avg_loss)

    def run(self, data: Dict[str, pd.DataFrame]) -> dict:
        """Run a backtest across the given symbol -> OHLCV mapping."""
        # 1. Train ensemble models on the in-sample slice.
        models = self._train_models(data)
        if not models:
            logger.warning("No models trained — backtest will be a no-op.")

        # 2. Pre-compute features for the *full* dataset for each symbol.
        feat_engine = FeatureEngine()
        feature_frames: Dict[str, pd.DataFrame] = {}
        for sym, df in data.items():
            if df is None or df.empty:
                continue
            feature_frames[sym] = feat_engine.transform(df)

        # 3. Strategies per symbol.
        strategies: Dict[str, MomentumMeanReversion] = {
            sym: MomentumMeanReversion(
                symbol=sym,
                stop_atr_mult=self.stop_atr_mult,
                take_profit_atr_mult=self.take_profit_atr_mult,
                max_holding_bars=self.max_holding_bars,
                confidence_threshold=self.confidence_threshold,
                regime_filter=self.regime_filter,
                regime_slope_threshold=self.regime_slope_threshold,
                long_only=self.long_only,
            )
            for sym in data
        }

        # 4. Walk forward over the union of test timestamps.
        test_index = self._aligned_test_index(data)
        bar_counters: Dict[str, int] = {sym: 0 for sym in data}

        for ts in test_index:
            mark: Dict[str, float] = {}
            for sym, feat_df in feature_frames.items():
                if ts not in feat_df.index:
                    continue
                row = feat_df.loc[ts]
                mark[sym] = float(row["close"])

            # Mark-to-market and equity update.
            equity = self.risk.update_equity(mark)
            self.equity_curve.append((ts, equity))

            for sym, feat_df in feature_frames.items():
                if ts not in feat_df.index:
                    continue
                row = feat_df.loc[ts]
                price = float(row["close"])
                atr = float(row.get("atr_14", 0.0))
                regime_slope = float(row.get("ema_200_slope", 0.0))
                bar_counters[sym] += 1
                bar_idx = bar_counters[sym]

                # Build the most recent N rows up to (and including) ts for LSTM.
                feat_slice = feat_df.loc[:ts]
                model = models.get(sym)
                if model is None:
                    signal = SignalResult(
                        symbol=sym,
                        direction=0,
                        confidence=0.0,
                        lgbm_score=0.0,
                        lstm_pred=0.0,
                    )
                else:
                    signal = model.predict_one(row, feat_slice)
                self.signals.append(signal)

                strat = strategies[sym]
                decision = strat.on_bar(
                    signal, bar_idx, price, atr, regime_slope=regime_slope
                )

                if decision.action == StrategyAction.CLOSE and strat.position:
                    self._close_trade(
                        sym, ts, price, decision.reason, strat, bar_idx
                    )
                elif decision.action in (
                    StrategyAction.OPEN_LONG,
                    StrategyAction.OPEN_SHORT,
                ):
                    direction = (
                        1 if decision.action == StrategyAction.OPEN_LONG else -1
                    )
                    stats = self._trade_stats_from_history()
                    notional, units = self.sizer.size(
                        equity=self.risk.state.equity,
                        price=price,
                        stats=stats,
                    )
                    risk_decision = self.risk.validate_open(
                        symbol=sym,
                        direction=direction,
                        price=price,
                        notional=notional,
                        units=units,
                    )
                    if not risk_decision.approved:
                        continue
                    self.risk.open_position(
                        symbol=sym,
                        direction=direction,
                        price=price,
                        units=risk_decision.units,
                        bar_index=bar_idx,
                        atr=atr,
                        commission_rate=self.commission_rate,
                        slippage_rate=self.slippage_rate,
                    )
                    strat.open_position(
                        direction=direction,
                        price=price,
                        atr=atr,
                        bar_index=bar_idx,
                        size=risk_decision.units,
                    )
                    self._open_meta[sym] = {
                        "entry_ts": ts,
                        "entry_bar": bar_idx,
                        "entry_confidence": float(signal.confidence),
                        "entry_regime_slope": float(regime_slope),
                    }

        # Close any positions still open at end of test.
        if self.equity_curve:
            last_ts = self.equity_curve[-1][0]
            for sym, strat in list(strategies.items()):
                if strat.position is None:
                    continue
                last_price = mark.get(sym) or strat.position.entry_price
                self._close_trade(
                    sym,
                    last_ts,
                    last_price,
                    "eod_flatten",
                    strat,
                    bar_counters.get(sym, 0),
                )
            # Final equity refresh.
            self.risk.update_equity(mark)
            self.equity_curve[-1] = (last_ts, self.risk.state.equity)

        return self.summary()

    def _close_trade(
        self,
        symbol: str,
        ts: pd.Timestamp,
        price: float,
        reason: str,
        strategy: MomentumMeanReversion,
        close_bar: int,
    ) -> None:
        risk_pos = self.risk.state.positions.get(symbol)
        if risk_pos is None:
            return
        entry_price = risk_pos.entry_price
        units = risk_pos.units
        direction = risk_pos.direction
        pos, pnl = self.risk.close_position(
            symbol=symbol,
            price=price,
            commission_rate=self.commission_rate,
            slippage_rate=self.slippage_rate,
        )
        ret_pct = (
            (price - entry_price) * direction / entry_price
            if entry_price > 0
            else 0.0
        )
        meta = self._open_meta.pop(symbol, {})
        entry_ts = meta.get("entry_ts", ts)
        entry_bar_meta = int(meta.get("entry_bar", risk_pos.entry_bar))
        entry_conf = meta.get("entry_confidence", 0.0)
        entry_slope = meta.get("entry_regime_slope", 0.0)
        bars_held = max(0, int(close_bar) - entry_bar_meta)
        strategy.close_position()
        self.trades.append(
            Trade(
                symbol=symbol,
                entry_time=str(entry_ts),
                exit_time=str(ts),
                direction=direction,
                entry_price=float(entry_price),
                exit_price=float(price),
                units=float(units),
                pnl=float(pnl),
                return_pct=float(ret_pct),
                reason=reason,
                entry_confidence=float(entry_conf),
                entry_regime_slope=float(entry_slope),
                bars_held=int(bars_held),
            )
        )

    def summary(self) -> dict:
        eq_df = pd.DataFrame(self.equity_curve, columns=["ts", "equity"]).set_index(
            "ts"
        )
        return {
            "equity_curve": eq_df,
            "trades": [asdict(t) for t in self.trades],
            "final_equity": float(eq_df["equity"].iloc[-1]) if not eq_df.empty else self.initial_capital,
            "initial_capital": self.initial_capital,
            "n_signals": len(self.signals),
        }
