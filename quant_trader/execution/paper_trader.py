"""Paper trading loop: poll, score, simulate fills."""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..data import DataFetcher
from ..features import FeatureEngine
from ..models import EnsembleModel, SignalResult
from ..risk import KellyPositionSizer, RiskManager
from ..risk.sizer import TradeStats
from ..strategies import MomentumMeanReversion, StrategyAction
from .alerts import Alerter

logger = logging.getLogger(__name__)

STATE_PATH = Path(__file__).parent / "portfolio_state.json"
TRADE_LOG_PATH = Path(__file__).parent / "trade_log.csv"

TRADE_LOG_FIELDS = [
    "timestamp",
    "symbol",
    "action",
    "direction",
    "price",
    "units",
    "pnl",
    "reason",
    "equity",
]


@dataclass
class TraderConfig:
    universe: List[str]
    timeframe: str = "1d"
    poll_interval: int = 60
    initial_capital: float = 100_000
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    max_position_pct: float = 0.02
    max_drawdown_kill: float = 0.10
    max_concurrent_positions: int = 5
    confidence_threshold: float = 0.6
    stop_atr_mult: float = 1.5
    take_profit_atr_mult: float = 2.5
    max_holding_bars: int = 10
    # Daily timeframe needs a long history: feature engine warmup is ~200
    # bars, models want at least 250 training rows on top of that.
    train_days: int = 1460
    step_lookback_days: int = 540
    seed: int = 42


class PaperTrader:
    """Paper trading loop with persistent state."""

    def __init__(
        self,
        config: TraderConfig,
        fetcher: DataFetcher | None = None,
        alerter: Alerter | None = None,
        state_path: Path | str = STATE_PATH,
        trade_log_path: Path | str = TRADE_LOG_PATH,
    ):
        self.config = config
        self.fetcher = fetcher or DataFetcher()
        self.alerter = alerter or Alerter()
        self.state_path = Path(state_path)
        self.trade_log_path = Path(trade_log_path)
        self.feat_engine = FeatureEngine()

        self.risk = RiskManager(
            initial_capital=config.initial_capital,
            max_position_pct=config.max_position_pct,
            max_drawdown_kill=config.max_drawdown_kill,
            max_concurrent_positions=config.max_concurrent_positions,
        )
        self.sizer = KellyPositionSizer(max_position_pct=config.max_position_pct)
        self.models: Dict[str, EnsembleModel] = {}
        self.strategies: Dict[str, MomentumMeanReversion] = {}
        self.last_seen_bar: Dict[str, pd.Timestamp] = {}
        self.bar_counters: Dict[str, int] = {sym: 0 for sym in config.universe}
        self.trades_history: List[Dict] = []
        self.signal_history: List[SignalResult] = []
        self._ensure_trade_log()
        self._load_state()

    # ----------------------- state persistence -----------------------
    def _ensure_trade_log(self) -> None:
        self.trade_log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.trade_log_path.exists():
            with open(self.trade_log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=TRADE_LOG_FIELDS)
                writer.writeheader()

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            with open(self.state_path, "r") as f:
                payload = json.load(f)
            self.risk.state.cash = payload.get("cash", self.risk.state.cash)
            self.risk.state.equity = payload.get("equity", self.risk.state.equity)
            self.risk.state.peak_equity = payload.get(
                "peak_equity", self.risk.state.peak_equity
            )
            self.risk.state.kill_switch = payload.get("kill_switch", False)
            for sym, pos in payload.get("positions", {}).items():
                from ..risk.manager import OpenPosition

                self.risk.state.positions[sym] = OpenPosition(**pos)
        except Exception as e:
            logger.warning("Failed to load state: %s", e)

    def save_state(self) -> None:
        payload = {
            "cash": self.risk.state.cash,
            "equity": self.risk.state.equity,
            "peak_equity": self.risk.state.peak_equity,
            "kill_switch": self.risk.state.kill_switch,
            "positions": {
                sym: asdict(pos) for sym, pos in self.risk.state.positions.items()
            },
            "drawdown": self.risk.state.drawdown(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(payload, f, indent=2)

    def _append_trade_log(self, row: Dict) -> None:
        with open(self.trade_log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TRADE_LOG_FIELDS)
            writer.writerow({k: row.get(k, "") for k in TRADE_LOG_FIELDS})

    # ----------------------- model training -----------------------
    def warmup(self) -> None:
        """Fetch training data and fit one ensemble per symbol."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=self.config.train_days)
        # Pre-fetch every universe frame so cross-asset features (e.g.
        # BTC 7d return) are available during model fits.
        cross_assets: Dict[str, pd.DataFrame] = {}
        for sym in self.config.universe:
            try:
                cross_assets[sym] = self.fetcher.get_bars(
                    sym, self.config.timeframe, start=start, end=end
                )
            except Exception as e:
                self.alerter.error(f"Cross-asset fetch failed for {sym}: {e}")
                cross_assets[sym] = pd.DataFrame()

        for sym in self.config.universe:
            self.strategies.setdefault(
                sym,
                MomentumMeanReversion(
                    symbol=sym,
                    stop_atr_mult=self.config.stop_atr_mult,
                    take_profit_atr_mult=self.config.take_profit_atr_mult,
                    max_holding_bars=self.config.max_holding_bars,
                    confidence_threshold=self.config.confidence_threshold,
                ),
            )
            try:
                df = cross_assets.get(sym, pd.DataFrame())
                if df.empty or len(df) < 250:
                    self.alerter.warn(
                        f"Skip {sym}: insufficient training data ({len(df)} bars)"
                    )
                    continue
                model = EnsembleModel(
                    confidence_threshold=self.config.confidence_threshold,
                    seed=self.config.seed,
                ).fit(df, symbol=sym, cross_assets=cross_assets)
                self.models[sym] = model
                # Stash so step() can rebuild features with the same
                # cross-asset context used at training time.
                self._cross_assets = cross_assets
                # Intentionally don't seed last_seen_bar here: we want the
                # first step() after warmup to score the latest bar as if
                # it were new, so the loop emits at least one signal even
                # when no fresh bar has printed since startup.
                self.alerter.signal(f"Model trained for {sym} ({len(df)} bars)")
            except Exception as e:
                self.alerter.error(f"Training failed for {sym}: {e}")

    # ----------------------- main loop -----------------------
    def step(self) -> None:
        """Single iteration: poll bars, score, decide, log."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=self.config.step_lookback_days)
        mark: Dict[str, float] = {}

        for sym in self.config.universe:
            try:
                df = self.fetcher.get_bars(sym, self.config.timeframe, start=start, end=end)
            except Exception as e:
                self.alerter.error(f"Fetch failed for {sym}: {e}")
                continue
            if df.empty:
                continue

            cross = getattr(self, "_cross_assets", None) or {}
            feat = FeatureEngine(cross_assets=cross).transform(df)
            if feat.empty:
                continue
            latest_ts = feat.index[-1]
            mark[sym] = float(feat["close"].iloc[-1])

            if self.last_seen_bar.get(sym) == latest_ts:
                # No new bar — still mark to market but skip signal handling.
                continue
            self.last_seen_bar[sym] = latest_ts
            self.bar_counters[sym] = self.bar_counters.get(sym, 0) + 1

            row = feat.iloc[-1]
            price = float(row["close"])
            atr = float(row.get("atr_14", 0.0))

            model = self.models.get(sym)
            if model is None:
                continue
            signal = model.predict_one(row, feat)
            self.signal_history.append(signal)
            if signal.direction != 0:
                self.alerter.signal(
                    f"{sym} dir={signal.direction:+d} "
                    f"conf={signal.confidence:.2f} lstm={signal.lstm_pred:+.4f}"
                )

            strat = self.strategies[sym]
            decision = strat.on_bar(signal, self.bar_counters[sym], price, atr)

            if decision.action == StrategyAction.CLOSE and strat.position is not None:
                self._close_trade(sym, latest_ts, price, decision.reason, strat)
            elif decision.action in (
                StrategyAction.OPEN_LONG,
                StrategyAction.OPEN_SHORT,
            ):
                direction = (
                    1 if decision.action == StrategyAction.OPEN_LONG else -1
                )
                self._open_trade(sym, latest_ts, price, atr, direction, strat)

        equity = self.risk.update_equity(mark)
        if self.risk.state.kill_switch:
            self.alerter.warn(
                f"KILL SWITCH ACTIVE: drawdown={self.risk.state.drawdown():.2%}"
            )
        self.save_state()

    def _open_trade(
        self,
        symbol: str,
        ts,
        price: float,
        atr: float,
        direction: int,
        strategy: MomentumMeanReversion,
    ) -> None:
        stats = self._stats_from_history()
        notional, units = self.sizer.size(
            equity=self.risk.state.equity, price=price, stats=stats
        )
        decision = self.risk.validate_open(
            symbol=symbol,
            direction=direction,
            price=price,
            notional=notional,
            units=units,
        )
        if not decision.approved:
            self.alerter.warn(f"Order rejected for {symbol}: {decision.reason}")
            return
        self.risk.open_position(
            symbol=symbol,
            direction=direction,
            price=price,
            units=decision.units,
            bar_index=self.bar_counters.get(symbol, 0),
            atr=atr,
            commission_rate=self.config.commission_rate,
            slippage_rate=self.config.slippage_rate,
        )
        strategy.open_position(
            direction=direction,
            price=price,
            atr=atr,
            bar_index=self.bar_counters.get(symbol, 0),
            size=decision.units,
        )
        self.alerter.trade(
            f"OPEN {symbol} {'LONG' if direction == 1 else 'SHORT'} "
            f"{decision.units:.4f}@{price:.2f}"
        )
        self._append_trade_log(
            {
                "timestamp": str(ts),
                "symbol": symbol,
                "action": "OPEN",
                "direction": direction,
                "price": price,
                "units": decision.units,
                "pnl": "",
                "reason": "ensemble_signal",
                "equity": self.risk.state.equity,
            }
        )

    def _close_trade(
        self,
        symbol: str,
        ts,
        price: float,
        reason: str,
        strategy: MomentumMeanReversion,
    ) -> None:
        pos, pnl = self.risk.close_position(
            symbol=symbol,
            price=price,
            commission_rate=self.config.commission_rate,
            slippage_rate=self.config.slippage_rate,
        )
        strategy.close_position()
        if pos is None:
            return
        self.trades_history.append(
            {
                "symbol": symbol,
                "exit_time": str(ts),
                "pnl": pnl,
                "direction": pos.direction,
                "entry_price": pos.entry_price,
                "exit_price": price,
                "units": pos.units,
                "reason": reason,
            }
        )
        self.alerter.trade(
            f"CLOSE {symbol} pnl={pnl:+,.2f} reason={reason}"
        )
        self._append_trade_log(
            {
                "timestamp": str(ts),
                "symbol": symbol,
                "action": "CLOSE",
                "direction": pos.direction,
                "price": price,
                "units": pos.units,
                "pnl": pnl,
                "reason": reason,
                "equity": self.risk.state.equity,
            }
        )

    def _stats_from_history(self) -> TradeStats:
        if not self.trades_history:
            return TradeStats()
        wins = [t["pnl"] for t in self.trades_history if t["pnl"] > 0]
        losses = [-t["pnl"] for t in self.trades_history if t["pnl"] <= 0]
        win_rate = len(wins) / max(1, len(self.trades_history))
        avg_win = sum(wins) / max(1, len(wins)) if wins else 0.01
        avg_loss = sum(losses) / max(1, len(losses)) if losses else 0.01
        return TradeStats(win_rate=win_rate, avg_win=avg_win, avg_loss=avg_loss)

    def run_forever(self) -> None:  # pragma: no cover - blocking loop
        self.alerter.signal("Paper trader starting up.")
        self.warmup()
        while True:
            try:
                self.step()
            except KeyboardInterrupt:
                self.alerter.warn("Paper trader stopping (KeyboardInterrupt)")
                self.save_state()
                return
            except Exception as e:
                self.alerter.error(f"Loop error: {e}")
            time.sleep(self.config.poll_interval)
