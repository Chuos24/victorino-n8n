"""Momentum / mean-reversion strategy driven by the ensemble signal."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..models.ensemble import SignalResult


class StrategyAction(str, Enum):
    HOLD = "hold"
    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    CLOSE = "close"


@dataclass
class Position:
    """Lightweight position record managed by the strategy."""

    symbol: str
    direction: int  # +1 long / -1 short
    entry_price: float
    entry_atr: float
    entry_bar: int
    size: float = 0.0


@dataclass
class StrategyDecision:
    action: StrategyAction
    reason: str = ""


class MomentumMeanReversion:
    """Per-symbol strategy. Holds a single open position at a time.

    Entry: ensemble produces a non-zero direction with sufficient confidence.
    Exits (any of):
      * Opposite signal fires.
      * Stop-loss: entry - ATR * stop_atr_mult (long) / + (short)
      * Take-profit: entry + ATR * take_profit_atr_mult (long) / - (short)
      * Time-stop: max_holding_bars elapsed.
    """

    def __init__(
        self,
        symbol: str,
        stop_atr_mult: float = 1.5,
        take_profit_atr_mult: float = 2.5,
        max_holding_bars: int = 48,
        confidence_threshold: float = 0.6,
    ):
        self.symbol = symbol
        self.stop_atr_mult = stop_atr_mult
        self.take_profit_atr_mult = take_profit_atr_mult
        self.max_holding_bars = max_holding_bars
        self.confidence_threshold = confidence_threshold
        self.position: Position | None = None

    def stop_price(self) -> float | None:
        if self.position is None:
            return None
        offset = self.position.entry_atr * self.stop_atr_mult
        return self.position.entry_price - offset * self.position.direction

    def target_price(self) -> float | None:
        if self.position is None:
            return None
        offset = self.position.entry_atr * self.take_profit_atr_mult
        return self.position.entry_price + offset * self.position.direction

    def on_bar(
        self,
        signal: SignalResult,
        bar_index: int,
        price: float,
        atr: float,
    ) -> StrategyDecision:
        # If position is open, check exits first.
        if self.position is not None:
            held = bar_index - self.position.entry_bar
            stop = self.stop_price()
            target = self.target_price()
            if self.position.direction == 1:
                if stop is not None and price <= stop:
                    return StrategyDecision(StrategyAction.CLOSE, "stop_loss")
                if target is not None and price >= target:
                    return StrategyDecision(StrategyAction.CLOSE, "take_profit")
            else:
                if stop is not None and price >= stop:
                    return StrategyDecision(StrategyAction.CLOSE, "stop_loss")
                if target is not None and price <= target:
                    return StrategyDecision(StrategyAction.CLOSE, "take_profit")
            if held >= self.max_holding_bars:
                return StrategyDecision(StrategyAction.CLOSE, "time_stop")
            if (
                signal.direction != 0
                and signal.direction != self.position.direction
                and signal.confidence >= self.confidence_threshold
            ):
                return StrategyDecision(StrategyAction.CLOSE, "opposite_signal")
            return StrategyDecision(StrategyAction.HOLD, "in_position")

        # No position open — consider entry.
        if signal.direction == 0 or signal.confidence < self.confidence_threshold:
            return StrategyDecision(StrategyAction.HOLD, "no_signal")
        if atr <= 0:
            return StrategyDecision(StrategyAction.HOLD, "no_atr")
        action = (
            StrategyAction.OPEN_LONG if signal.direction == 1 else StrategyAction.OPEN_SHORT
        )
        return StrategyDecision(action, "ensemble_signal")

    def open_position(
        self, direction: int, price: float, atr: float, bar_index: int, size: float
    ) -> Position:
        self.position = Position(
            symbol=self.symbol,
            direction=direction,
            entry_price=price,
            entry_atr=atr,
            entry_bar=bar_index,
            size=size,
        )
        return self.position

    def close_position(self) -> Position | None:
        pos, self.position = self.position, None
        return pos
