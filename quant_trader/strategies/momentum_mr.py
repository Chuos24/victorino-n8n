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
        regime_filter: bool = False,
        regime_slope_threshold: float = 0.0,
        long_only: bool = False,
        atr_pct_low: float = 0.0,
        atr_pct_high: float = 1.0,
        min_agreement_delta: float = 0.0,
    ):
        self.symbol = symbol
        self.stop_atr_mult = stop_atr_mult
        self.take_profit_atr_mult = take_profit_atr_mult
        self.max_holding_bars = max_holding_bars
        self.confidence_threshold = confidence_threshold
        self.regime_filter = regime_filter
        self.regime_slope_threshold = regime_slope_threshold
        self.long_only = long_only
        self.atr_pct_low = atr_pct_low
        self.atr_pct_high = atr_pct_high
        self.min_agreement_delta = min_agreement_delta
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
        regime_slope: float | None = None,
        atr_pct: float | None = None,
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
        # Regime filter: only take trades when 200-EMA slope is above the
        # threshold (i.e. trending up). Sideways/down regimes are skipped.
        if self.regime_filter and regime_slope is not None:
            if regime_slope <= self.regime_slope_threshold:
                return StrategyDecision(StrategyAction.HOLD, "regime_block")
        # Long-only override: trade-log analysis showed shorts have negative
        # expectancy (PF 0.67) while longs are profitable (PF 1.13). When
        # `long_only` is set, drop short signals entirely.
        if self.long_only and signal.direction == -1:
            return StrategyDecision(StrategyAction.HOLD, "long_only_block")
        # ATR percentile filter: only enter when current ATR sits inside the
        # configured percentile band of its trailing 252-bar range. Very
        # low ATR is chop / whipsaw territory; very high ATR signals regime
        # breaks where momentum signals tend to misfire.
        if atr_pct is not None and (
            self.atr_pct_low > 0.0 or self.atr_pct_high < 1.0
        ):
            if atr_pct < self.atr_pct_low or atr_pct > self.atr_pct_high:
                return StrategyDecision(StrategyAction.HOLD, "atr_band_block")
        # Agreement delta: require LGBM's top-class probability to lead the
        # runner-up by at least `min_agreement_delta`. Near-tie predictions
        # (e.g. 0.34/0.33/0.33 across UP/FLAT/DOWN) get filtered out.
        if signal.agreement_delta < self.min_agreement_delta:
            return StrategyDecision(StrategyAction.HOLD, "low_agreement")
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
