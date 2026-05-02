"""Portfolio-aware risk manager and order validator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class OpenPosition:
    symbol: str
    direction: int
    entry_price: float
    units: float
    entry_bar: int = 0
    entry_atr: float = 0.0


@dataclass
class PortfolioState:
    """Equity = cash + unrealized PnL.

    `cash` represents realised capital. When a position opens we pay only the
    entry-side commission/slippage cost from cash; the principal stays
    "earmarked" but not removed (it is recovered on exit). This keeps the
    accounting simple while still penalising trades for fees.
    """

    cash: float
    equity: float
    peak_equity: float
    positions: Dict[str, OpenPosition] = field(default_factory=dict)
    kill_switch: bool = False

    def drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, 1 - self.equity / self.peak_equity)


@dataclass
class RiskDecision:
    approved: bool
    reason: str
    size: float = 0.0
    units: float = 0.0


class RiskManager:
    """Validate every order against portfolio rules."""

    def __init__(
        self,
        initial_capital: float = 100_000,
        max_position_pct: float = 0.02,
        max_drawdown_kill: float = 0.10,
        max_concurrent_positions: int = 5,
    ):
        self.initial_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.max_drawdown_kill = max_drawdown_kill
        self.max_concurrent_positions = max_concurrent_positions
        self.state = PortfolioState(
            cash=initial_capital,
            equity=initial_capital,
            peak_equity=initial_capital,
        )

    def unrealized_pnl(self, mark_to_market: Dict[str, float]) -> float:
        total = 0.0
        for sym, pos in self.state.positions.items():
            price = mark_to_market.get(sym)
            if price is None:
                continue
            total += (price - pos.entry_price) * pos.direction * pos.units
        return total

    def update_equity(self, mark_to_market: Dict[str, float]) -> float:
        self.state.equity = self.state.cash + self.unrealized_pnl(mark_to_market)
        if self.state.equity > self.state.peak_equity:
            self.state.peak_equity = self.state.equity
        if self.state.drawdown() >= self.max_drawdown_kill:
            self.state.kill_switch = True
        return self.state.equity

    def validate_open(
        self,
        symbol: str,
        direction: int,
        price: float,
        notional: float,
        units: float,
    ) -> RiskDecision:
        if self.state.kill_switch:
            return RiskDecision(False, "kill_switch_active")
        if symbol in self.state.positions:
            return RiskDecision(False, "already_open")
        if len(self.state.positions) >= self.max_concurrent_positions:
            return RiskDecision(False, "max_positions_reached")
        if direction not in (-1, 1):
            return RiskDecision(False, "invalid_direction")
        if price <= 0 or units <= 0:
            return RiskDecision(False, "invalid_size")
        cap = self.state.equity * self.max_position_pct
        if notional > cap:
            notional = cap
            units = cap / price
        return RiskDecision(True, "ok", size=notional, units=units)

    def open_position(
        self,
        symbol: str,
        direction: int,
        price: float,
        units: float,
        bar_index: int = 0,
        atr: float = 0.0,
        commission_rate: float = 0.0,
        slippage_rate: float = 0.0,
    ) -> OpenPosition:
        fill_price = price * (1 + slippage_rate * direction)
        # Charge commission only.
        notional = fill_price * units
        self.state.cash -= notional * commission_rate
        pos = OpenPosition(
            symbol=symbol,
            direction=direction,
            entry_price=fill_price,
            units=units,
            entry_bar=bar_index,
            entry_atr=atr,
        )
        self.state.positions[symbol] = pos
        return pos

    def close_position(
        self,
        symbol: str,
        price: float,
        commission_rate: float = 0.0,
        slippage_rate: float = 0.0,
    ) -> tuple[OpenPosition | None, float]:
        pos = self.state.positions.pop(symbol, None)
        if pos is None:
            return None, 0.0
        fill_price = price * (1 - slippage_rate * pos.direction)
        pnl = (fill_price - pos.entry_price) * pos.direction * pos.units
        notional = fill_price * pos.units
        self.state.cash += pnl - notional * commission_rate
        return pos, pnl
