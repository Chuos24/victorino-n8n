"""Kelly-criterion position sizing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TradeStats:
    """Aggregate stats used to derive Kelly fraction."""

    win_rate: float = 0.5
    avg_win: float = 0.01
    avg_loss: float = 0.01

    @property
    def b(self) -> float:
        if self.avg_loss <= 0:
            return 1.0
        return self.avg_win / self.avg_loss


class KellyPositionSizer:
    """Half-Kelly sizing capped at `max_position_pct` of equity."""

    def __init__(
        self,
        max_position_pct: float = 0.02,
        kelly_scale: float = 0.5,
        min_pct: float = 0.001,
    ):
        self.max_position_pct = max_position_pct
        self.kelly_scale = kelly_scale
        self.min_pct = min_pct

    def kelly_fraction(self, stats: TradeStats) -> float:
        b = stats.b
        if b <= 0:
            return 0.0
        p = max(0.0, min(1.0, stats.win_rate))
        q = 1 - p
        f = (b * p - q) / b
        return max(0.0, f) * self.kelly_scale

    def size(
        self, equity: float, price: float, stats: TradeStats
    ) -> tuple[float, float]:
        """Return (notional_dollars, units)."""
        if equity <= 0 or price <= 0:
            return 0.0, 0.0
        f = self.kelly_fraction(stats)
        pct = max(self.min_pct, min(f, self.max_position_pct))
        notional = equity * pct
        units = notional / price
        return notional, units
