"""Tests for the risk manager and position sizer."""

from __future__ import annotations

from quant_trader.risk import KellyPositionSizer, RiskManager
from quant_trader.risk.sizer import TradeStats


def test_position_cap_enforced():
    rm = RiskManager(initial_capital=100_000, max_position_pct=0.02)
    # Try to open a position twice the cap.
    decision = rm.validate_open(
        symbol="X",
        direction=1,
        price=100.0,
        notional=10_000,  # 10% of equity, way over 2% cap
        units=100.0,
    )
    assert decision.approved
    # validate_open scales down: notional should be at most 2% of equity.
    assert decision.size <= 100_000 * 0.02 + 1e-6


def test_kill_switch_at_10pct_drawdown():
    rm = RiskManager(initial_capital=100_000, max_drawdown_kill=0.10)
    rm.state.peak_equity = 100_000
    rm.state.cash = 89_000  # 11% drawdown
    rm.update_equity({})
    assert rm.state.kill_switch
    decision = rm.validate_open(
        symbol="Y", direction=1, price=10.0, notional=100, units=10
    )
    assert not decision.approved
    assert decision.reason == "kill_switch_active"


def test_max_concurrent_positions():
    rm = RiskManager(initial_capital=100_000, max_concurrent_positions=2)
    for sym in ["A", "B"]:
        rm.open_position(symbol=sym, direction=1, price=10.0, units=1.0)
    decision = rm.validate_open(
        symbol="C", direction=1, price=10.0, notional=10, units=1
    )
    assert not decision.approved
    assert decision.reason == "max_positions_reached"


def test_kelly_sizer_capped():
    sizer = KellyPositionSizer(max_position_pct=0.02)
    stats = TradeStats(win_rate=0.99, avg_win=1.0, avg_loss=0.01)  # huge edge
    notional, units = sizer.size(equity=100_000, price=100.0, stats=stats)
    # Hard cap at 2%.
    assert notional <= 100_000 * 0.02 + 1e-6
