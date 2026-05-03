"""Phase-15 sizing sweep: hold every strategy/feature/filter knob fixed at
the Phase-14 production config and sweep `max_position_pct` only. The
starting balance is reduced from $100k → $1k so the per-trade dollar
moves are visible at small position sizes.

Sizes: 0.05, 0.10, 0.20, 0.30, 0.40 (i.e. 5 %, 10 %, 20 %, 30 %, 40 %).
Reports total return, max drawdown, DD duration, and final equity for
each. Saves the full table to `backtest/results/sizing_sweep.json` and
prints a clean comparison.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_trader.backtest import Backtester, compute_metrics
from quant_trader.data import DataFetcher, DataPipeline


SIZES = [0.05, 0.10, 0.20, 0.30, 0.40]
INITIAL_CAPITAL = 1_000.0


def _bt(settings: dict, max_pct: float) -> Backtester:
    """Build a Backtester with every Phase-14 production knob held fixed
    except for `max_position_pct`."""
    return Backtester(
        initial_capital=INITIAL_CAPITAL,
        commission_rate=settings.get("commission_rate", 0.001),
        slippage_rate=settings.get("slippage_rate", 0.0005),
        max_position_pct=max_pct,
        max_drawdown_kill=settings.get("max_drawdown_kill", 0.10),
        max_concurrent_positions=settings.get("max_concurrent_positions", 5),
        confidence_threshold=settings.get("signal_confidence_threshold", 0.45),
        stop_atr_mult=settings.get("stop_atr_mult", 1.5),
        take_profit_atr_mult=settings.get("take_profit_atr_mult", 2.5),
        max_holding_bars=settings.get("max_holding_bars", 10),
        seed=settings.get("seed", 42),
        regime_filter=settings.get("regime_filter", True),
        regime_slope_threshold=settings.get("regime_slope_threshold", 0.0),
        long_only=settings.get("long_only", True),
        atr_pct_low=settings.get("atr_pct_low", 0.0),
        atr_pct_high=settings.get("atr_pct_high", 1.0),
        min_agreement_delta=settings.get("min_agreement_delta", 0.05),
    )


def main() -> int:
    settings = yaml.safe_load(open("quant_trader/config/settings.yaml"))
    fetcher = DataFetcher()
    pipe = DataPipeline(fetcher)
    data = pipe.fetch_all(
        universe=settings["universe"],
        timeframe=settings["timeframe"],
        start=settings["backtest_start"],
        end=datetime.now(timezone.utc),
    )

    rows: list[dict] = []
    for size in SIZES:
        bt = _bt(settings, size)
        bt_summary = bt.run(data)
        m = compute_metrics(
            bt_summary,
            risk_free_rate=settings.get("risk_free_rate", 0.05),
            timeframe=settings["timeframe"],
        )
        m_rf0 = compute_metrics(
            bt_summary, risk_free_rate=0.0, timeframe=settings["timeframe"]
        )
        rows.append({
            "max_position_pct": size,
            "initial_capital": INITIAL_CAPITAL,
            "n_trades": m["n_trades"],
            "win_rate": m["win_rate"],
            "profit_factor": m["profit_factor"],
            "total_return": m["total_return"],
            "max_drawdown": m["max_drawdown"],
            "max_drawdown_duration_days": m["max_drawdown_duration_days"],
            "final_equity": m["final_equity"],
            "sharpe_tw_rf0": m_rf0["sharpe_trade_weighted"],
        })

    # Pretty-print comparison table.
    header = (
        f"{'max_pos_%':>10} | {'trades':>6} | {'PF':>5} | "
        f"{'total_ret':>9} | {'max_dd':>7} | {'dd_days':>7} | "
        f"{'final_eq':>10}"
    )
    sep = "-" * len(header)
    print("\nPhase-15 sizing sweep ($1,000 starting balance, all other "
          "knobs held at Phase-14 production)")
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['max_position_pct']:>9.2%} | "
            f"{r['n_trades']:>6d} | "
            f"{r['profit_factor']:>5.2f} | "
            f"{r['total_return']:>+9.2%} | "
            f"{r['max_drawdown']:>+7.2%} | "
            f"{r['max_drawdown_duration_days']:>7.0f} | "
            f"${r['final_equity']:>9,.2f}"
        )
    print(sep)

    out = Path("backtest/results/sizing_sweep.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "initial_capital": INITIAL_CAPITAL,
            "universe": settings["universe"],
            "timeframe": settings["timeframe"],
            "signal_confidence_threshold": settings["signal_confidence_threshold"],
            "regime_filter": settings.get("regime_filter"),
            "long_only": settings.get("long_only"),
            "atr_pct_low": settings.get("atr_pct_low"),
            "atr_pct_high": settings.get("atr_pct_high"),
            "min_agreement_delta": settings.get("min_agreement_delta"),
            "max_drawdown_kill": settings.get("max_drawdown_kill"),
        },
        "results": rows,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
