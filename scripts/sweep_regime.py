"""Tighten / loosen the regime slope threshold to see if any value of the
filter improves on the best long-only run from the main sweep."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_trader.backtest import Backtester, compute_metrics
from quant_trader.data import DataFetcher, DataPipeline


SLOPES = [-0.05, -0.02, -0.01, 0.0, 0.01, 0.02]
THRESHOLD = 0.70


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

    rows = []
    for slope in SLOPES:
        bt = Backtester(
            initial_capital=settings.get("initial_capital", 100_000),
            commission_rate=settings.get("commission_rate", 0.001),
            slippage_rate=settings.get("slippage_rate", 0.0005),
            max_position_pct=settings.get("max_position_pct", 0.02),
            max_drawdown_kill=settings.get("max_drawdown_kill", 0.10),
            max_concurrent_positions=settings.get("max_concurrent_positions", 5),
            confidence_threshold=THRESHOLD,
            stop_atr_mult=settings.get("stop_atr_mult", 1.5),
            take_profit_atr_mult=settings.get("take_profit_atr_mult", 2.5),
            max_holding_bars=settings.get("max_holding_bars", 10),
            seed=settings.get("seed", 42),
            regime_filter=True,
            regime_slope_threshold=slope,
            long_only=True,
        )
        summary = bt.run(data)
        m = compute_metrics(
            summary,
            risk_free_rate=settings.get("risk_free_rate", 0.05),
            timeframe=settings["timeframe"],
        )
        rows.append({"slope": slope, **m})
        print(
            f"slope={slope:+.3f}  n={m['n_trades']:4d}  "
            f"wr={m['win_rate']:.2%}  PF={m['profit_factor']:.2f}  "
            f"Sharpe={m['sharpe']:.2f}  ret={m['total_return']:.2%}"
        )

    Path("backtest/results/regime_slope_sweep.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
