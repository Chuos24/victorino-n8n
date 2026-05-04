"""Run the canonical backtest at the production settings, then re-compute
metrics with risk_free_rate=0 as a secondary view. Saves both metric sets
into `backtest/results/latest_run.json` (primary) and
`backtest/results/latest_run_rf0.json` (rf=0 sanity check).
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_trader.backtest import Backtester, compute_metrics, render_metrics_table
from quant_trader.backtest.analytics import save_results
from quant_trader.data import DataFetcher, DataPipeline


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

    bt = Backtester(
        initial_capital=settings.get("initial_capital", 100_000),
        commission_rate=settings.get("commission_rate", 0.001),
        slippage_rate=settings.get("slippage_rate", 0.0005),
        max_position_pct=settings.get("max_position_pct", 0.02),
        max_drawdown_kill=settings.get("max_drawdown_kill", 0.10),
        max_concurrent_positions=settings.get("max_concurrent_positions", 5),
        confidence_threshold=settings.get("signal_confidence_threshold", 0.72),
        stop_atr_mult=settings.get("stop_atr_mult", 1.5),
        take_profit_atr_mult=settings.get("take_profit_atr_mult", 2.5),
        max_holding_bars=settings.get("max_holding_bars", 10),
        seed=settings.get("seed", 42),
        regime_filter=settings.get("regime_filter", False),
        regime_slope_threshold=settings.get("regime_slope_threshold", 0.0),
        long_only=settings.get("long_only", False),
        atr_pct_low=settings.get("atr_pct_low", 0.0),
        atr_pct_high=settings.get("atr_pct_high", 1.0),
        min_agreement_delta=settings.get("min_agreement_delta", 0.0),
    )
    summary = bt.run(data)

    rf_primary = settings.get("risk_free_rate", 0.05)
    metrics_primary = compute_metrics(summary, risk_free_rate=rf_primary, timeframe=settings["timeframe"])
    metrics_rf0 = compute_metrics(summary, risk_free_rate=0.0, timeframe=settings["timeframe"])

    print(f"\n=== Primary metrics (rf={rf_primary}) ===")
    render_metrics_table(metrics_primary)
    print(f"\n=== Secondary metrics (rf=0.0) ===")
    render_metrics_table(metrics_rf0)

    save_results(summary, metrics_primary, Path("backtest/results/latest_run.json"))
    save_results(summary, metrics_primary, Path("quant_trader/backtest/results/latest_run.json"))
    save_results(summary, metrics_rf0, Path("backtest/results/latest_run_rf0.json"))
    print(
        "\nSaved backtest/results/latest_run.json (primary) and "
        "backtest/results/latest_run_rf0.json (rf=0)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
