"""CLI entry point for the AI Quant Trader."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from quant_trader.config import load_settings
from quant_trader.data import DataFetcher, DataPipeline


def cmd_fetch(settings: dict) -> int:
    fetcher = DataFetcher()
    pipe = DataPipeline(fetcher)
    results = pipe.fetch_all(
        universe=settings["universe"],
        timeframe=settings["timeframe"],
        start=settings["backtest_start"],
        end=datetime.now(timezone.utc),
    )
    for sym, df in results.items():
        print(f"{sym}: {len(df)} bars")
    return 0


def cmd_backtest(settings: dict) -> int:
    from quant_trader.backtest import (
        Backtester,
        compute_metrics,
        render_metrics_table,
    )
    from quant_trader.backtest.analytics import save_results

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
        confidence_threshold=settings.get("signal_confidence_threshold", 0.6),
        stop_atr_mult=settings.get("stop_atr_mult", 1.5),
        take_profit_atr_mult=settings.get("take_profit_atr_mult", 2.5),
        max_holding_bars=settings.get("max_holding_bars", 48),
        seed=settings.get("seed", 42),
    )
    summary = bt.run(data)
    metrics = compute_metrics(
        summary,
        risk_free_rate=settings.get("risk_free_rate", 0.05),
        timeframe=settings["timeframe"],
    )
    render_metrics_table(metrics)
    out = Path(__file__).parent / "quant_trader" / "backtest" / "results" / "latest_run.json"
    save_results(summary, metrics, out)
    print(f"\nResults saved to {out}")
    return 0


def cmd_paper(settings: dict, once: bool = False) -> int:
    from quant_trader.execution.paper_trader import PaperTrader, TraderConfig

    cfg = TraderConfig(
        universe=settings["universe"],
        timeframe=settings["timeframe"],
        poll_interval=settings.get("poll_interval", 60),
        initial_capital=settings.get("initial_capital", 100_000),
        commission_rate=settings.get("commission_rate", 0.001),
        slippage_rate=settings.get("slippage_rate", 0.0005),
        max_position_pct=settings.get("max_position_pct", 0.02),
        max_drawdown_kill=settings.get("max_drawdown_kill", 0.10),
        max_concurrent_positions=settings.get("max_concurrent_positions", 5),
        confidence_threshold=settings.get("signal_confidence_threshold", 0.6),
        stop_atr_mult=settings.get("stop_atr_mult", 1.5),
        take_profit_atr_mult=settings.get("take_profit_atr_mult", 2.5),
        max_holding_bars=settings.get("max_holding_bars", 10),
        seed=settings.get("seed", 42),
    )
    trader = PaperTrader(cfg)
    if once:
        trader.alerter.signal("Paper trader (one-shot) starting up.")
        trader.warmup()
        trader.step()
        trader.save_state()
        signals = [s for s in trader.signal_history if s.direction != 0]
        print(
            f"\nOne-shot complete. Bars seen: {sum(trader.bar_counters.values())}, "
            f"non-zero signals: {len(signals)}, "
            f"open positions: {len(trader.risk.state.positions)}, "
            f"equity=${trader.risk.state.equity:,.2f}"
        )
        for s in signals[:10]:
            print(
                f"  {s.symbol}: dir={s.direction:+d} conf={s.confidence:.3f} "
                f"lstm={s.lstm_pred:+.4f}"
            )
        return 0
    trader.run_forever()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="AI Quant Trader")
    parser.add_argument(
        "--mode",
        choices=["fetch", "backtest", "paper"],
        required=True,
        help="What to do.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Override path to settings.yaml",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="For paper mode: warm up, run a single step, then exit.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = load_settings(args.config)
    if args.mode == "fetch":
        return cmd_fetch(settings)
    if args.mode == "backtest":
        return cmd_backtest(settings)
    if args.mode == "paper":
        return cmd_paper(settings, once=args.once)
    return 1


if __name__ == "__main__":
    sys.exit(main())
