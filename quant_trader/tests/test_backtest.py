"""End-to-end mini backtest on synthetic 30-day hourly data."""

from __future__ import annotations

from quant_trader.backtest import Backtester, compute_metrics


def test_backtest_runs_end_to_end(synthetic_ohlcv):
    bt = Backtester(initial_capital=100_000, train_fraction=0.6)
    summary = bt.run({"BTC-USD": synthetic_ohlcv})

    metrics = compute_metrics(summary, timeframe="1h")
    expected_keys = {
        "total_return",
        "cagr",
        "sharpe",
        "sortino",
        "max_drawdown",
        "max_drawdown_duration_days",
        "win_rate",
        "profit_factor",
        "avg_win",
        "avg_loss",
        "n_trades",
        "final_equity",
    }
    assert expected_keys.issubset(metrics.keys())
    # Backtester should have produced an equity curve and a metrics blob.
    assert summary["final_equity"] > 0
