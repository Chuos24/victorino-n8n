"""Sweep ATR percentile band × min_agreement_delta × confidence threshold
on the long-only configuration. Applies the trade-weighted Sharpe (rf=0 and
rf=0.05) so the metric isn't dominated by the all-bar / risk-free-rate bug.

Selection rule: keep runs with n_trades >= 50 and profit_factor > 1.2; pick
the highest trade-weighted Sharpe (rf=0). Falls back to highest PF if no
qualifier exists.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_trader.backtest import Backtester, compute_metrics
from quant_trader.backtest.analytics import save_results
from quant_trader.data import DataFetcher, DataPipeline


MIN_TRADES = 50
THRESHOLDS = [0.65, 0.70, 0.72, 0.75]
ATR_BANDS = [
    (0.0, 1.0),
    (0.10, 0.90),
    (0.20, 0.80),
    (0.30, 0.70),
    (0.30, 1.00),
    (0.00, 0.70),
]
AGREEMENTS = [0.0, 0.05, 0.10]


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

    runs: dict = {}
    rows: list = []
    for thr in THRESHOLDS:
        for lo, hi in ATR_BANDS:
            for ad in AGREEMENTS:
                bt = Backtester(
                    initial_capital=settings.get("initial_capital", 100_000),
                    commission_rate=settings.get("commission_rate", 0.001),
                    slippage_rate=settings.get("slippage_rate", 0.0005),
                    max_position_pct=settings.get("max_position_pct", 0.02),
                    max_drawdown_kill=settings.get("max_drawdown_kill", 0.10),
                    max_concurrent_positions=settings.get("max_concurrent_positions", 5),
                    confidence_threshold=thr,
                    stop_atr_mult=settings.get("stop_atr_mult", 1.5),
                    take_profit_atr_mult=settings.get("take_profit_atr_mult", 2.5),
                    max_holding_bars=settings.get("max_holding_bars", 10),
                    seed=settings.get("seed", 42),
                    regime_filter=False,
                    long_only=True,
                    atr_pct_low=lo,
                    atr_pct_high=hi,
                    min_agreement_delta=ad,
                )
                summary = bt.run(data)
                m_rf0 = compute_metrics(summary, risk_free_rate=0.0, timeframe=settings["timeframe"])
                m_rfp = compute_metrics(summary, risk_free_rate=settings.get("risk_free_rate", 0.05), timeframe=settings["timeframe"])
                row = {
                    "threshold": thr,
                    "atr_low": lo,
                    "atr_high": hi,
                    "min_agreement_delta": ad,
                    "n_trades": m_rf0["n_trades"],
                    "win_rate": m_rf0["win_rate"],
                    "profit_factor": m_rf0["profit_factor"],
                    "sharpe_tw_rf0": m_rf0["sharpe_trade_weighted"],
                    "sharpe_tw_rfp": m_rfp["sharpe_trade_weighted"],
                    "sharpe_all_rf0": m_rf0["sharpe"],
                    "max_dd": m_rf0["max_drawdown"],
                    "total_return": m_rf0["total_return"],
                    "invested_periods": m_rf0["invested_periods"],
                    "total_periods": m_rf0["total_periods"],
                }
                rows.append(row)
                runs[(thr, lo, hi, ad)] = (summary, m_rfp, m_rf0)
                print(
                    f"thr={thr:.2f}  ATR[{lo:.2f},{hi:.2f}]  agr={ad:.2f}  "
                    f"n={m_rf0['n_trades']:3d}  PF={m_rf0['profit_factor']:.2f}  "
                    f"Sharpe_tw_rf0={m_rf0['sharpe_trade_weighted']:+.2f}  "
                    f"Sharpe_tw_rf={m_rfp['sharpe_trade_weighted']:+.2f}  "
                    f"ret={m_rf0['total_return']:+.2%}"
                )

    eligible = [r for r in rows if r["n_trades"] >= MIN_TRADES]
    pf_qual = [r for r in eligible if r["profit_factor"] > 1.2]
    if pf_qual:
        best = max(pf_qual, key=lambda r: r["sharpe_tw_rf0"])
    elif eligible:
        best = max(eligible, key=lambda r: r["profit_factor"])
    else:
        best = max(rows, key=lambda r: r["n_trades"])
    print(
        f"\nSelected: thr={best['threshold']:.2f}  ATR[{best['atr_low']:.2f},{best['atr_high']:.2f}]  "
        f"agr={best['min_agreement_delta']:.2f}  n={best['n_trades']}  "
        f"PF={best['profit_factor']:.2f}  Sharpe_tw_rf0={best['sharpe_tw_rf0']:+.2f}  "
        f"Sharpe_tw_rfp={best['sharpe_tw_rfp']:+.2f}"
    )

    summary, m_rfp, m_rf0 = runs[
        (best["threshold"], best["atr_low"], best["atr_high"], best["min_agreement_delta"])
    ]
    save_results(summary, m_rfp, Path("backtest/results/latest_run.json"))
    save_results(summary, m_rfp, Path("quant_trader/backtest/results/latest_run.json"))
    save_results(summary, m_rf0, Path("backtest/results/latest_run_rf0.json"))

    sweep_path = Path("backtest/results/filter_sweep.json")
    sweep_path.write_text(
        json.dumps({"results": rows, "best": best, "min_trades": MIN_TRADES}, indent=2)
    )
    print(f"\nSweep saved to {sweep_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
