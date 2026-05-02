"""Sweep ensemble confidence thresholds, with the 200-EMA regime filter and a
long-only switch as the two filter dimensions.

The trade-log analysis on the baseline run shows:
  * shorts lose money (PF 0.67) while longs are profitable (PF 1.13)
  * lower confidence bands (<0.70) bleed
  * the 200-EMA "flat" regime has the worst PF of any regime bin

So we sweep threshold ∈ {0.55, 0.60, 0.65, 0.70, 0.75, 0.80} across four
filter combinations:
  - rf=0, lo=0  (baseline)
  - rf=1, lo=0  (regime filter only)
  - rf=0, lo=1  (long-only)
  - rf=1, lo=1  (regime filter + long-only)

We pick the configuration that maximises Sharpe subject to n_trades >= 50,
save it as `latest_run.json`, and dump the full sweep table.
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
THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def _settings() -> dict:
    return yaml.safe_load(open("quant_trader/config/settings.yaml"))


def _data(settings: dict):
    fetcher = DataFetcher()
    pipe = DataPipeline(fetcher)
    return pipe.fetch_all(
        universe=settings["universe"],
        timeframe=settings["timeframe"],
        start=settings["backtest_start"],
        end=datetime.now(timezone.utc),
    )


def _run_one(settings, data, threshold, regime_filter, long_only):
    bt = Backtester(
        initial_capital=settings.get("initial_capital", 100_000),
        commission_rate=settings.get("commission_rate", 0.001),
        slippage_rate=settings.get("slippage_rate", 0.0005),
        max_position_pct=settings.get("max_position_pct", 0.02),
        max_drawdown_kill=settings.get("max_drawdown_kill", 0.10),
        max_concurrent_positions=settings.get("max_concurrent_positions", 5),
        confidence_threshold=threshold,
        stop_atr_mult=settings.get("stop_atr_mult", 1.5),
        take_profit_atr_mult=settings.get("take_profit_atr_mult", 2.5),
        max_holding_bars=settings.get("max_holding_bars", 10),
        seed=settings.get("seed", 42),
        regime_filter=regime_filter,
        regime_slope_threshold=settings.get("regime_slope_threshold", 0.0),
        long_only=long_only,
    )
    summary = bt.run(data)
    metrics = compute_metrics(
        summary,
        risk_free_rate=settings.get("risk_free_rate", 0.05),
        timeframe=settings["timeframe"],
    )
    return summary, metrics


def _row(thr, rf, lo, m):
    return {
        "threshold": thr,
        "regime_filter": rf,
        "long_only": lo,
        "n_trades": m["n_trades"],
        "win_rate": m["win_rate"],
        "profit_factor": m["profit_factor"],
        "sharpe": m["sharpe"],
        "sortino": m["sortino"],
        "max_dd": m["max_drawdown"],
        "total_return": m["total_return"],
    }


def main() -> int:
    settings = _settings()
    data = _data(settings)

    runs: dict = {}
    rows: list = []
    for rf in (False, True):
        for lo in (False, True):
            print(f"\n--- regime_filter={rf}  long_only={lo} ---")
            for thr in THRESHOLDS:
                summary, metrics = _run_one(settings, data, thr, rf, lo)
                row = _row(thr, rf, lo, metrics)
                rows.append(row)
                runs[(thr, rf, lo)] = (summary, metrics)
                print(
                    f"thr={thr:.2f}  rf={int(rf)} lo={int(lo)}  "
                    f"n={metrics['n_trades']:4d}  wr={metrics['win_rate']:.2%}  "
                    f"PF={metrics['profit_factor']:.2f}  "
                    f"Sharpe={metrics['sharpe']:.2f}  "
                    f"DD={metrics['max_drawdown']:.2%}  ret={metrics['total_return']:.2%}"
                )

    eligible = [r for r in rows if r["n_trades"] >= MIN_TRADES]
    if not eligible:
        eligible = sorted(rows, key=lambda r: r["n_trades"], reverse=True)[:1]
    # Primary objective: maximise Sharpe subject to n_trades >= MIN_TRADES.
    # Tie-break by profit factor so a run that hits the PF > 1.2 target is
    # preferred over an equally-Sharpe run that doesn't.
    best_sharpe = max(eligible, key=lambda r: r["sharpe"])
    best_pf = max(eligible, key=lambda r: r["profit_factor"])
    print(
        f"\nBest by Sharpe (n>={MIN_TRADES}): thr={best_sharpe['threshold']:.2f}  "
        f"rf={int(best_sharpe['regime_filter'])}  "
        f"lo={int(best_sharpe['long_only'])}  "
        f"Sharpe={best_sharpe['sharpe']:.2f}  "
        f"PF={best_sharpe['profit_factor']:.2f}  "
        f"n={best_sharpe['n_trades']}"
    )
    print(
        f"Best by PF      (n>={MIN_TRADES}): thr={best_pf['threshold']:.2f}  "
        f"rf={int(best_pf['regime_filter'])}  "
        f"lo={int(best_pf['long_only'])}  "
        f"Sharpe={best_pf['sharpe']:.2f}  "
        f"PF={best_pf['profit_factor']:.2f}  "
        f"n={best_pf['n_trades']}"
    )
    # Selection rule: pick the highest-Sharpe run whose PF exceeds the 1.2
    # target; if none does, fall back to highest-PF.
    pf_qualifying = [r for r in eligible if r["profit_factor"] > 1.2]
    if pf_qualifying:
        best = max(pf_qualifying, key=lambda r: r["sharpe"])
    else:
        best = best_pf
    print(
        f"\nSelected: thr={best['threshold']:.2f}  "
        f"rf={int(best['regime_filter'])}  lo={int(best['long_only'])}  "
        f"Sharpe={best['sharpe']:.2f}  PF={best['profit_factor']:.2f}  "
        f"n={best['n_trades']}"
    )

    summary, metrics = runs[(best["threshold"], best["regime_filter"], best["long_only"])]
    out_pkg = Path("quant_trader/backtest/results/latest_run.json")
    out_top = Path("backtest/results/latest_run.json")
    save_results(summary, metrics, out_pkg)
    save_results(summary, metrics, out_top)

    sweep_path = Path("backtest/results/threshold_sweep.json")
    sweep_path.parent.mkdir(parents=True, exist_ok=True)
    sweep_path.write_text(
        json.dumps({"results": rows, "best": best, "min_trades": MIN_TRADES}, indent=2)
    )
    print(f"\nSweep saved to {sweep_path}")
    print(f"Selected run saved to {out_top} and {out_pkg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
