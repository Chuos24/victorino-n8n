"""Phase-13 experiment runner: test the user's two approaches for getting
trade count to 150+ without dropping PF below 1.2, plus a feasible
substitute for approach 1.

  A1-literal: add QQQ and IWM to the daily universe
  A1-sub    : add LTC/ADA/DOT/LINK/MATIC to the daily universe (Coin Metrics)
  A2        : drop from 1d → 4h bars

QQQ, IWM and 4h bars all sit behind blocked hosts on this sandbox
(yfinance / Stooq / NASDAQ / Alpha Vantage), so A1-literal and A2 are
expected to come back with 0 bars / 0 trades. We run them anyway so the
DECISIONS.md table reports the *measured* failure mode rather than the
hypothesised one.

Each run uses the Phase-12 production knobs: confidence_threshold=0.65,
long_only=true, regime_filter=true, atr_pct_low/high=0/1,
min_agreement_delta=0.05. Reports `n_trades`, `profit_factor`,
trade-weighted Sharpe (rf=0), per-symbol PF, and
invested-bar/total-bar fractions.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_trader.backtest import Backtester, compute_metrics
from quant_trader.backtest.analytics import save_results
from quant_trader.data import DataFetcher, DataPipeline


CONFIGS = [
    {
        "name": "phase12_baseline",
        "label": "Phase-12 baseline (BTC/ETH/SPY, 1d)",
        "universe": ["BTC-USD", "ETH-USD", "SPY"],
        "timeframe": "1d",
    },
    {
        "name": "a1_literal",
        "label": "A1 literal: + QQQ + IWM (1d)",
        "universe": ["BTC-USD", "ETH-USD", "SPY", "QQQ", "IWM"],
        "timeframe": "1d",
    },
    {
        "name": "a1_substitute",
        "label": "A1 substitute: + LTC/ADA/DOT/LINK (1d, Coin Metrics)",
        "universe": [
            "BTC-USD", "ETH-USD", "SPY",
            "LTC-USD", "ADA-USD", "DOT-USD", "LINK-USD",
        ],
        "timeframe": "1d",
    },
    {
        "name": "a2_4h",
        "label": "A2 literal: drop 1d → 4h bars (BTC/ETH/SPY)",
        "universe": ["BTC-USD", "ETH-USD", "SPY"],
        "timeframe": "4h",
    },
]


def _settings() -> dict:
    return yaml.safe_load(open("quant_trader/config/settings.yaml"))


def _bt(settings: dict, threshold: float = 0.65) -> Backtester:
    return Backtester(
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
        regime_filter=True,
        regime_slope_threshold=settings.get("regime_slope_threshold", 0.0),
        long_only=True,
        atr_pct_low=0.0,
        atr_pct_high=1.0,
        min_agreement_delta=settings.get("min_agreement_delta", 0.05),
    )


def _per_symbol_pf(trades) -> dict:
    wins = defaultdict(float)
    losses = defaultdict(float)
    counts = defaultdict(int)
    for t in trades:
        counts[t.symbol] += 1
        if t.pnl > 0:
            wins[t.symbol] += t.pnl
        else:
            losses[t.symbol] += -t.pnl
    out = {}
    for s in counts:
        pf = (wins[s] / losses[s]) if losses[s] > 0 else float("inf")
        out[s] = {"n": counts[s], "pf": pf, "pnl": wins[s] - losses[s]}
    return out


def main() -> int:
    settings = _settings()
    fetcher = DataFetcher()
    pipe = DataPipeline(fetcher)
    summary_rows: list[dict] = []

    for cfg in CONFIGS:
        print(f"\n=== {cfg['label']} ===")
        data = pipe.fetch_all(
            universe=cfg["universe"],
            timeframe=cfg["timeframe"],
            start=settings["backtest_start"],
            end=datetime.now(timezone.utc),
        )
        bars = {s: len(d) for s, d in data.items()}
        print(f"bars per symbol: {bars}")

        bt = _bt(settings)
        bt_summary = bt.run(data)
        m = compute_metrics(bt_summary, risk_free_rate=0.0, timeframe=cfg["timeframe"])
        per_sym = _per_symbol_pf(bt.trades)
        sym_str = "  ".join(
            f"{s}:{per_sym[s]['n']}/{per_sym[s]['pf']:.2f}"
            for s in sorted(per_sym)
        ) or "(no trades)"

        print(
            f"n_trades={m['n_trades']:4d}  PF={m['profit_factor']:.2f}  "
            f"win_rate={m['win_rate']:.2%}  "
            f"Sharpe_tw_rf0={m['sharpe_trade_weighted']:+.2f}  "
            f"invested={m['invested_periods']}/{m['total_periods']}  "
            f"return={m['total_return']:+.2%}"
        )
        print(f"per-symbol [n/PF]: {sym_str}")

        summary_rows.append(
            {
                "config": cfg["name"],
                "label": cfg["label"],
                "universe": cfg["universe"],
                "timeframe": cfg["timeframe"],
                "bars_per_symbol": bars,
                "n_trades": m["n_trades"],
                "profit_factor": m["profit_factor"],
                "win_rate": m["win_rate"],
                "sharpe_tw_rf0": m["sharpe_trade_weighted"],
                "sharpe_tw_rfp": compute_metrics(
                    bt_summary, risk_free_rate=0.05, timeframe=cfg["timeframe"]
                )["sharpe_trade_weighted"],
                "max_drawdown": m["max_drawdown"],
                "total_return": m["total_return"],
                "invested_periods": m["invested_periods"],
                "total_periods": m["total_periods"],
                "per_symbol": per_sym,
            }
        )

        # Save the run JSON for whichever config qualifies.
        out_pkg = Path(f"quant_trader/backtest/results/run_{cfg['name']}.json")
        save_results(bt_summary, m, out_pkg)
        out_top = Path(f"backtest/results/run_{cfg['name']}.json")
        save_results(bt_summary, m, out_top)

    # Pick the config that hits >=150 trades with PF > 1.2; tie-break by
    # Sharpe_tw_rf0.
    qualifying = [
        r for r in summary_rows
        if r["n_trades"] >= 150 and r["profit_factor"] > 1.2
    ]
    if qualifying:
        winner = max(qualifying, key=lambda r: r["sharpe_tw_rf0"])
    else:
        # Fallback: highest PF among configs with at least double the
        # baseline trade count.
        with_volume = [r for r in summary_rows if r["n_trades"] >= 80]
        winner = (
            max(with_volume, key=lambda r: r["profit_factor"])
            if with_volume
            else max(summary_rows, key=lambda r: r["n_trades"])
        )

    print(
        f"\nWinner: {winner['config']}  "
        f"n={winner['n_trades']}  PF={winner['profit_factor']:.2f}  "
        f"Sharpe_tw_rf0={winner['sharpe_tw_rf0']:+.2f}"
    )

    # Promote winner to canonical latest_run.json.
    src_pkg = Path(f"quant_trader/backtest/results/run_{winner['config']}.json")
    src_top = Path(f"backtest/results/run_{winner['config']}.json")
    Path("backtest/results/latest_run.json").write_bytes(src_top.read_bytes())
    Path("quant_trader/backtest/results/latest_run.json").write_bytes(src_pkg.read_bytes())

    out = Path("backtest/results/universe_sweep.json")
    out.write_text(
        json.dumps({"results": summary_rows, "winner": winner["config"]}, indent=2, default=str)
    )
    print(f"\nSweep saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
