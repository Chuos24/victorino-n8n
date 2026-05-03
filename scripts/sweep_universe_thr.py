"""Sweep confidence thresholds on the Phase-13 substitute universe
(BTC/ETH/SPY/LTC/ADA/DOT/LINK) to find the highest trade count that still
keeps PF > 1.2. The Phase-12 production knobs are otherwise held fixed:
long_only=true, regime_filter=true, ATR band off, agreement_delta=0.05.
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


THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]
UNIVERSE = [
    "BTC-USD", "ETH-USD", "SPY",
    "LTC-USD", "ADA-USD", "DOT-USD", "LINK-USD",
]


def main() -> int:
    settings = yaml.safe_load(open("quant_trader/config/settings.yaml"))
    fetcher = DataFetcher()
    pipe = DataPipeline(fetcher)
    data = pipe.fetch_all(
        universe=UNIVERSE,
        timeframe="1d",
        start=settings["backtest_start"],
        end=datetime.now(timezone.utc),
    )

    rows = []
    runs = {}
    for thr in THRESHOLDS:
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
            regime_filter=True,
            regime_slope_threshold=0.0,
            long_only=True,
            atr_pct_low=0.0,
            atr_pct_high=1.0,
            min_agreement_delta=0.05,
        )
        summary = bt.run(data)
        m = compute_metrics(summary, risk_free_rate=0.0, timeframe="1d")
        m_rfp = compute_metrics(summary, risk_free_rate=0.05, timeframe="1d")
        per_sym = defaultdict(lambda: [0, 0.0, 0.0])
        for t in bt.trades:
            per_sym[t.symbol][0] += 1
            if t.pnl > 0:
                per_sym[t.symbol][1] += t.pnl
            else:
                per_sym[t.symbol][2] += -t.pnl
        sym_str = "  ".join(
            f"{s}:{per_sym[s][0]}/{(per_sym[s][1] / per_sym[s][2] if per_sym[s][2] > 0 else 9.99):.2f}"
            for s in sorted(per_sym)
        )
        rows.append({
            "threshold": thr,
            "n_trades": m["n_trades"],
            "win_rate": m["win_rate"],
            "profit_factor": m["profit_factor"],
            "sharpe_tw_rf0": m["sharpe_trade_weighted"],
            "sharpe_tw_rfp": m_rfp["sharpe_trade_weighted"],
            "max_dd": m["max_drawdown"],
            "total_return": m["total_return"],
            "invested_periods": m["invested_periods"],
            "total_periods": m["total_periods"],
            "per_symbol": {s: {"n": v[0], "pnl_w": v[1], "pnl_l": v[2]} for s, v in per_sym.items()},
        })
        runs[thr] = (summary, m_rfp)
        print(
            f"thr={thr:.2f}  n={m['n_trades']:4d}  "
            f"PF={m['profit_factor']:.2f}  Sharpe_tw_rf0={m['sharpe_trade_weighted']:+.2f}  "
            f"DD={m['max_drawdown']:.2%}  ret={m['total_return']:+.2%}  per_sym=[{sym_str}]"
        )

    # Selection: highest n_trades with PF > 1.2; tiebreak by Sharpe.
    pf_qual = [r for r in rows if r["profit_factor"] > 1.2]
    if pf_qual:
        winner = max(pf_qual, key=lambda r: (r["n_trades"], r["sharpe_tw_rf0"]))
    else:
        winner = max(rows, key=lambda r: r["profit_factor"])
    print(
        f"\nSelected: thr={winner['threshold']:.2f}  n={winner['n_trades']}  "
        f"PF={winner['profit_factor']:.2f}  Sharpe_tw_rf0={winner['sharpe_tw_rf0']:+.2f}"
    )

    summary, m_rfp = runs[winner["threshold"]]
    save_results(summary, m_rfp, Path("backtest/results/latest_run.json"))
    save_results(summary, m_rfp, Path("quant_trader/backtest/results/latest_run.json"))
    Path("backtest/results/universe_thr_sweep.json").write_text(
        json.dumps({"results": rows, "winner": winner}, indent=2, default=str)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
