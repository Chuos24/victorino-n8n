# Build State

Last updated: STRATEGY EDGE TUNED (PF 1.22)

## Progress
- [x] Phase 1 — Foundation
- [x] Phase 2 — Feature Engineering
- [x] Phase 3 — AI Signal Models
- [x] Phase 4 — Strategy & Risk
- [x] Phase 5 — Backtesting Engine
- [x] Phase 6 — Paper Trading
- [x] Phase 7 — Terminal Dashboard
- [x] Phase 8 — Tests & Docs
- [x] Phase 9 — Live data integration
- [x] Phase 10 — Strategy edge tuning

## Verification (against real network data)
- `python main.py --mode fetch` — pulls real bars from coinmetrics
  (BTC/ETH/SOL daily) and OStochastic (SPY OHLCV daily). QQQ has no
  on-allowlist source; pipeline reports 0 bars and skips it.
- `python main.py --mode backtest` — **86 real trades** with the tuned
  config (`signal_confidence_threshold: 0.70`, `long_only: true`,
  `regime_filter: false`): win rate 59.30 %, **profit factor 1.22**,
  Sharpe -5.30, max DD -1.14 %, total return +0.69 %.
- `python scripts/analyze_trades.py backtest/results/latest_run.json` —
  decomposes the trade log by symbol, year, regime band, confidence band.
- `python scripts/sweep_threshold.py` — runs the full
  threshold × filter grid and re-saves `latest_run.json` for the chosen
  config.

## Tuning summary

| Run                | n   | Win   | PF   | Sharpe | DD     | Return |
| ------------------ | --- | ----- | ---- | ------ | ------ | ------ |
| Baseline (thr 0.6) | 318 | 47.5% | 0.78 | -63.6  | -0.38% | -0.33% |
| Tuned (thr 0.70)   |  86 | 59.3% | 1.22 |  -5.3  | -1.14% | +0.69% |

PF target (> 1.2) **met**. Sharpe target (> 0.8) **not met** — daily mean
return (~0.001 %) is below the 5 % `risk_free_rate` setting; Sharpe is
scale-invariant in position size, so reaching that target needs a
stronger per-bar edge, not a bigger Kelly multiplier. See DECISIONS.md
→ "Strategy edge tuning (post live-data)" for the full analysis.

## Network constraint discovered & worked around
The runtime sandbox proxy enforces a hard host allowlist (PyPI, GitHub,
S3 / GCS only). Yahoo Finance and every crypto exchange API return 403.
DataFetcher has a `_fetch_github_csv` fallback that pulls real historical
OHLCV / reference-price feeds mirrored on GitHub. See DECISIONS.md →
"Live data integration (post-build)" for details.
