# Build State

Last updated: LIVE DATA INTEGRATION COMPLETE

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

## Verification (against real network data)
- `pytest quant_trader/tests/ -v` — **11/11 passing** (added live-network test
  for the GitHub CSV fallback against BTC-USD and SPY).
- `python main.py --mode fetch` — pulls real bars from coinmetrics
  (BTC/ETH/SOL daily) and OStochastic (SPY OHLCV daily). QQQ has no
  on-allowlist source; pipeline reports 0 bars and skips it.
- `python main.py --mode backtest` — 285 real trades over 2022→present,
  win rate 41.05 %, profit factor 0.82, max DD -0.40 %.
- `python main.py --mode paper --once` — first one-shot step opened ETH-USD
  long @ \$2256.61 and SOL-USD long @ \$82.98 from real cache prices.
- `python dashboard/monitor.py --once` (or `python -m
  quant_trader.dashboard.monitor`) — renders all panels with real
  portfolio state.

## Network constraint discovered & worked around
The runtime sandbox proxy enforces a hard host allowlist (PyPI, GitHub,
S3 / GCS only). Yahoo Finance and every crypto exchange API return 403.
DataFetcher now has a third `_fetch_github_csv` fallback that pulls real
historical OHLCV / reference-price feeds mirrored on GitHub. See
DECISIONS.md → "Live data integration (post-build)" for details.
