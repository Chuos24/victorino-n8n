# Build State

Last updated: SHARPE FIX + PER-BAR EDGE FILTERS (literal user spec)

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
- [x] Phase 11 — Sharpe fix + per-bar edge filters

## Verification (against real network data)
- `python main.py --mode fetch` — pulls real bars from coinmetrics
  (BTC/ETH/SOL daily) and OStochastic (SPY OHLCV daily). QQQ has no
  on-allowlist source; pipeline reports 0 bars and skips it.
- `python main.py --mode backtest` — runs the canonical backtest with
  the tuned config (`signal_confidence_threshold: 0.72`, `long_only:
  true`, `atr_pct_low: 0.30`, `atr_pct_high: 0.70`,
  `min_agreement_delta: 0.05`).
- `python scripts/run_final.py` — same trade log, two metric snapshots
  (rf=0.05 primary, rf=0 secondary). Saves `latest_run.json` and
  `latest_run_rf0.json`.
- `python scripts/sweep_filters.py` — full ATR-band × threshold ×
  agreement-delta sweep (72 runs); saves `filter_sweep.json`.
- `python scripts/analyze_trades.py backtest/results/latest_run.json` —
  decomposes by symbol, year, regime band, confidence band, and now
  ATR-percentile band / agreement-delta band.

## Sharpe calculation fix

The original Sharpe applied the daily-equivalent risk-free rate to every
calendar bar — including the 721 of 875 bars where the long-only
strategy was uninvested. That dragged the reported Sharpe to -5.30 even
when total return was positive. Fix:

- `Backtester` now records an `invested_curve` (open-position count per
  bar) alongside the equity curve.
- `compute_metrics` now reports four Sharpe variants: all-bars / rf, all-bars / rf=0,
  trade-weighted / rf, trade-weighted / rf=0. The trade-weighted rf=0
  number is the sane comparison for the user's `Sharpe > 0.8` target.
- Same change for Sortino. The metrics table also exposes `Invested
  Bars` / `Total Bars` so the share of capital-at-work is visible.

## Tuning summary

| Run                          | n   | Win   | PF   | Sharpe (all, rf) | Sharpe (tw, rf=0) | Return |
| ---------------------------- | --- | ----- | ---- | ---------------- | ----------------- | ------ |
| Baseline (thr 0.6, no filters) | 318 | 47.5% | 0.78 | -63.6 | n/a    | -0.33% |
| Phase-10 tuned (thr 0.70, lo) |  86 | 59.3% | 1.22 |  -5.3 | n/a    | +0.69% |
| **Phase-11 (literal spec)**   |  29 | 48.3% | 0.48 | -12.6 | -1.49 | -0.96% |

PF target (> 1.2) **not met** (0.48). Trade-weighted-rf=0 Sharpe target
(> 0.8) **not met** (-1.49). The Sharpe-fix did its job — the rf=0
variants are sane numbers — but the literal 30-70 ATR band excludes the
high-volatility regime where long-only crypto/SPY catches its winners,
collapsing PF. Headline finding from `scripts/sweep_filters.py` (72-run
grid): no filter combination on the current data hits PF > 1.2 with
n_trades ≥ 50. The next iteration should attack the underlying signal
(more / better features, alternate training windows, dropping SOL where
PF stays ground-floor), not stack more entry filters. See DECISIONS.md
→ "Sharpe-fix + per-bar edge filters" for the full table.

## Network constraint discovered & worked around
The runtime sandbox proxy enforces a hard host allowlist (PyPI, GitHub,
S3 / GCS only). Yahoo Finance and every crypto exchange API return 403.
DataFetcher has a `_fetch_github_csv` fallback that pulls real historical
OHLCV / reference-price feeds mirrored on GitHub. See DECISIONS.md →
"Live data integration (post-build)" for details.
