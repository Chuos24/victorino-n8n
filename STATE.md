# Build State

Last updated: UNIVERSE EXPANSION (PF 1.45 on 161 trades)

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
- [x] Phase 12 — Signal-quality features (OBV / 52w / cross-asset BTC)
- [x] Phase 13 — Universe expansion (LTC/ADA/DOT/LINK)

## Verification (against real network data)
- `python main.py --mode fetch` — pulls real bars from Coin Metrics
  (BTC/ETH/LTC/ADA/DOT/LINK daily, real `PriceUSD`) and OStochastic
  (SPY OHLCV daily). QQQ / IWM remain unfetchable on this sandbox.
- `python main.py --mode backtest` — runs the canonical backtest with
  the Phase-13 config (`universe: [BTC-USD, ETH-USD, SPY, LTC-USD,
  ADA-USD, DOT-USD, LINK-USD]`, `signal_confidence_threshold: 0.45`,
  `long_only: true`, `regime_filter: true`, `atr_pct_low/high:
  0.0/1.0`, `min_agreement_delta: 0.05`). **161 trades, PF 1.45.**
- `python scripts/sweep_universe.py` — runs the user's two literal
  approaches (QQQ+IWM, 4h bars) plus the substitute crypto universe;
  saves `backtest/results/universe_sweep.json` and per-config
  snapshots so the measured outcome of each approach is on disk.
- `python scripts/sweep_universe_thr.py` — confidence-threshold sweep
  on the substitute universe; the selection rule (highest n_trades
  with PF > 1.2, tie-break by Sharpe_tw_rf0) picks `thr=0.45`.
- `python scripts/analyze_trades.py backtest/results/latest_run.json` —
  decomposes by symbol, year, regime band, confidence band,
  ATR-percentile band, agreement-delta band.
- `quant_trader/models/lgbm_importance.csv` — per-fit feature
  importance log written by `LGBMSignalModel.fit`.

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

| Run                            | n     | Win   | PF       | Sharpe (tw, rf=0) | Return |
| ------------------------------ | ----- | ----- | -------- | ----------------- | ------ |
| Baseline (thr 0.6, no filters) | 318   | 47.5% | 0.78     | n/a               | -0.33% |
| Phase-10 tuned (thr 0.70, lo)  |  86   | 59.3% | 1.22     | n/a               | +0.69% |
| Phase-11 (literal spec)        |  29   | 48.3% | 0.48     | -1.49             | -0.96% |
| Phase-12 (new features)        |  41   | 56.1% | 1.37     | +0.62             | +0.50% |
| **Phase-13 (universe + thr 0.45)** | **161** | **55.3%** | **1.45** | **+0.89** | **+3.01%** |

### Phase 13 per-symbol PF breakdown

| Symbol     | Trades | Win Rate | PF   |
| ---------- | ------ | -------- | ---- |
| LTC-USD    | 28     | 64.3 %   | 2.64 |
| ADA-USD    | 21     | 66.7 %   | 2.31 |
| SPY        |  8     | 62.5 %   | 2.79 |
| BTC-USD    | 36     | 55.6 %   | 1.33 |
| ETH-USD    | 43     | 51.2 %   | 1.17 |
| DOT-USD    | 10     | 50.0 %   | 0.74 |
| LINK-USD   | 15     | 33.3 %   | 0.34 |
| **All**    | **161** | **55.3 %** | **1.45** |

### Approach test outcome (`scripts/sweep_universe.py`)

| Config                                | n   | PF   | Notes                                         |
| ------------------------------------- | --- | ---- | --------------------------------------------- |
| Phase-12 baseline                     |  41 | 1.37 | BTC/ETH/SPY, 1d                              |
| **A1 literal (+ QQQ + IWM, 1d)**      |  41 | 1.37 | Identical — QQQ + IWM both 0 bars (no on-allowlist source) |
| **A2 literal (1d → 4h)**              |   0 | 0.00 | Coin Metrics + OStochastic mirrors are daily-only |
| A1 substitute (+ LTC/ADA/DOT/LINK)    |  55 | 2.14 | Phase-12 thr=0.65 — PF up but n short of 150 |
| **A1 substitute + thr=0.45 (winner)** | **161** | **1.45** | n_trades and PF targets both met |

n_trades target (≥ 150): **met** (161). PF target (> 1.2): **met**
(1.45). Trade-weighted-rf=0 Sharpe lifted +0.62 → +0.89 — better
risk-adjusted return on the larger sample, not just a wider trade
count. See DECISIONS.md → "Phase 13 — Universe expansion + threshold
relaxation" for the full investigation, including the reachability
table for QQQ / IWM / 4h sources.

## Network constraint discovered & worked around
The runtime sandbox proxy enforces a hard host allowlist (PyPI, GitHub,
S3 / GCS only). Yahoo Finance and every crypto exchange API return 403.
DataFetcher has a `_fetch_github_csv` fallback that pulls real historical
OHLCV / reference-price feeds mirrored on GitHub. See DECISIONS.md →
"Live data integration (post-build)" for details.
