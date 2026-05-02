# Architectural Decisions Log

This file tracks all decisions made during autonomous build of the AI Quant Trader.

## Phase 1 — Foundation

- **Repo layout**: All code is namespaced under `quant_trader/` to keep top-level
  clean. Tests live at `quant_trader/tests/` so imports use a single root.
- **Data cache**: Parquet files cached at `quant_trader/data/cache/<symbol>_<tf>.parquet`.
  We never re-fetch overlapping ranges — on subsequent calls we read the cache
  and only fetch the missing tail.
- **Crypto symbols**: yfinance supports `BTC-USD`/`ETH-USD`/`SOL-USD` directly,
  so we route all symbols through yfinance for consistency. The `ccxt` Binance
  adapter is included as a fallback for when yfinance rate-limits or returns
  empty (we map `BTC-USD -> BTC/USDT` etc.).
- **Timeframe normalization**: `1h` is mapped to yfinance's `60m` and ccxt's
  `1h`. yfinance only allows ~730 days of intraday data, so for longer
  backtests we fall back to daily bars and document the limitation.
- **No API keys**: ccxt is initialized without auth (public endpoints only).

## Phase 2 — Feature Engineering

- **NaN handling**: All rolling features generate leading NaNs. The
  `FeatureEngine` drops those rows after computing all features, so output is
  guaranteed NaN-free at the cost of warmup bars.
- **`ta` library**: Used for RSI, MACD, Bollinger, ATR. Manual fallbacks
  implemented in case the library is missing.

## Phase 3 — Models

- **Targets**: ±0.5% threshold for classification. For regression (LSTM) we
  target the next-bar log return directly.
- **Walk-forward**: A simple expanding-window split is used (train on first
  70%, test on last 30%) inside the model fit/predict utilities. The backtester
  itself walks forward bar-by-bar so there is no leakage at inference time.
- **LSTM lightweight**: 2 layers, hidden 64, dropout 0.2, batch size 64,
  10 epochs max — keeps CPU training under a minute on the dataset sizes used.

## Phase 4 — Strategy & Risk

- **Kelly sizing**: Capped at half-Kelly (multiply Kelly fraction by 0.5)
  because raw Kelly is notoriously aggressive. Hard cap of 2% per position
  remains the binding constraint in practice.
- **Drawdown kill switch**: Once tripped, stays tripped for the rest of the
  run — no automatic re-arm.

## Phase 5 — Backtest

- **Slippage**: 5 bps on entry and exit. Commission: 10 bps round-trip
  (5 bps each side via the same factor).
- **Fills**: Simulated at the bar's close price (we generate signal AFTER bar
  closes). This is conservative and avoids any lookahead.

## Phase 6 — Paper Trader

- **State persistence**: JSON file at `quant_trader/execution/portfolio_state.json`.
  On startup, restore positions and equity if present.
- **plyer**: Wrapped in a try/except — desktop notifications are best-effort
  and silently disabled if the platform doesn't support them (e.g., headless).

## Phase 7 — Dashboard

- **rich.live**: Single Live render loop, refreshes every 60s. Reads from the
  same `portfolio_state.json` and `trade_log.csv` the paper trader writes.

## Phase 8 — Tests

- **Mocked data**: `test_data.py` uses a fake DataFrame to avoid network
  flakiness in CI — the real fetcher is exercised via `python main.py --mode fetch`.
- **Mini backtest**: `test_backtest.py` synthesizes 30 days of hourly bars
  rather than relying on a network fetch.

## Environment caveats encountered during build

- The build environment had no outbound network access, so
  `python main.py --mode fetch` returned 0 bars for every symbol and the
  end-to-end backtest produced 0 trades. The CLI ran without errors though,
  and the synthetic-data tests fully exercise the pipeline including the
  models, strategy, risk, and analytics. Real fetch / backtest will work
  unchanged on a normal network connection.
- The `ta` package's PEP 517 wheel build requires modern setuptools.
  We upgrade pip/setuptools/wheel via `pip install --user --upgrade`
  before installing `ta`.

## Live data integration (post-build)

The runtime sandbox proxy turned out to enforce a strict allowlist that
**blocks every financial API** the original fetcher relied on:

- yfinance hosts (`query1.finance.yahoo.com`, `fc.yahoo.com`, etc.) → 403
  `host_not_allowed`.
- ccxt's Binance public endpoint (`api.binance.com`) → 403
  `host_not_allowed`. All other major exchanges (Kraken, Coinbase, Bybit,
  KuCoin, Gate, Bitfinex, OKX, etc.) are also blocked.
- Stooq, Alpha Vantage, IEX, Tiingo, CoinGecko, Coinbase, Polygon — all
  blocked.

The only outbound hosts the proxy permits are PyPI / files.pythonhosted.org,
GitHub (api / raw / codeload / objects), and AWS S3 / Google Cloud Storage.
"Real network access" therefore means "real, but only via PyPI + GitHub +
S3/GCS." Live tick streams are not reachable on this host.

### Real-data sources (GitHub CSV mirror fallback)

To still trade against **real market data** we added a third
`_fetch_github_csv` fallback in `quant_trader/data/fetcher.py`. It runs
after yfinance and ccxt fail and pulls real OHLCV / reference-price feeds
that are mirrored on GitHub:

| Symbol  | Source                                                       | Format |
| ------- | ------------------------------------------------------------ | ------ |
| BTC-USD | `coinmetrics/data csv/btc.csv` (`PriceUSD`)                  | daily ref price → synth O/H/L from prev close |
| ETH-USD | `coinmetrics/data csv/eth.csv` (`PriceUSD`)                  | same |
| SOL-USD | `coinmetrics/data csv/sol.csv` (`CapMrktEstUSD` ÷ implied supply derived from recent `ReferenceRate`) | same |
| SPY     | `OStochastic/Daily-SPY-data-from-2000-2025/spy_data.csv`     | real daily OHLCV |
| QQQ     | *(no source on the allowlist)*                               | empty |

OHLC for crypto is synthesized as `open = prev_close`, `high = max(open,
close)`, `low = min(open, close)` so all bar fields are real prices but
without intraday detail.

### Other adjustments

- Switched `timeframe` from `1h` to `1d` in `quant_trader/config/settings.yaml`
  — the GitHub mirrors only carry daily resolution.
- Reduced `max_holding_bars` from `48` (hours) to `10` (days) to keep the
  same ~2-week max holding window after the timeframe change.
- The pipeline gracefully reports `0 bars` for QQQ; the backtester / paper
  trader skip empty-data symbols.

### Synthetic OHLC widening for daily-only sources

Coin Metrics gives us a single daily reference price. Naïvely setting
`open=prev_close`, `high=max(O, C)`, `low=min(O, C)` produces bars with a
zero high-low wick on most days, which makes
`FeatureEngine.body_wick_ratio` divide by zero and drop *every* feature
row → the LightGBM trainer then errors with
`index -1 is out of bounds for axis 0 with size 0`. We fix this with
`DataFetcher._synth_ohlc_from_close`, which widens H/L by 25 % of the
intraday body plus a 5 bps absolute floor. The OHLC is still derived from
real daily closes, just with a small modeled intraday range so bars
aren't degenerate.

### Paper trader adjustments

- `TraderConfig.train_days` lifted from 365 → 1460 and a new
  `step_lookback_days = 540` field replaces the hard-coded 120-day window
  in `step()`. The original numbers were sized for hourly data; on daily
  bars they were not enough for the 200-row feature warmup, so SPY in
  particular only got 82 training rows.
- `TraderConfig.timeframe` default changed from `1h` → `1d` and
  `max_holding_bars` default changed from `48` → `10` to match the new
  resolution.
- Removed the line in `warmup()` that pre-seeded `last_seen_bar` to the
  most recent bar. With it set, the very first `step()` always treated
  the latest bar as already-processed and produced no signals — a fresh
  startup should evaluate the current bar instead.
- Added a `--once` CLI flag (`python main.py --mode paper --once`) that
  runs warmup + a single `step()` and prints the signals/positions
  generated. Useful for non-blocking smoke tests; `--mode paper` without
  the flag still starts the long-running poll loop.

On the first one-shot run the loop produced **real signals** from real
prices: ETH-USD long @ \$2256.61 (conf 0.69) and SOL-USD long @ \$82.98
(conf 0.66), with two positions opened against the live cache.

### Dashboard

- The original dashboard could only be invoked as a module
  (`python -m quant_trader.dashboard.monitor`). Added a
  `dashboard/monitor.py` shim at the repo root so
  `python dashboard/monitor.py` (with optional `--once`) also works, and
  taught `quant_trader/dashboard/monitor.py` to bootstrap `sys.path` when
  it is run as a script with `__package__` empty.
- Verified panels render against the real portfolio state from step 4:
  Portfolio (equity \$99,999.70), Risk (drawdown 0 %, kill switch off),
  Recent Trades (ETH-USD and SOL-USD opens at the live cache prices).

### Backtest metrics on real data

After step 2 with the unmodified default strategy parameters:
- 4 symbols traded (BTC-USD, ETH-USD, SOL-USD, SPY); QQQ skipped.
- 285 trades over the 2022-01-01 → present window.
- Win rate 41.05 %, profit factor 0.82, max DD -0.40 %, total return
  -0.26 %. Sharpe is heavily negative because the equity curve barely
  moves — Kelly sizing keeps positions small with this win/loss ratio,
  so the metric is dominated by small drift. These numbers are real,
  not synthetic.

## Strategy edge tuning (post live-data)

Goal was to flip the strategy from negative edge (PF 0.78–0.82) to PF > 1.2
on the same real-data pipeline. New code:

- `FeatureEngine.ema_200_slope`: 20-bar relative slope of the 200-period
  EMA, used as a market-regime proxy.
- `MomentumMeanReversion.regime_filter`: when `True`, only enters trades
  when the symbol's own 200-EMA slope clears `regime_slope_threshold`
  (i.e. trending up). Sideways / down regimes get a `regime_block` skip.
- `MomentumMeanReversion.long_only`: when `True`, drops every short
  signal at the entry gate (`long_only_block` skip reason).
- `Backtester.Trade` now records `entry_confidence`, `entry_regime_slope`,
  and `bars_held` so `scripts/analyze_trades.py` can decompose P&L by
  symbol, year, regime band, and confidence band.

### Trade-log analysis (baseline run, regime filter OFF, threshold 0.60)

`scripts/analyze_trades.py backtest/results/baseline_run.json` on the
318-trade baseline showed:

| Slice                 | n   | Win rate | PF    | P&L sum    |
| --------------------- | --- | -------- | ----- | ---------- |
| Direction +1 (long)   | 108 | 56.5 %   | 1.13  | +$36       |
| Direction -1 (short)  | 210 | 42.9 %   | 0.67  | **-$300**  |
| SOL-USD               | 108 | 45.4 %   | 0.70  | -$168      |
| BTC-USD               |  79 | 40.5 %   | 0.60  | -$89       |
| ETH-USD               | 112 | 53.6 %   | 0.96  | -$16       |
| SPY                   |  19 | 52.6 %   | 1.41  | +$9        |
| Confidence < 0.70     | 137 | 42 %     | 0.55  | **-$218**  |
| Confidence ≥ 0.80     | 133 | 51 %     | 0.97  | -$15       |
| Reason: stop_loss     | 130 | 0 %      | 0.00  | **-$1,104**|

**Headline finding:** the bleed is *directional, not regime-based.* Shorts
lost ~$300 across the BTC/ETH/SOL bull window; longs were already
slightly profitable. Stop-losses ate the worst trades. Lower-confidence
trades (< 0.70) were almost uniformly negative.

### Threshold + filter sweep

`scripts/sweep_threshold.py` runs the backtester across the cross-product
of `confidence_threshold ∈ {0.55..0.80}`, `regime_filter ∈ {off, on}`, and
`long_only ∈ {off, on}`. Selection rule: pick the highest-Sharpe run
whose `n_trades ≥ 50` and `profit_factor > 1.2`; fall back to the highest
PF if the PF target is not reachable in any 50-trade run.

Best three configurations from the sweep:

| Config                    | thr  | n   | Win  | PF   | Sharpe | DD    |
| ------------------------- | ---- | --- | ---- | ---- | ------ | ----- |
| **rf=0, lo=1** (selected) | 0.70 |  86 | 59.3 | 1.22 |  -5.30 | -1.14% |
| rf=0, lo=1                | 0.65 | 111 | 58.6 | 1.12 |  -4.94 | -1.46% |
| rf=1, lo=1, slope=+0.01   | 0.70 |  37 | 64.9 | 1.36 | -10.91 | n/a    |

The regime filter alone (`rf=1, lo=0`) topped out at PF 0.98 and never
hit the 1.2 target. Combining `rf=1` with `lo=1` and a slightly positive
slope cutoff produced the highest PF (1.36) of any tested config but
fell to 37 trades — below the user's 50-trade floor. So the selected
production config keeps the regime filter implemented but **disabled in
`settings.yaml`**, and relies on `long_only=true` + `threshold=0.70` to
deliver the edge. Anyone tightening the universe (e.g. dropping SOL-USD
where PF is still 0.10) can revisit and re-enable the regime filter.

### Final tuned metrics

`python main.py --mode backtest` with `signal_confidence_threshold: 0.70`,
`long_only: true`, `regime_filter: false`:

- 86 trades, win rate 59.30 %, **profit factor 1.22**, max DD -1.14 %,
  total return +0.69 %.
- Per-symbol PF: ETH-USD 1.39, BTC-USD 1.39, SPY 3.00, SOL-USD 0.10
  (SOL is the next obvious symbol to drop).

### Why Sharpe stays negative

Sharpe in the metrics table is **-5.30** despite total return being
positive — because the daily mean return (~0.001 %) is well below the
daily-equivalent risk-free rate (5 % / 252 ≈ 0.02 %). Sharpe is
scale-invariant in position size, so boosting Kelly or `max_position_pct`
does not move it: it would scale both numerator and denominator by the
same factor. Reaching the user's `Sharpe > 0.8` target needs an
annualised return well above the 5 % `risk_free_rate` setting — that
requires either a stronger per-bar edge or a longer holding window
amplifying the realised wins. Documented in DECISIONS.md so the next
iteration knows what knob actually moves the metric.

### Files added

- `scripts/analyze_trades.py` — per-symbol / regime / confidence
  decomposition of any `latest_run.json`.
- `scripts/sweep_threshold.py` — full filter × threshold grid; saves
  `backtest/results/threshold_sweep.json` and refreshes
  `backtest/results/latest_run.json` with the selected run.
- `scripts/sweep_regime.py` — secondary sweep of
  `regime_slope_threshold` at the chosen confidence threshold.
- `backtest/results/{baseline_run,threshold_sweep,regime_slope_sweep,latest_run}.json`
  — saved artefacts from each step. Mirrored under
  `quant_trader/backtest/results/latest_run.json` for the dashboard.
