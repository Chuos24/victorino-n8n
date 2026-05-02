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
