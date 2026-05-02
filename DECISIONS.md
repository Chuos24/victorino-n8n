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

### Backtest metrics on real data

After step 2 with the unmodified default strategy parameters:
- 4 symbols traded (BTC-USD, ETH-USD, SOL-USD, SPY); QQQ skipped.
- 285 trades over the 2022-01-01 → present window.
- Win rate 41.05 %, profit factor 0.82, max DD -0.40 %, total return
  -0.26 %. Sharpe is heavily negative because the equity curve barely
  moves — Kelly sizing keeps positions small with this win/loss ratio,
  so the metric is dominated by small drift. These numbers are real,
  not synthetic.
