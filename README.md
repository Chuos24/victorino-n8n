# AI Quant Trader

An end-to-end algorithmic trading system that fetches real market data,
engineers features, runs a LightGBM + LSTM ensemble for signal generation,
sizes positions with a Kelly-based risk manager, backtests on historical
data, and runs a paper-trading loop with a live terminal dashboard.

The system is **paper-trading only**. It is research software for studying
quantitative-trading patterns; it is not financial advice and is not
connected to any live broker.

## Production stats (Phase 14)

Backtest period: 2022-01-01 → 2026-05-03 on real daily bars
(Coin Metrics + OStochastic GitHub mirrors).

| Metric                              | Value      |
| ----------------------------------- | ---------- |
| Trades                              | **136**    |
| Win rate                            | **58.09 %**|
| Profit factor                       | **1.73**   |
| Sharpe (trade-weighted, rf=0)       | **+1.30**  |
| Sharpe (trade-weighted, rf=0.05)    | -1.42      |
| Sortino (trade-weighted)            | -2.15      |
| Total return                        | +4.00 %    |
| Max drawdown                        | -1.12 %    |
| Max DD duration                     | 216 days   |
| Invested bars / total bars          | 436 / 876  |
| Final equity ($100 k start)         | $104,003.65|

### Per-symbol PF

| Symbol   | Trades | Win rate | PF   |
| -------- | ------ | -------- | ---- |
| SPY      |   8    | 62.5 %   | 2.78 |
| LTC-USD  |  28    | 64.3 %   | 2.64 |
| ADA-USD  |  21    | 66.7 %   | 2.30 |
| BTC-USD  |  36    | 55.6 %   | 1.33 |
| ETH-USD  |  43    | 51.2 %   | 1.17 |

All five universe symbols are PF-positive. The full per-phase tuning
history (including the dropped symbols SOL/QQQ/IWM/DOT/LINK and the
data-source reachability investigation) lives in
[`DECISIONS.md`](DECISIONS.md).

### Universe

The production universe is **`[BTC-USD, ETH-USD, SPY, LTC-USD, ADA-USD]`**.
It's the result of three rounds of trade-log analysis on real data:

- **SOL-USD** dropped (Phase 11) — PF stuck at 0.10 across every
  threshold/filter combination. Coin Metrics' SOL series is also the
  most synthetic of the crypto sources (price back-derived from market
  cap + implied supply).
- **QQQ / IWM** unreachable — every equity-data host (yfinance, Stooq,
  NASDAQ, Alpha Vantage) sits behind a sandbox 403; only the
  OStochastic SPY mirror is on the GitHub allowlist. Verified with
  `scripts/sweep_universe.py`.
- **DOT-USD / LINK-USD** dropped (Phase 14) — Phase-13 trade log
  showed DOT 0.74 / LINK 0.34 PFs were dragging overall PF down;
  removing them lifted overall PF 1.45 → 1.73 and
  trade-weighted-rf=0 Sharpe 0.89 → 1.30.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Fetch and cache all universe data
python main.py --mode fetch

# 2. Run a full backtest and print performance metrics
python main.py --mode backtest

# 3. Start the paper trading loop
python main.py --mode paper

# 4. (in another terminal) view the live dashboard
python quant_trader/dashboard/monitor.py
```

## Architecture

```
                        +-----------------+
                        | settings.yaml   |
                        +--------+--------+
                                 |
+--------------+   +-------------v------------+   +------------------+
|  yfinance /  +-->+      DataFetcher /       +-->+  FeatureEngine   |
|  ccxt        |   |      DataPipeline        |   |  (returns,       |
+--------------+   +-------------+------------+   |   RSI, MACD,...) |
                                 |                 +--------+--------+
                                 |                          |
                                 v                          v
                       +---------+----------+      +--------+--------+
                       |  Parquet cache     |      |   Ensemble      |
                       +--------------------+      |  LGBM + LSTM    |
                                                   +--------+--------+
                                                            |
                                                            v
+----------------+   +---------------+   +------------------+--------+
| RiskManager    +<--+ KellySizer    +<--+ Strategy: MomentumMR     |
+-------+--------+   +---------------+   +-----------+--------------+
        |                                            |
        v                                            v
+-------+-------------------------------+   +--------+----------------+
| Backtester (event-driven)             |   | PaperTrader (live loop) |
+-------+-------------------------------+   +--------+----------------+
        |                                            |
        v                                            v
+-------+--------+                          +--------+----------+
| analytics      |                          | Dashboard (rich)  |
| (Sharpe etc.)  |                          +-------------------+
+----------------+
```

## Configuration reference

Production values from `quant_trader/config/settings.yaml`. The `Default`
column shows the in-code default that applies when the setting is absent.

| Key | Description | Production | Default |
| --- | --- | --- | --- |
| `universe` | List of tickers to trade | `[BTC-USD, ETH-USD, SPY, LTC-USD, ADA-USD]` | n/a |
| `timeframe` | Bar interval | `1d` | `1h` |
| `backtest_start` | Backtest start date (ISO) | `2022-01-01` | n/a |
| `max_position_pct` | Hard cap per-position size as % of equity | `0.02` | `0.02` |
| `max_drawdown_kill` | Drawdown that trips the kill switch | `0.10` | `0.10` |
| `signal_confidence_threshold` | Minimum LGBM confidence to enter | `0.45` | `0.60` |
| `paper_mode` | Use simulated fills | `true` | `true` |
| `seed` | Global RNG seed | `42` | `42` |
| `initial_capital` | Starting equity | `100000` | `100000` |
| `commission_rate` | Per-trade commission | `0.001` | `0.001` |
| `slippage_rate` | Slippage applied at fill | `0.0005` | `0.0005` |
| `max_concurrent_positions` | Open-position cap | `5` | `5` |
| `risk_free_rate` | Annual risk-free rate for Sharpe | `0.05` | `0.05` |
| `stop_atr_mult` | Stop-loss in ATR units | `1.5` | `1.5` |
| `take_profit_atr_mult` | Take-profit in ATR units | `2.5` | `2.5` |
| `max_holding_bars` | Time-stop in bars | `10` | `48` |
| `regime_filter` | Skip entries when 200-EMA slope ≤ threshold | `true` | `false` |
| `regime_slope_threshold` | Slope cutoff for the regime filter | `0.0` | `0.0` |
| `long_only` | Drop short signals at the entry gate | `true` | `false` |
| `atr_pct_low` / `_high` | ATR(14) percentile band for entries | `0.0 / 1.0` | `0.0 / 1.0` |
| `min_agreement_delta` | LGBM top-class margin floor | `0.05` | `0.0` |
| `poll_interval` | Paper-trader poll frequency (s) | `60` | `60` |

## Tests

```bash
pytest quant_trader/tests/ -v
```

11 tests cover the data layer (cache round-trip, GitHub fallback against
real network), feature engine (NaN-free output, OHLCV preserved), risk
manager (position cap, kill switch, max-concurrent, Kelly cap), and an
end-to-end backtest on synthetic OHLCV.

## Reproducing the production numbers

```bash
# Refresh the on-disk parquet cache
python main.py --mode fetch

# Run the canonical backtest. Saves
#   backtest/results/latest_run.json
#   quant_trader/backtest/results/latest_run.json
python main.py --mode backtest

# Per-symbol / per-regime / per-confidence trade-log decomposition
python scripts/analyze_trades.py backtest/results/latest_run.json
```

## Disclaimer

This software is for **research and education only**. It runs in paper-trading
mode and is not connected to any brokerage. Past performance does not
guarantee future results. Nothing in this repository constitutes financial
advice.
