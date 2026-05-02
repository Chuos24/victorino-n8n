# AI Quant Trader

An end-to-end algorithmic trading system that fetches market data, engineers
features, runs an LightGBM + LSTM ensemble for signal generation, sizes
positions with a Kelly-based risk manager, backtests on historical data, and
runs a paper-trading loop with a live terminal dashboard.

The system is **paper-trading only**. It is research software for studying
quantitative-trading patterns; it is not financial advice and is not connected
to any live broker.

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

| Key | Description | Default |
| --- | --- | --- |
| `universe` | List of tickers to trade | `[BTC-USD, ETH-USD, SOL-USD, SPY, QQQ]` |
| `timeframe` | Bar interval | `1h` |
| `backtest_start` | Backtest start date (ISO) | `2022-01-01` |
| `max_position_pct` | Hard cap per-position size as % of equity | `0.02` |
| `max_drawdown_kill` | Drawdown that trips the kill switch | `0.10` |
| `signal_confidence_threshold` | Minimum LGBM confidence to enter | `0.60` |
| `paper_mode` | Use simulated fills | `true` |
| `seed` | Global RNG seed | `42` |
| `initial_capital` | Starting equity | `100000` |
| `commission_rate` | Per-trade commission | `0.001` |
| `slippage_rate` | Slippage applied at fill | `0.0005` |
| `max_concurrent_positions` | Open-position cap | `5` |
| `risk_free_rate` | Annual risk-free rate for Sharpe | `0.05` |
| `stop_atr_mult` | Stop-loss in ATR units | `1.5` |
| `take_profit_atr_mult` | Take-profit in ATR units | `2.5` |
| `max_holding_bars` | Time-stop in bars | `48` |
| `poll_interval` | Paper-trader poll frequency (s) | `60` |

## Disclaimer

This software is for **research and education only**. It runs in paper-trading
mode and is not connected to any brokerage. Past performance does not
guarantee future results. Nothing in this repository constitutes financial
advice.
