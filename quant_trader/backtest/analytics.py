"""Performance metrics computed from a backtest result."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table


PERIODS_PER_YEAR = {
    "1m": 60 * 24 * 252,
    "5m": 12 * 24 * 252,
    "15m": 4 * 24 * 252,
    "1h": 24 * 252,
    "1d": 252,
}


def _annualization_factor(equity: pd.DataFrame, timeframe: str) -> float:
    if timeframe in PERIODS_PER_YEAR:
        return PERIODS_PER_YEAR[timeframe]
    if equity.empty or len(equity) < 2:
        return 252.0
    span_days = max(
        1.0, (equity.index[-1] - equity.index[0]).total_seconds() / 86400.0
    )
    return len(equity) * (365.0 / span_days)


def compute_metrics(
    summary: dict,
    risk_free_rate: float = 0.05,
    timeframe: str = "1h",
) -> dict[str, Any]:
    equity_df: pd.DataFrame = summary.get("equity_curve")
    if equity_df is None or equity_df.empty:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "sharpe_rf0": 0.0,
            "sharpe_trade_weighted": 0.0,
            "sharpe_trade_weighted_rf0": 0.0,
            "sortino": 0.0,
            "sortino_trade_weighted": 0.0,
            "invested_periods": 0,
            "total_periods": 0,
            "max_drawdown": 0.0,
            "max_drawdown_duration_days": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "n_trades": 0,
            "final_equity": summary.get("final_equity", 0.0),
        }

    equity = equity_df["equity"].astype(float)
    returns = equity.pct_change().fillna(0.0)
    initial = float(summary.get("initial_capital", equity.iloc[0]))
    final = float(equity.iloc[-1])
    total_return = final / initial - 1.0 if initial > 0 else 0.0

    span_days = max(
        1.0, (equity.index[-1] - equity.index[0]).total_seconds() / 86400.0
    )
    years = span_days / 365.25
    cagr = ((final / initial) ** (1 / years) - 1) if initial > 0 and years > 0 else 0.0

    ann = _annualization_factor(equity_df, timeframe)
    rf_per_period = risk_free_rate / ann
    excess = returns - rf_per_period
    std = returns.std()
    sharpe = (
        float(excess.mean() / std * math.sqrt(ann)) if std and not math.isnan(std) and std > 0 else 0.0
    )
    downside = returns[returns < 0]
    dstd = downside.std()
    sortino = (
        float(excess.mean() / dstd * math.sqrt(ann))
        if dstd and not math.isnan(dstd) and dstd > 0
        else 0.0
    )

    # Trade-weighted Sharpe: only count bars where the strategy was actually
    # invested. The all-bars Sharpe applied the daily-equivalent risk-free
    # rate to every calendar day including days with zero exposure, which
    # subtracted ~0.02 % from a stream whose own daily mean is ~0.001 %.
    # That made the Sharpe deeply negative even when realised returns were
    # positive. The fix: compute mean / std / rf only over the subset of
    # bars on which at least one position was open (or, as a fallback when
    # the engine doesn't expose `invested_curve`, bars whose return is
    # non-zero — the equity curve is flat by construction on uninvested
    # daily bars).
    invested_df = summary.get("invested_curve")
    if invested_df is not None and not invested_df.empty:
        invested_mask = (
            invested_df["open_positions"].reindex(returns.index).fillna(0) > 0
        )
    else:
        invested_mask = returns != 0
    invested_returns = returns[invested_mask]
    invested_std = invested_returns.std()
    invested_excess = invested_returns - rf_per_period
    sharpe_trade_weighted = (
        float(invested_excess.mean() / invested_std * math.sqrt(ann))
        if invested_std and not math.isnan(invested_std) and invested_std > 0
        else 0.0
    )
    invested_downside = invested_returns[invested_returns < 0]
    invested_dstd = invested_downside.std()
    sortino_trade_weighted = (
        float(invested_excess.mean() / invested_dstd * math.sqrt(ann))
        if invested_dstd and not math.isnan(invested_dstd) and invested_dstd > 0
        else 0.0
    )

    # rf=0 Sharpe — useful sanity check when the supplied risk-free rate
    # dominates a barely-non-zero return stream.
    sharpe_rf0 = (
        float(returns.mean() / std * math.sqrt(ann))
        if std and not math.isnan(std) and std > 0 else 0.0
    )
    sharpe_trade_weighted_rf0 = (
        float(invested_returns.mean() / invested_std * math.sqrt(ann))
        if invested_std and not math.isnan(invested_std) and invested_std > 0
        else 0.0
    )

    invested_periods = int(invested_mask.sum())

    # Max drawdown.
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = float(drawdown.min())
    # DD duration: longest stretch where equity stayed below a previous peak.
    dd_dur = 0.0
    current_run_start = None
    longest_run_seconds = 0.0
    for ts, eq in equity.items():
        if eq < running_max.loc[ts]:
            if current_run_start is None:
                current_run_start = ts
            longest_run_seconds = max(
                longest_run_seconds, (ts - current_run_start).total_seconds()
            )
        else:
            current_run_start = None
    dd_dur = longest_run_seconds / 86400.0

    # Trade-level metrics.
    trades = summary.get("trades", [])
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = (len(wins) / len(pnls)) if pnls else 0.0
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    profit_factor = (
        float(sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else 0.0
    )

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "sharpe_rf0": float(sharpe_rf0),
        "sharpe_trade_weighted": float(sharpe_trade_weighted),
        "sharpe_trade_weighted_rf0": float(sharpe_trade_weighted_rf0),
        "sortino": float(sortino),
        "sortino_trade_weighted": float(sortino_trade_weighted),
        "invested_periods": int(invested_periods),
        "total_periods": int(len(returns)),
        "max_drawdown": float(max_dd),
        "max_drawdown_duration_days": float(dd_dur),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "n_trades": int(len(trades)),
        "final_equity": float(final),
        "initial_capital": float(initial),
    }


def render_metrics_table(metrics: dict[str, Any]) -> None:
    """Pretty-print metrics with rich."""
    console = Console()
    table = Table(title="Backtest Performance", show_lines=False)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    fmt = {
        "total_return": ("Total Return", lambda v: f"{v:.2%}"),
        "cagr": ("CAGR", lambda v: f"{v:.2%}"),
        "sharpe": ("Sharpe (all bars)", lambda v: f"{v:.2f}"),
        "sharpe_rf0": ("Sharpe (rf=0)", lambda v: f"{v:.2f}"),
        "sharpe_trade_weighted": (
            "Sharpe (trade-weighted)", lambda v: f"{v:.2f}"
        ),
        "sharpe_trade_weighted_rf0": (
            "Sharpe (trade-weighted, rf=0)", lambda v: f"{v:.2f}"
        ),
        "sortino": ("Sortino Ratio", lambda v: f"{v:.2f}"),
        "sortino_trade_weighted": (
            "Sortino (trade-weighted)", lambda v: f"{v:.2f}"
        ),
        "invested_periods": ("Invested Bars", lambda v: f"{v}"),
        "total_periods": ("Total Bars", lambda v: f"{v}"),
        "max_drawdown": ("Max Drawdown", lambda v: f"{v:.2%}"),
        "max_drawdown_duration_days": (
            "Max DD Duration (days)",
            lambda v: f"{v:.1f}",
        ),
        "win_rate": ("Win Rate", lambda v: f"{v:.2%}"),
        "profit_factor": ("Profit Factor", lambda v: f"{v:.2f}"),
        "avg_win": ("Avg Win ($)", lambda v: f"{v:,.2f}"),
        "avg_loss": ("Avg Loss ($)", lambda v: f"{v:,.2f}"),
        "n_trades": ("# Trades", lambda v: f"{v}"),
        "final_equity": ("Final Equity", lambda v: f"${v:,.2f}"),
        "initial_capital": ("Initial Capital", lambda v: f"${v:,.2f}"),
    }
    for key, (label, formatter) in fmt.items():
        if key in metrics:
            table.add_row(label, formatter(metrics[key]))
    console.print(table)


def save_results(
    summary: dict, metrics: dict[str, Any], path: str | Path
) -> None:
    """Write metrics + a slim equity-curve snapshot to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    eq_df: pd.DataFrame = summary.get("equity_curve")
    eq_serial = []
    if eq_df is not None and not eq_df.empty:
        eq_serial = [
            {"ts": str(ts), "equity": float(eq)}
            for ts, eq in eq_df["equity"].items()
        ]
    payload = {
        "metrics": metrics,
        "equity_curve": eq_serial,
        "trades": summary.get("trades", []),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
