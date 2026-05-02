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
            "sortino": 0.0,
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
        "sortino": float(sortino),
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
        "sharpe": ("Sharpe Ratio", lambda v: f"{v:.2f}"),
        "sortino": ("Sortino Ratio", lambda v: f"{v:.2f}"),
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
