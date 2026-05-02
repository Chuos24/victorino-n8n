"""Live terminal dashboard for the paper trader.

Reads `portfolio_state.json` and `trade_log.csv` produced by the paper
trader and renders a multi-panel rich.live view that refreshes every 60s.
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..execution.paper_trader import STATE_PATH, TRADE_LOG_PATH


def _load_state() -> dict:
    if not Path(STATE_PATH).exists():
        return {}
    try:
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_trades(n: int = 10) -> List[dict]:
    if not Path(TRADE_LOG_PATH).exists():
        return []
    try:
        with open(TRADE_LOG_PATH, "r") as f:
            rows = list(csv.DictReader(f))
        return rows[-n:]
    except Exception:
        return []


def _portfolio_panel(state: dict) -> Panel:
    equity = float(state.get("equity", 0.0))
    cash = float(state.get("cash", 0.0))
    peak = float(state.get("peak_equity", equity))
    daily_pnl = equity - peak  # quick proxy: deviation from peak
    color = "green" if daily_pnl >= 0 else "red"
    body = Text()
    body.append(f"Equity: ${equity:,.2f}\n", style="bold")
    body.append(f"Cash:   ${cash:,.2f}\n")
    body.append(f"Peak:   ${peak:,.2f}\n")
    body.append("Δ vs peak: ", style="white")
    body.append(f"{daily_pnl:+,.2f}\n", style=color)
    return Panel(body, title="Portfolio", border_style="cyan")


def _positions_panel(state: dict) -> Panel:
    table = Table(expand=True)
    for col in ["Symbol", "Dir", "Entry", "Units", "Bar", "ATR"]:
        table.add_column(col)
    for sym, pos in state.get("positions", {}).items():
        direction = pos.get("direction", 0)
        dir_str = "[green]LONG[/]" if direction == 1 else "[red]SHORT[/]"
        table.add_row(
            sym,
            dir_str,
            f"{float(pos.get('entry_price', 0)):.2f}",
            f"{float(pos.get('units', 0)):.4f}",
            str(pos.get("entry_bar", 0)),
            f"{float(pos.get('entry_atr', 0)):.2f}",
        )
    return Panel(table, title="Open Positions", border_style="green")


def _trades_panel() -> Panel:
    rows = _load_trades(10)
    table = Table(expand=True)
    for col in ["Time", "Symbol", "Action", "Dir", "Price", "Units", "PnL", "Reason"]:
        table.add_column(col)
    for r in rows:
        pnl_val = r.get("pnl") or ""
        try:
            pnl_f = float(pnl_val)
            pnl_str = f"[green]{pnl_f:+,.2f}[/]" if pnl_f >= 0 else f"[red]{pnl_f:+,.2f}[/]"
        except (TypeError, ValueError):
            pnl_str = ""
        table.add_row(
            (r.get("timestamp") or "")[:19],
            r.get("symbol", ""),
            r.get("action", ""),
            r.get("direction", ""),
            r.get("price", ""),
            r.get("units", ""),
            pnl_str,
            r.get("reason", ""),
        )
    return Panel(table, title="Recent Trades", border_style="magenta")


def _signals_panel() -> Panel:
    rows = _load_trades(50)
    # We use the trade log for actionable signals; explicit signals aren't
    # persisted to disk, so show the most recent trade-actions as the
    # closest proxy for "latest signals".
    table = Table(expand=True)
    for col in ["Time", "Symbol", "Action", "Dir", "Price"]:
        table.add_column(col)
    for r in rows[-5:]:
        table.add_row(
            (r.get("timestamp") or "")[:19],
            r.get("symbol", ""),
            r.get("action", ""),
            r.get("direction", ""),
            r.get("price", ""),
        )
    return Panel(table, title="Latest Activity", border_style="blue")


def _risk_panel(state: dict) -> Panel:
    dd = float(state.get("drawdown", 0.0))
    kill = state.get("kill_switch", False)
    body = Text()
    body.append(f"Drawdown: {dd:.2%}\n")
    body.append("Kill Switch: ")
    if kill:
        body.append("ON\n", style="bold red")
    else:
        body.append("OFF\n", style="bold green")
    body.append(
        f"Last update: {state.get('timestamp', 'n/a')}\n",
        style="dim",
    )
    return Panel(body, title="Risk Status", border_style="red" if kill else "yellow")


def render(layout: Layout) -> None:
    state = _load_state()
    layout["portfolio"].update(_portfolio_panel(state))
    layout["positions"].update(_positions_panel(state))
    layout["signals"].update(_signals_panel())
    layout["trades"].update(_trades_panel())
    layout["risk"].update(_risk_panel(state))


def make_layout() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="top", size=8),
        Layout(name="middle", ratio=1),
        Layout(name="trades", size=12),
    )
    layout["top"].split_row(
        Layout(name="portfolio"),
        Layout(name="risk"),
    )
    layout["middle"].split_row(
        Layout(name="positions"),
        Layout(name="signals"),
    )
    return layout


def main(refresh: int = 60, once: bool = False) -> None:
    """Run the dashboard. Pass ``once=True`` for a single render (testing)."""
    console = Console()
    layout = make_layout()
    if once:
        render(layout)
        console.print(layout)
        return
    with Live(layout, console=console, refresh_per_second=1, screen=True):
        while True:
            try:
                render(layout)
                time.sleep(refresh)
            except KeyboardInterrupt:
                return


if __name__ == "__main__":  # pragma: no cover
    main()
