"""Alerting: console + log file + best-effort desktop notifications."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

logger = logging.getLogger(__name__)

LOG_PATH = Path(__file__).parent / "alerts.log"


class Alerter:
    """Emit human-readable alerts to console, file, and OS notifier."""

    def __init__(self, log_path: Path | str = LOG_PATH):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.console = Console()

    def _now(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    def _emit(self, kind: str, msg: str, color: str = "white") -> None:
        ts = self._now()
        line = f"[{ts}] [{kind}] {msg}"
        try:
            with open(self.log_path, "a") as f:
                f.write(line + "\n")
        except Exception:  # pragma: no cover
            pass
        self.console.print(f"[{color}][{ts}][/] [bold]{kind}[/] {msg}")
        self._desktop_notify(kind, msg)

    def signal(self, msg: str) -> None:
        self._emit("SIGNAL", msg, color="cyan")

    def trade(self, msg: str) -> None:
        self._emit("TRADE", msg, color="green")

    def warn(self, msg: str) -> None:
        self._emit("WARN", msg, color="yellow")

    def error(self, msg: str) -> None:
        self._emit("ERROR", msg, color="red")

    def _desktop_notify(self, title: str, msg: str) -> None:
        try:  # pragma: no cover - depends on platform
            from plyer import notification

            notification.notify(title=f"Quant: {title}", message=msg, timeout=4)
        except Exception:
            pass
