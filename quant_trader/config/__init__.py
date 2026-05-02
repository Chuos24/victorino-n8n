"""Configuration loader for the quant trader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).parent / "settings.yaml"


def load_settings(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config and return as a plain dict."""
    p = Path(path) if path else _CONFIG_PATH
    with open(p, "r") as f:
        return yaml.safe_load(f)
