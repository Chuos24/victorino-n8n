"""Group a trade log by symbol, year, regime, confidence band — find where the
strategy bleeds the most P&L. Reads `backtest/results/<run>.json` and prints a
summary suitable for sticking in DECISIONS.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path("backtest/results/baseline_run.json")


def _load_trades(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text())
    df = pd.DataFrame(payload["trades"])
    if df.empty:
        return df
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True, errors="coerce")
    df["year"] = df["entry_time"].dt.year
    df["month"] = df["entry_time"].dt.month
    return df


def _stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0}
    pnls = df["pnl"]
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    pf = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
    return {
        "n": int(len(df)),
        "win_rate": float((pnls > 0).mean()),
        "pnl_sum": float(pnls.sum()),
        "avg_pnl": float(pnls.mean()),
        "profit_factor": float(pf),
    }


def _print_group(title: str, df: pd.DataFrame, by: str) -> None:
    print(f"\n=== {title} ===")
    if df.empty:
        print("(no trades)")
        return
    rows = []
    for key, sub in df.groupby(by):
        s = _stats(sub)
        s[by] = key
        rows.append(s)
    out = pd.DataFrame(rows).sort_values("pnl_sum")
    out = out[[by, "n", "win_rate", "pnl_sum", "avg_pnl", "profit_factor"]]
    print(
        out.to_string(
            index=False,
            float_format=lambda v: f"{v:+,.4f}" if abs(v) < 1e3 else f"{v:+,.2f}",
        )
    )


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else DEFAULT_PATH
    df = _load_trades(path)
    if df.empty:
        print(f"No trades in {path}")
        return 0

    overall = _stats(df)
    print(f"Run: {path}")
    print(
        f"Trades: {overall['n']}  win_rate={overall['win_rate']:.2%}  "
        f"pnl_sum={overall['pnl_sum']:+.2f}  PF={overall['profit_factor']:.2f}"
    )

    _print_group("By symbol", df, "symbol")
    _print_group("By year", df, "year")
    _print_group("By exit reason", df, "reason")
    _print_group("By direction", df, "direction")

    if "entry_regime_slope" in df.columns:
        bins = pd.cut(
            df["entry_regime_slope"],
            bins=[-1.0, -0.05, -0.005, 0.005, 0.05, 1.0],
            labels=["strong_down", "down", "flat", "up", "strong_up"],
        )
        df = df.assign(regime_bin=bins.astype(str))
        _print_group("By regime (200-EMA slope)", df, "regime_bin")

    if "entry_confidence" in df.columns:
        bins = pd.cut(
            df["entry_confidence"],
            bins=[0.0, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 1.0],
            labels=["<0.55", "0.55-0.60", "0.60-0.65", "0.65-0.70", "0.70-0.75", "0.75-0.80", ">=0.80"],
        )
        df = df.assign(conf_bin=bins.astype(str))
        _print_group("By confidence band", df, "conf_bin")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
