"""Top-level shim that delegates to quant_trader.dashboard.monitor.

Lets `python dashboard/monitor.py` work from the repo root in addition to
`python -m quant_trader.dashboard.monitor`.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from quant_trader.dashboard.monitor import main


if __name__ == "__main__":
    main(once="--once" in sys.argv)
