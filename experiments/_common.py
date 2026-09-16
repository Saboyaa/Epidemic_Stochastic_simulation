"""Utilidades compartilhadas pelos scripts de experimentos."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epidemic.config import DATA_RAW, DATA_PROC, FIG_DIR, TAB_DIR, GAMMA, SCENARIOS  # noqa: E402,F401


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)
