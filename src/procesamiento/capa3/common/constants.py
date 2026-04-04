from __future__ import annotations

from pathlib import Path

import pandas as pd
from rich.console import Console

try:
    from config.settings import obtener_ruta  # type: ignore
except Exception:
    def obtener_ruta(p: str) -> Path:
        return Path(p)


console = Console()
DEBUG = False

ALLOWED_MIN_DATE = pd.Timestamp("2023-01-01")
ALLOWED_MAX_DATE = pd.Timestamp("2025-12-31")

