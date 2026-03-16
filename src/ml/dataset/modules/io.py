from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_partitioned_parquet_dir(base: Path) -> pd.DataFrame:
    """Lee un directorio con parquets particionados tipo Spark (subcarpetas date=... etc.)."""
    base = Path(base).resolve()
    if not base.exists():
        raise FileNotFoundError(f"No existe: {base}")

    files = list(base.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No hay .parquet dentro de: {base}")

    return pd.concat([pd.read_parquet(fp) for fp in files], ignore_index=True)


def ensure_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"[{name}] Faltan columnas: {miss}")


def safe_date_for_filename(s: str) -> str:
    return s.strip().replace("/", "-").replace("\\", "-").replace(":", "-")