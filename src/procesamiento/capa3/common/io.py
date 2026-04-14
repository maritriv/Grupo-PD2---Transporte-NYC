from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import pandas as pd

from .constants import console


def safe_remove_dir(path: Path) -> None:
    path = Path(path).resolve()
    if not path.exists():
        return

    try:
        shutil.rmtree(path)
    except (PermissionError, OSError) as exc:
        console.print(f"[yellow]Aviso:[/yellow] No se pudo borrar '{path}': {exc}")
        console.print("  Continuando sin borrar... (el modo overwrite recreará el directorio)")


def cleanup_dataset_output(out_base: Path, dataset_name: str, label: str = "dataset") -> None:
    target = out_base / dataset_name

    console.print(f"[yellow]Limpiando salida {label} (overwrite)...[/yellow]")
    if target.exists():
        console.print(f"  - borrando {target}")
        safe_remove_dir(target)
    else:
        console.print(f"  - no existe, se creara: {target}")


def _safe_partition_value(v) -> str:
    s = str(v)
    s = s.replace("/", "-").replace("\\", "-").replace(":", "-").strip()
    return s


def write_partitioned_dataset(
    df: pd.DataFrame,
    out_dir: Path,
    partition_cols: Iterable[str],
) -> None:
    if df.empty:
        return

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    part_cols = list(partition_cols)

    missing_part_cols = [c for c in part_cols if c not in df.columns]
    if missing_part_cols:
        raise ValueError(
            "No se puede particionar: faltan columnas de particion "
            f"en el dataset final: {missing_part_cols}"
        )

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    if "timestamp_hour" in df.columns:
        df["timestamp_hour"] = pd.to_datetime(df["timestamp_hour"], errors="coerce")

    for keys, g in df.groupby(part_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        part_dir = out_dir
        for col, val in zip(part_cols, keys):
            part_dir = part_dir / f"{col}={_safe_partition_value(val)}"
        part_dir.mkdir(parents=True, exist_ok=True)

        fname = f"part_{uuid.uuid4().hex}.parquet"
        g.to_parquet(part_dir / fname, index=False, engine="pyarrow")


def list_all_parquets(base: Path) -> List[Path]:
    base = Path(base).resolve()
    if not base.exists():
        return []
    return sorted([p for p in base.rglob("*.parquet") if p.is_file()])


def iter_month_partitions(
    base: Path,
    year_prefix: str = "year=",
    month_prefix: str = "month=",
) -> Iterator[Tuple[int, int, List[Path]]]:
    """
    Busca parquets en estructura tipo:
      <base>/.../year=YYYY/month=MM/*.parquet
    """
    base = Path(base).resolve()
    if not base.exists():
        raise FileNotFoundError(f"No existe: {base}")

    month_map: Dict[Tuple[int, int], List[Path]] = {}

    for fp in base.rglob("*.parquet"):
        if not fp.is_file():
            continue

        year: Optional[int] = None
        month: Optional[int] = None

        for part in fp.parts:
            if part.startswith(year_prefix):
                try:
                    year = int(part.split("=", 1)[1])
                except Exception:
                    year = None
            elif part.startswith(month_prefix):
                try:
                    month = int(part.split("=", 1)[1])
                except Exception:
                    month = None

        if year is None or month is None:
            continue

        month_map.setdefault((year, month), []).append(fp)

    for (year, month) in sorted(month_map.keys()):
        yield year, month, sorted(month_map[(year, month)])


def resolve_layer2_input_path(path: Path) -> Path:
    """
    Hace robusta la ruta de entrada TLC ante la variacion standardized/standarized.
    """
    path = Path(path).resolve()
    if path.exists():
        return path

    p = str(path)
    alt = None
    if "standardized" in p:
        alt = Path(p.replace("standardized", "standarized"))
    elif "standarized" in p:
        alt = Path(p.replace("standarized", "standardized"))

    if alt is not None and alt.exists():
        console.print(
            f"[yellow]Aviso:[/yellow] in-dir no existe ({path}). "
            f"Se usara automaticamente: {alt}"
        )
        return alt.resolve()

    return path

