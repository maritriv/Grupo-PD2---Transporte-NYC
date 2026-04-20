from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

try:
    import dask.dataframe as dd
except ImportError:
    dd = None


def read_partitioned_parquet_dir(base: Path, use_dask: bool = False, blocksize: str = "64MB") -> pd.DataFrame | dd.DataFrame:
    """
    Lee un directorio con parquets particionados tipo Spark.
    
    Soporta estructuras:
    - dir/*.parquet
    - dir/date=YYYY-MM-DD/*.parquet
    - dir/year=YYYY/month=MM/*.parquet
    
    Args:
        base: Ruta al directorio
        use_dask: Si True, devuelve Dask DataFrame (lazy). Si False, carga en memoria con pandas.
        blocksize: Tamaño de bloque para Dask (ej: "64MB", "32MB")
    
    Returns:
        pd.DataFrame si use_dask=False, dd.DataFrame si use_dask=True
    """
    base = Path(base).resolve()
    if not base.exists():
        raise FileNotFoundError(f"No existe: {base}")

    files = list(base.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No hay .parquet dentro de: {base}")

    if use_dask:
        if dd is None:
            raise ImportError("Instala dask[dataframe]: uv add 'dask[dataframe]'")
        # Dask detecta automáticamente particiones
        return dd.read_parquet(str(base), blocksize=blocksize)
    else:
        # Cargar tradicional (para datasets pequeños)
        return pd.concat([pd.read_parquet(fp) for fp in sorted(files)], ignore_index=True)


def read_partitioned_parquet_dir_dask(
    base: Path | str,
    blocksize: str = "64MB",
) -> dd.DataFrame:
    """
    Lee un directorio particionado con Dask en chunks para evitar llenar memoria.
    
    Soporta estructuras tipo Spark:
    - dir/*.parquet
    - dir/date=YYYY-MM-DD/*.parquet
    - dir/year=YYYY/month=MM/*.parquet
    - dir/year=YYYY/month=MM/day=DD/*.parquet
    
    Args:
        base: Ruta al directorio raíz particionado
        blocksize: Tamaño máximo de bloque en memoria (default: 64MB)
    
    Returns:
        Dask DataFrame (lazy)
    """
    if dd is None:
        raise ImportError(
            "Debe instalar dask[dataframe] para cargar datasets grandes. "
            "Ejecuta: uv add 'dask[dataframe]'"
        )
    
    base = Path(base).resolve()
    if not base.exists():
        raise FileNotFoundError(f"No existe: {base}")
    
    files = list(base.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No hay .parquet en {base} (subdirectorios incluidos)")
    
    # Dask detecta automáticamente particiones cuando está bien la estructura
    # Simplemente pasa el directorio base
    ddf = dd.read_parquet(str(base), blocksize=blocksize)
    return ddf


def collect_dask_with_filter(
    ddf: dd.DataFrame,
    filter_func: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """
    Ejecuta un Dask DataFrame aplicando filtros por partición antes de colectar.
    
    Args:
        ddf: Dask DataFrame
        filter_func: Función que recibe un partition (pd.DataFrame) y devuelve un pd.DataFrame filtrado.
                     Útil para filtros de fecha que reducen datos antes de cargar en memoria.
    
    Returns:
        pd.DataFrame con los datos colectados
    """
    if filter_func is not None:
        ddf = ddf.map_partitions(filter_func, meta=ddf._meta)
    
    return ddf.compute()


def read_partitioned_parquet_dir_spark(
    base: Path | str,
) -> "pyspark.sql.DataFrame":
    """
    Lee un directorio particionado con Spark (lazy).
    
    Soporta estructuras tipo Spark:
    - dir/*.parquet
    - dir/date=YYYY-MM-DD/*.parquet
    - dir/year=YYYY/month=MM/*.parquet
    - dir/year=YYYY/month=MM/day=DD/*.parquet
    
    Args:
        base: Ruta al directorio raíz particionado
    
    Returns:
        PySpark DataFrame (lazy)
    """
    try:
        from config.spark_manager import SparkManager
    except ImportError:
        raise ImportError(
            "Debe instalar spark y config. "
            "Asegúrate de que SparkManager esté disponible."
        )
    
    base = Path(base).resolve()
    if not base.exists():
        raise FileNotFoundError(f"No existe: {base}")
    
    files = list(base.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No hay .parquet en {base} (subdirectorios incluidos)")
    
    spark = SparkManager.get_session()
    # Spark detecta automáticamente particiones cuando está bien la estructura
    return spark.read.parquet(str(base))


def _year_month_pairs(min_date: str, max_date: str) -> list[tuple[int, int]]:
    start = datetime.fromisoformat(min_date).replace(day=1)
    end = datetime.fromisoformat(max_date).replace(day=1)
    pairs: list[tuple[int, int]] = []
    while start <= end:
        pairs.append((start.year, start.month))
        if start.month == 12:
            start = start.replace(year=start.year + 1, month=1)
        else:
            start = start.replace(month=start.month + 1)
    return pairs


def collect_spark_with_filter(
    sdf: "pyspark.sql.DataFrame",
    min_date: str | None = None,
    max_date: str | None = None,
) -> pd.DataFrame:
    """
    Filtra un Spark DataFrame por fecha y lo convierte a Pandas.
    
    Args:
        sdf: Spark DataFrame
        min_date: Filtro de fecha mínima (string YYYY-MM-DD)
        max_date: Filtro de fecha máxima (string YYYY-MM-DD)
    
    Returns:
        pd.DataFrame con los datos colectados y filtrados
    """
    # Si el parquet está particionado con year/month y tenemos fecha, filtramos primero por partición.
    partition_cols = set(sdf.columns)
    if min_date and max_date and {"year", "month"}.issubset(partition_cols):
        pairs = _year_month_pairs(min_date, max_date)
        if pairs:
            condition = None
            for year, month in pairs:
                pair_cond = (sdf["year"] == year) & (sdf["month"] == month)
                condition = pair_cond if condition is None else condition | pair_cond
            if condition is not None:
                sdf = sdf.filter(condition)

    # Aplicar filtros en Spark ANTES de convertir a Pandas (más eficiente)
    if min_date is not None:
        sdf = sdf.filter(sdf["date"] >= min_date)
    if max_date is not None:
        sdf = sdf.filter(sdf["date"] <= max_date)

    # Asegurar que la conversión a Pandas no use Arrow en entornos de memoria limitada
    spark = sdf.sparkSession
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")

    return sdf.toPandas()


def ensure_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"[{name}] Faltan columnas: {miss}")


def safe_date_for_filename(s: str) -> str:
    return s.strip().replace("/", "-").replace("\\", "-").replace(":", "-")