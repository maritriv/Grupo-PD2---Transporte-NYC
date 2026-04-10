# src/procesamiento/capa1/core_io.py
import re
import gc
from pathlib import Path
from typing import Callable, Dict, Tuple, Optional, Union, Any
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console

def _extract_expected_year_month_from_filename(path: Path) -> Optional[Tuple[int, int]]:
    """
    Extrae YYYY-MM del nombre tipo:
        yellow_tripdata_2023-07.parquet
    Si no puede extraerlo, devuelve (None, None).
    """
    m = re.search(r"(\d{4})-(\d{2})", path.stem)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))

def procesar_archivo_en_batches(
    input_path: Path, 
    output_path: Path, 
    funcion_limpieza: Callable[
        [pd.DataFrame, Tuple[Optional[int], Optional[int]]],
        Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]
    ],
    batch_size: int = 500_000
) -> Dict[str, Any]:
    """
    Lee un parquet en batches, aplica la funcion_limpieza y lo guarda.
    Mantiene el uso de memoria RAM muy bajo.
    """
    pf = pq.ParquetFile(input_path)
    date = _extract_expected_year_month_from_filename(input_path)
    writer = None
    stats = {
        "n_rows": 0,
        "n_clean_rows": 0,
        "null_count": 0,
        "changed_rows": 0,
        "removed_rows": 0,
        "removed_reasons": {},
    }
    n_columns = 0
    console = Console()

    with console.status(f"[cyan]Validando {input_path.name} (batch_size={batch_size})..."):
        for batch in pf.iter_batches(batch_size=batch_size):
            df_batch = batch.to_pandas()
            stats["n_rows"] += len(df_batch)
            
            # Aplicamos la función inyectada (las reglas específicas del servicio)
            resultado_limpieza = funcion_limpieza(df_batch, date)
            if isinstance(resultado_limpieza, tuple):
                df_limpio, cleaning_stats = resultado_limpieza
                cleaning_stats = cleaning_stats or {}
            else:
                df_limpio = resultado_limpieza
                cleaning_stats = {}

            stats["n_clean_rows"] += len(df_limpio)
            removed_rows_batch = len(df_batch) - len(df_limpio)
            stats["removed_rows"] += int(cleaning_stats.get("removed_rows", removed_rows_batch))
            stats["changed_rows"] += int(cleaning_stats.get("changed_rows", 0))
            batch_reasons = cleaning_stats.get("removed_reasons", {}) or {}
            for reason, count in batch_reasons.items():
                stats["removed_reasons"][reason] = int(stats["removed_reasons"].get(reason, 0)) + int(count)

            if n_columns == 0:
                n_columns = len(df_limpio.columns)
            stats["null_count"] += df_limpio.isnull().sum().sum()
            
            if len(df_limpio) > 0:
                table = pa.Table.from_pandas(df_limpio)

                # Inicializamos el writer usando el schema del primer batch limpio
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema, compression='snappy')

                writer.write_table(table)
                del table
            
            del df_batch, df_limpio
            gc.collect()

    if writer:
        writer.close()

    stats["null_prct"] = stats["null_count"] * 100 / (stats["n_rows"] * n_columns) if stats["n_rows"] > 0 else None
    
    return stats
