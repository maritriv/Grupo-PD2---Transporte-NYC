"""
Utilidades de configuración de memoria para entrenamiento local.
"""
from __future__ import annotations

import os
from typing import Optional


def get_max_workers(override: Optional[int] = None) -> int:
    """
    Calcula número de workers (processes) recomendados para sklearn parallelization.
    
    Args:
        override: Si se especifica, usar este valor directamente
    
    Returns:
        Número de workers
    """
    if override is not None:
        return max(1, override)
    
    # Heurística: usar N_CPU - 1, mínimo 1
    try:
        import multiprocessing
        n_cpu = multiprocessing.cpu_count()
        return max(1, n_cpu - 1)
    except Exception:
        return 1


def get_dask_blocksize() -> str:
    """
    Determina blocksize recomendado para Dask según disponibilidad de RAM.
    
    Returns:
        blocksize como string (ej: "64MB", "32MB", "128MB")
        - RAM <= 4GB: "32MB"
        - 4GB < RAM < 16GB: "64MB"
        - RAM >= 16GB: "128MB"
    """
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024**3)
    except Exception:
        available_gb = 4.0  # Default conservador
    
    if available_gb < 4:
        return "32MB"
    elif available_gb < 16:
        return "64MB"
    else:
        return "128MB"


def warn_memory_config() -> None:
    """Alerta si la memoria disponible es baja."""
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024**3)
        total_gb = psutil.virtual_memory().total / (1024**3)
        
        if available_gb < 2:
            print(
                f"MEMORIA BAJA: {available_gb:.1f}GB disponibles de {total_gb:.1f}GB.\n"
                f"   Considera:\n"
                f"   - Cerrar otras aplicaciones\n"
                f"   - Reducir --sample-frac o usar filtros de fecha\n"
                f"   - Aumentar Dask blocksize (--dask-blocksize '32MB')"
            )
    except Exception:
        pass
