# config/settings.py
"""
Carga la configuración desde config.yaml
"""
import yaml
from typing import List, Dict, Any
from pathlib import Path

_RAIZ_PROYECTO = Path(__file__).resolve().parents[1]

def cargar_config():
    """Carga el archivo config.yaml y retorna un diccionario con la configuración."""
    raiz = _RAIZ_PROYECTO
    config_path = raiz / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"No se encontró config.yaml en {raiz}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def obtener_ruta(tipo: str) -> Path:
    """
    Obtiene la ruta absoluta a cualquier directorio del proyecto.
    
    Args:
        tipo: Puede ser una ruta del config.yaml usando notación de punto
              Ejemplos: 'datos', 'outputs', o subcarpetas como 'datos/raw'
    
    Returns:
        Path absoluto al directorio
    
    Ejemplos:
        obtener_ruta('datos')           # proyecto/data/
        obtener_ruta('datos/raw')       # proyecto/data/raw/
        obtener_ruta('outputs')         # proyecto/outputs/
        obtener_ruta('outputs/figures') # proyecto/outputs/figures/
    """
    config = cargar_config()
    raiz = _RAIZ_PROYECTO
    
    # Dividir el tipo en partes (ej: 'datos/raw' -> ['datos', 'raw'])
    partes = tipo.split('/')
    
    # Primera parte es la clave principal en config['rutas']
    clave_principal = partes[0]
    
    if clave_principal not in config['rutas']:
        raise ValueError(
            f"Ruta '{clave_principal}' no encontrada en config.yaml. "
            f"Disponibles: {list(config['rutas'].keys())}"
        )
    
    # Construir la ruta base
    ruta = raiz / config['rutas'][clave_principal]
    
    # Añadir subcarpetas si las hay
    for subcarpeta in partes[1:]:
        ruta /= subcarpeta
    
    return ruta

def obtener_servicios_habilitados() -> List[str]:
    """
    Obtiene los servicios habilitados para descargar

    Returns:
        Lista con los servicios habilitados
    """

    config = cargar_config()
    servicios = config['servicios']
    return [servicio for servicio in servicios.keys() if servicios[servicio]['habilitado']]

def obtener_config_eventos() -> Dict[str, Any]:
    """
    Devuelve la configuración de eventos NYC.
    """
    config = cargar_config()
    return config.get("eventos", {})


# Cargar configuración al importar el módulo
config = cargar_config()
servicios_habilitados = obtener_servicios_habilitados()
eventos_config = obtener_config_eventos()