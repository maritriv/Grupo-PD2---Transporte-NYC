# config/minio_client.py
"""
Cliente de MinIO para operaciones de almacenamiento en object storage.
"""
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from minio import Minio
from minio.error import S3Error
import logging

logger = logging.getLogger(__name__)


class MinioManager:
    """
    Gestor de operaciones con MinIO.
    
    Attributes:
        client: Cliente de MinIO
        default_bucket: Bucket por defecto para operaciones
    
    Ejemplos:
        >>> minio = MinioManager()
        >>> minio.subir_archivo("data.csv", "mi-bucket", "path/data.csv")
        >>> minio.descargar_archivo("mi-bucket", "path/data.csv", "local.csv")
        >>> archivos = minio.listar_archivos("mi-bucket", prefix="path/")
    """
    
    def __init__(self):
        """
        Inicializa el cliente de MinIO.
        """
        credentials_path = Path(__file__).resolve().parents[1] / "credentials.json"
        
        self.credentials = self._cargar_credenciales(credentials_path)
        self.client = self._crear_cliente()
        self.default_bucket = "pd2"
        self.base_dir = "macbrides/"
        
        logger.info(f"Cliente MinIO inicializado.")
    
    def _cargar_credenciales(self, path: Path) -> Dict[str, Any]:
        """Carga las credenciales desde el archivo JSON."""
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró credentials.json en {path}\n"
                "Asegúrate de tener un archivo credentials.json")
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _crear_cliente(self) -> Minio:
        """Crea y retorna el cliente de MinIO."""
        url = self.credentials.get("url", "https://minio.fdi.ucm.es")
        endpoint = url.replace("https://", "").replace("http://", "").rstrip("/")

        return Minio(
            endpoint=endpoint,
            access_key=self.credentials["accessKey"],
            secret_key=self.credentials["secretKey"],
            secure=url.startswith("https://"),
        )
    
    def subir_archivo(
        self,
        archivo_local: str | Path,
        objeto_nombre: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Sube un archivo a MinIO.
        
        Args:
            archivo_local: Ruta del archivo local a subir
            objeto_nombre: Nombre del objeto en MinIO (usa el nombre del archivo si no se especifica)
            metadata: Metadatos opcionales para el objeto
            
        Returns:
            Nombre del objeto subido
            
        Ejemplos:
            >>> minio.subir_archivo("data/raw/datos.csv", "procesados/datos.csv")
            >>> minio.subir_archivo("informe.pdf")
        """
        archivo_local = Path(archivo_local)
        bucket_name = self.default_bucket
        objeto_nombre = self.base_dir + objeto_nombre or archivo_local.name
        
        if not archivo_local.exists():
            raise FileNotFoundError(f"Archivo local no encontrado: {archivo_local}")
        
        try:
            # Subir archivo
            self.client.fput_object(
                bucket_name=bucket_name,
                object_name=objeto_nombre,
                file_path=str(archivo_local),
                metadata=metadata
            )
            
            logger.info(f"Archivo subido: {archivo_local} -> {bucket_name}/{objeto_nombre}")
            return objeto_nombre
            
        except S3Error as e:
            logger.error(f"Error al subir archivo: {e}")
            raise
    
    def descargar_archivo(
        self,
        objeto_nombre: str,
        archivo_destino: str | Path
    ) -> Path:
        """
        Descarga un archivo desde MinIO.
        
        Args:
            objeto_nombre: Nombre del objeto en MinIO
            archivo_destino: Ruta donde guardar el archivo
            
        Returns:
            Path del archivo descargado
            
        Ejemplos:
            >>> minio.descargar_archivo("mi-bucket", "datos/file.csv", "local/file.csv")
        """
        archivo_destino = Path(archivo_destino)
        bucket_name = self.default_bucket
        
        # Crear directorio destino si no existe
        archivo_destino.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self.client.fget_object(
                bucket_name=bucket_name,
                object_name=objeto_nombre,
                file_path=str(archivo_destino)
            )
            
            logger.info(f"Archivo descargado: {bucket_name}/{objeto_nombre} -> {archivo_destino}")
            return archivo_destino
            
        except S3Error as e:
            logger.error(f"Error al descargar archivo: {e}")
            raise
    
    def listar_archivos(
        self,
        prefix: str = "",
        recursive: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Lista los archivos en un bucket.
        
        Args:
            prefix: Prefijo para filtrar objetos
            recursive: Si True, lista recursivamente
            
        Returns:
            Lista de diccionarios con información de cada objeto
            
        Ejemplos:
            >>> archivos = minio.listar_archivos(prefix="datos/")
            >>> for arch in archivos:
            ...     print(f"{arch['nombre']} - {arch['tamaño']} bytes")
        """
        bucket_name = self.default_bucket
        prefix = self.base_dir + prefix
        
        try:
            objetos = self.client.list_objects(
                bucket_name=bucket_name,
                prefix=prefix,
                recursive=recursive
            )
            
            archivos = []
            for obj in objetos:
                archivos.append({
                    "nombre": obj.object_name,
                    "tamaño": obj.size,
                    "etag": obj.etag,
                    "ultima_modificacion": obj.last_modified,
                    "content_type": obj.content_type
                })
            
            logger.info(f"Encontrados {len(archivos)} archivos en {bucket_name}/{prefix}")
            return archivos
            
        except S3Error as e:
            logger.error(f"Error al listar archivos: {e}")
            raise
    
    def eliminar_archivo(self, objeto_nombre: str) -> bool:
        """
        Elimina un archivo de MinIO.
        
        Args:
            objeto_nombre: Nombre del objeto a eliminar
            
        Returns:
            True si se eliminó exitosamente
        """
        bucket_name = self.default_bucket
        objeto_nombre = self.base_dir + objeto_nombre
        try:
            self.client.remove_object(bucket_name, objeto_nombre)
            logger.info(f"Archivo eliminado: {bucket_name}/{objeto_nombre}")
            return True
            
        except S3Error as e:
            logger.error(f"Error al eliminar archivo: {e}")
            raise
    
    def archivo_existe(self, objeto_nombre: str) -> bool:
        """
        Verifica si un archivo existe en MinIO.
        
        Args:
            objeto_nombre: Nombre del objeto
            
        Returns:
            True si el archivo existe, False en caso contrario
        """
        bucket_name = self.default_bucket
        objeto_nombre = self.base_dir + objeto_nombre
        try:
            self.client.stat_object(bucket_name, objeto_nombre)
            return True
        except S3Error as e:
            if e.code == "NoSuchKey":
                return False
            raise
    
    def subir_directorio(
        self,
        directorio_local: str | Path,
        prefijo_destino: str = ""
    ) -> List[str]:
        """
        Sube todos los archivos de un directorio a MinIO.
        
        Args:
            directorio_local: Ruta del directorio local
            bucket_name: Nombre del bucket
            prefijo_destino: Prefijo para los objetos en MinIO
            
        Returns:
            Lista de nombres de objetos subidos
        """
        directorio_local = Path(directorio_local)
        bucket_name = self.default_bucket
        prefijo_destino = self.base_dir + prefijo_destino
        
        if not directorio_local.is_dir():
            raise ValueError(f"{directorio_local} no es un directorio válido")
        
        archivos_subidos = []
        
        for archivo in directorio_local.rglob("*"):
            if archivo.is_file():
                # Calcular ruta relativa
                ruta_relativa = archivo.relative_to(directorio_local)
                objeto_nombre = f"{prefijo_destino}/{ruta_relativa}".lstrip("/")
                
                self.subir_archivo(archivo, bucket_name, objeto_nombre)
                archivos_subidos.append(objeto_nombre)
        
        logger.info(f"Directorio subido: {len(archivos_subidos)} archivos desde {directorio_local}")
        return archivos_subidos


# Función de conveniencia para obtener una instancia del manager
def obtener_minio_manager() -> MinioManager:
    """
    Obtiene una instancia configurada de MinioManager.
        
    Returns:
        Instancia de MinioManager
        
    Ejemplos:
        >>> from config.minio_manager import obtener_minio_manager
        >>> minio = obtener_minio_manager()
        >>> minio.subir_archivo("datos.csv")
    """
    return MinioManager()