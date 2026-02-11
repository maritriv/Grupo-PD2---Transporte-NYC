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

from config.settings import obtener_bucket_default

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
    
    def __init__(self, default_bucket: str, credentials_path: Optional[Path] = None):
        """
        Inicializa el cliente de MinIO.
        
        Args:
            credentials_path: Ruta al archivo credentials.json
                            Si no se proporciona, busca en la raíz del proyecto
            default_bucket: Nombre del bucket por defecto
        """
        if credentials_path is None:
            # Asumir que está en la raíz del proyecto
            credentials_path = Path(__file__).resolve().parents[1] / "credentials.json"
        
        self.credentials = self._cargar_credenciales(credentials_path)
        self.client = self._crear_cliente()
        self.default_bucket = default_bucket
        
        logger.info(f"Cliente MinIO inicializado. Endpoint: {self.credentials['endpoint']}")
    
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
        return Minio(
            endpoint="play.min.io",
            access_key=self.credentials["access_key"],
            secret_key=self.credentials["secret_key"],
            secure=self.credentials.get("secure", True)
        )
    
    def crear_bucket(self, bucket_name: str) -> bool:
        """
        Crea un bucket si no existe.
        
        Args:
            bucket_name: Nombre del bucket
            
        Returns:
            True si se creó el bucket, False si ya existía
        """
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                logger.info(f"Bucket '{bucket_name}' creado exitosamente")
                return True
            else:
                logger.info(f"Bucket '{bucket_name}' ya existe")
                return False
        except S3Error as e:
            logger.error(f"Error al crear bucket '{bucket_name}': {e}")
            raise
    
    def subir_archivo(
        self,
        archivo_local: str | Path,
        bucket_name: Optional[str] = None,
        objeto_nombre: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Sube un archivo a MinIO.
        
        Args:
            archivo_local: Ruta del archivo local a subir
            bucket_name: Nombre del bucket (usa default_bucket si no se especifica)
            objeto_nombre: Nombre del objeto en MinIO (usa el nombre del archivo si no se especifica)
            metadata: Metadatos opcionales para el objeto
            
        Returns:
            Nombre del objeto subido
            
        Ejemplos:
            >>> minio.subir_archivo("data/raw/datos.csv", "mi-bucket", "procesados/datos.csv")
            >>> minio.subir_archivo("informe.pdf")  # Usa bucket por defecto
        """
        archivo_local = Path(archivo_local)
        bucket_name = bucket_name or self.default_bucket
        objeto_nombre = objeto_nombre or archivo_local.name
        
        if not bucket_name:
            raise ValueError("Se debe especificar bucket_name o configurar default_bucket")
        
        if not archivo_local.exists():
            raise FileNotFoundError(f"Archivo local no encontrado: {archivo_local}")
        
        try:
            # Crear bucket si no existe
            self.crear_bucket(bucket_name)
            
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
        bucket_name: str,
        objeto_nombre: str,
        archivo_destino: str | Path
    ) -> Path:
        """
        Descarga un archivo desde MinIO.
        
        Args:
            bucket_name: Nombre del bucket
            objeto_nombre: Nombre del objeto en MinIO
            archivo_destino: Ruta donde guardar el archivo
            
        Returns:
            Path del archivo descargado
            
        Ejemplos:
            >>> minio.descargar_archivo("mi-bucket", "datos/file.csv", "local/file.csv")
        """
        archivo_destino = Path(archivo_destino)
        
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
        bucket_name: Optional[str] = None,
        prefix: str = "",
        recursive: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Lista los archivos en un bucket.
        
        Args:
            bucket_name: Nombre del bucket (usa default_bucket si no se especifica)
            prefix: Prefijo para filtrar objetos
            recursive: Si True, lista recursivamente
            
        Returns:
            Lista de diccionarios con información de cada objeto
            
        Ejemplos:
            >>> archivos = minio.listar_archivos("mi-bucket", prefix="datos/")
            >>> for arch in archivos:
            ...     print(f"{arch['nombre']} - {arch['tamaño']} bytes")
        """
        bucket_name = bucket_name or self.default_bucket
        
        if not bucket_name:
            raise ValueError("Se debe especificar bucket_name o configurar default_bucket")
        
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
    
    def eliminar_archivo(self, bucket_name: str, objeto_nombre: str) -> bool:
        """
        Elimina un archivo de MinIO.
        
        Args:
            bucket_name: Nombre del bucket
            objeto_nombre: Nombre del objeto a eliminar
            
        Returns:
            True si se eliminó exitosamente
        """
        try:
            self.client.remove_object(bucket_name, objeto_nombre)
            logger.info(f"Archivo eliminado: {bucket_name}/{objeto_nombre}")
            return True
            
        except S3Error as e:
            logger.error(f"Error al eliminar archivo: {e}")
            raise
    
    def archivo_existe(self, bucket_name: str, objeto_nombre: str) -> bool:
        """
        Verifica si un archivo existe en MinIO.
        
        Args:
            bucket_name: Nombre del bucket
            objeto_nombre: Nombre del objeto
            
        Returns:
            True si el archivo existe, False en caso contrario
        """
        try:
            self.client.stat_object(bucket_name, objeto_nombre)
            return True
        except S3Error as e:
            if e.code == "NoSuchKey":
                return False
            raise
    
    def obtener_url_presigned(
        self,
        bucket_name: str,
        objeto_nombre: str,
        expiracion_segundos: int = 3600
    ) -> str:
        """
        Genera una URL pre-firmada para acceso temporal al archivo.
        
        Args:
            bucket_name: Nombre del bucket
            objeto_nombre: Nombre del objeto
            expiracion_segundos: Tiempo de expiración en segundos (default: 1 hora)
            
        Returns:
            URL pre-firmada
            
        Ejemplos:
            >>> url = minio.obtener_url_presigned("mi-bucket", "informe.pdf", 7200)
            >>> print(f"Compartir: {url}")
        """
        try:
            from datetime import timedelta
            
            url = self.client.presigned_get_object(
                bucket_name=bucket_name,
                object_name=objeto_nombre,
                expires=timedelta(seconds=expiracion_segundos)
            )
            
            logger.info(f"URL generada para {bucket_name}/{objeto_nombre} (expira en {expiracion_segundos}s)")
            return url
            
        except S3Error as e:
            logger.error(f"Error al generar URL presigned: {e}")
            raise
    
    def subir_directorio(
        self,
        directorio_local: str | Path,
        bucket_name: Optional[str] = None,
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
        bucket_name = bucket_name or self.default_bucket
        
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
        >>> from config.minio_client import obtener_minio_manager
        >>> minio = obtener_minio_manager()
        >>> minio.subir_archivo("datos.csv")
    """
    bucket_por_defecto = obtener_bucket_default()
    return MinioManager(default_bucket=bucket_por_defecto)