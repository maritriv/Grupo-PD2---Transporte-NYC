# config/spark_manager.py

import findspark
import os
import sys
import logging
import subprocess
import pyspark
from pyspark.sql import SparkSession
from config.settings import cargar_config, obtener_ruta

# Configuración de logs
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%H:%M:%S]",
    handlers=[
        RichHandler(
            show_path=False,
            show_time=True,
            rich_tracebacks=True,
            markup=True,
        )
    ]
)
logger = logging.getLogger("SparkManager")        

class SparkManager:
    _session = None

    @classmethod
    def get_session(cls):
        """Implementación Singleton para no crear múltiples sesiones."""
        if cls._session is None:
            cls._preparar_entorno()
            cls._session = cls._crear_sesion()
        return cls._session
    
    @staticmethod
    def _preparar_entorno():
        """Configura dinámicamente JAVA_HOME antes de iniciar."""
        try:
            findspark.init()
            # 2. Configurar JAVA_HOME si no está puesto
            if "JAVA_HOME" not in os.environ:
                try:
                    # En Linux/WSL/Mac, intentamos encontrar la ruta de java automáticamente
                    java_path = subprocess.check_output(['which', 'java']).decode().strip()
                    if java_path:
                        # Obtenemos la ruta base (quitando /bin/java)
                        java_home = os.path.dirname(os.path.dirname(os.path.realpath(java_path)))
                        os.environ["JAVA_HOME"] = java_home
                        logger.info(f"[dim]JAVA_HOME configurado automáticamente en:[/dim] {java_home}")
                except Exception:
                    logger.warning("No se pudo detectar JAVA_HOME automáticamente. Asegúrate de tener Java instalado.")
        except Exception as e:
            logger.error(f"Error preparando el entorno: {e}")

    @staticmethod
    def _crear_sesion():
        try:
            # 1. Inicializar findspark (Busca Spark en el sistema automáticamente)
            findspark.init()
            
            # 2. Cargar parámetros del config.yaml
            conf = cargar_config().get('spark', {})
            
            # 3. Definir rutas temporales dentro del proyecto (para que no ensucie el sistema)
            warehouse_path = str(obtener_ruta('data') / "spark-warehouse")
            temp_path = str(obtener_ruta('data') / "spark-temp")
            
            # Asegurar que existan
            os.makedirs(temp_path, exist_ok=True)

            logger.info(f"[bold yellow]Iniciando Spark Session:[/bold yellow] {conf.get('app_name', 'NYC_SaaS')}")

            log4j_path = obtener_ruta('config') / "log4j2.properties"

            builder = (
                SparkSession.builder
                .appName(conf.get('app_name', 'NYC_Taxi'))
                .master(conf.get('master', 'local[*]'))
                .config("spark.driver.memory", conf.get('driver_memory', '4g'))
                .config("spark.executor.memory", conf.get('executor_memory', '2g'))
                .config("spark.sql.warehouse.dir", warehouse_path)
                .config("spark.local.dir", temp_path)
                .config("spark.ui.port", conf.get('ui_port', 4050))
                .config("spark.sql.shuffle.partitions", conf.get('shuffle_partitions', 16))
                .config("spark.ui.showConsoleProgress", "false")
                .config("spark.sql.execution.arrow.pyspark.enabled", "true") # Acelera Pandas <-> Spark
                .config(
                    "spark.driver.extraJavaOptions",
                    f"-Djava.io.tmpdir={temp_path} -Dlog4j.configurationFile=file:{log4j_path}"
                )
                .config(
                    "spark.executor.extraJavaOptions",
                    f"-Djava.io.tmpdir={temp_path} -Dlog4j.configurationFile=file:{log4j_path}"
                )
            )

            # 4. Crear la sesión
            session = builder.getOrCreate()

            # --- SILENCIAR LOGS ---
            # Esto le dice al motor de Java que solo muestre errores críticos
            session.sparkContext.setLogLevel("ERROR")
            
            # También silenciamos los logs de Py4J (la comunicación Python-Java)
            logging.getLogger("py4j").setLevel(logging.ERROR)

            logger.info("[bold yellow]Spark Session creada con éxito.[/bold yellow]")
            return session

        except Exception as e:
            logger.error(f"Error al iniciar Spark: {e}")
            raise e

    @staticmethod
    def stop_session():
        if SparkManager._session:
            SparkManager._session.stop()
            SparkManager._session = None
            logger.info("[bold yellow]Spark Session detenida.[/bold yellow]")