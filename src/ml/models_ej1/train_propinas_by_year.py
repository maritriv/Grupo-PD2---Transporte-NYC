"""
Script wrapper para entrenar modelos de propinas por año.

Ejecuta el modelo para cada año de forma independiente, generando:
- Modelos por año
- Reportes separados por año
- Leaderboard consolidado

Uso:
    uv run -m src.ml.models_ej1.train_propinas_by_year

    # Con opciones personalizadas
    uv run -m src.ml.models_ej1.train_propinas_by_year --years 2023 2024 --models random_forest hist_gradient_boosting
    
    # Solo año 2024
    uv run -m src.ml.models_ej1.train_propinas_by_year --years 2024
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.ml.models_ej1.model_b_propinas import run_training, SUPPORTED_MODEL_NAMES
from config.settings import obtener_ruta
from config.spark_manager import SparkManager

console = Console()


def get_year_date_range(year: int) -> tuple[str, str]:
    """Retorna el rango de fechas para el segundo semestre de un año."""
    return f"{year}-10-01", f"{year}-12-31"


def run_training_by_year(
    years: list[int],
    out_subdir: str = "ml/propinas",
    input_subdir: str = "aggregated/ex1b/df_trip_level_tips",
    target_col: str = "target_tip_amount",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    random_state: int = 42,
    save_model_flag: bool = True,
    feature_scope: str = "strict_apriori",
    model_names: list[str] | None = None,
    use_spark: bool = True,
    n_jobs: int | None = None,
) -> dict[int, dict[str, Any]]:
    """
    Entrena modelos de propinas para cada año de forma independiente.
    
    Estructura de outputs generada:
        outputs/ml/propinas/
        ├── by_year/
        │   ├── 2023/
        │   │   ├── dataset_profile.json
        │   │   ├── training_summary.json
        │   │   ├── {model}_report.json          (uno por cada modelo entrenado)
        │   │   ├── {model}.pkl                  (modelo serializado)
        │   │   ├── {model}_feature_importance.csv
        │   │   └── {model}_test_predictions.parquet
        │   ├── 2024/
        │   │   └── (idem para 2024)
        │   ├── 2025/
        │   │   └── (idem para 2025)
        │   └── consolidation_report.json        (reporte consolidado de todos los años)
    
    Args:
        years: Lista de años a procesar
        out_subdir: Subdirectorio bajo outputs/ (default: ml/propinas)
        input_subdir: Subdirectorio bajo data/ para los datos de entrada
        target_col: Target a predecir
        train_frac: Fracción de train
        val_frac: Fracción de val
        random_state: Semilla aleatoria
        save_model_flag: Si guardar los modelos
        feature_scope: Alcance de features
        model_names: Lista de modelos a entrenar
        use_spark: Si usar Spark
        n_jobs: Número de jobs paralelos
    
    Returns:
        Diccionario con resultados por año
    """
    # Resolver rutas usando obtener_ruta desde config.settings
    out_base_dir = obtener_ruta('outputs') / out_subdir
    input_dir = obtener_ruta('data') / input_subdir
    
    all_results = {}
    
    for year in years:
        console.print(Panel(f"[bold cyan]Procesando año {year}[/bold cyan]", expand=False))
        
        min_date, max_date = get_year_date_range(year)
        
        # Crear directorio específico para el año: outputs/ml/propinas/by_year/{año}
        year_out_dir = out_base_dir / "by_year" / str(year)
        
        try:
            result = run_training(
                input_dir=str(input_dir),
                out_dir=str(year_out_dir),
                target_col=target_col,
                train_frac=train_frac,
                val_frac=val_frac,
                random_state=random_state,
                save_model_flag=save_model_flag,
                min_date=min_date,
                max_date=max_date,
                feature_scope=feature_scope,
                model_names=model_names,
                use_spark=use_spark,
                n_jobs=n_jobs,
            )
            
            all_results[year] = result
            console.print(f"[green]Año {year} procesado exitosamente[/green]\n")
            
        except Exception as e:
            console.print(f"[red]Error procesando año {year}: {e}[/red]\n")
            all_results[year] = {"error": str(e)}
    
    # Generar reporte consolidado: outputs/ml/propinas/by_year/consolidation_report.json
    consolidation_path = out_base_dir / "by_year" / "consolidation_report.json"
    consolidation_path.parent.mkdir(parents=True, exist_ok=True)
    
    with consolidation_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "years": years,
                "target_col": target_col,
                "feature_scope": feature_scope,
                "results": all_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    
    # Mostrar leaderboard consolidado
    show_consolidated_leaderboard(all_results)
    
    console.print(f"\n[bold green]Reporte consolidado guardado en: {consolidation_path}[/bold green]")
    
    return all_results


def show_consolidated_leaderboard(results: dict[int, dict[str, Any]]) -> None:
    """Muestra un leaderboard consolidado por año."""
    leaderboard = Table(title="Leaderboard Consolidado por Año", header_style="bold green")
    leaderboard.add_column("Año", style="bold white")
    leaderboard.add_column("Modelo")
    leaderboard.add_column("Split")
    leaderboard.add_column("MAE", justify="right")
    leaderboard.add_column("RMSE", justify="right")
    leaderboard.add_column("R2", justify="right")
    
    for year in sorted(results.keys()):
        result = results[year]
        
        if "error" in result:
            leaderboard.add_row(str(year), "ERROR", "-", "-", "-", "-")
            continue
        
        models = result.get("models", {})
        for model_idx, (model_name, model_report) in enumerate(models.items()):
            metrics = model_report.get("metrics", {})
            
            # Mostrar split de validación
            val_metrics = metrics.get("val", {})
            leaderboard.add_row(
                str(year) if model_idx == 0 else "",
                model_name,
                "val",
                f"{val_metrics.get('mae', 0):.4f}",
                f"{val_metrics.get('rmse', 0):.4f}",
                f"{val_metrics.get('r2', 0):.4f}",
            )
            
            # Mostrar split de test
            test_metrics = metrics.get("test", {})
            leaderboard.add_row(
                "",
                model_name,
                "test",
                f"{test_metrics.get('mae', 0):.4f}",
                f"{test_metrics.get('rmse', 0):.4f}",
                f"{test_metrics.get('r2', 0):.4f}",
            )
    
    console.print(leaderboard)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena modelos de propinas por año (EX1b) usando Spark."
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2023, 2024, 2025],
        help="Años a procesar. Default: 2023 2024 2025",
    )
    parser.add_argument(
        "--input-subdir",
        default="aggregated/ex1b/df_trip_level_tips",
        help="Subdirectorio bajo data/ para los datos (default: aggregated/ex1b/df_trip_level_tips).",
    )
    parser.add_argument(
        "--out-subdir",
        default="ml/propinas",
        help="Subdirectorio bajo outputs/ (default: ml/propinas).",
    )
    parser.add_argument(
        "--target-col",
        choices=["target_tip_amount", "target_tip_pct"],
        default="target_tip_amount",
        help="Target a predecir.",
    )
    parser.add_argument("--train-frac", type=float, default=0.70, help="Fracción temporal para train.")
    parser.add_argument("--val-frac", type=float, default=0.15, help="Fracción temporal para val.")
    parser.add_argument("--random-state", type=int, default=42, help="Semilla aleatoria.")
    parser.add_argument(
        "--no-save-model",
        action="store_true",
        help="Si se activa, no serializa los modelos entrenados.",
    )
    parser.add_argument(
        "--feature-scope",
        choices=["strict_apriori", "all"],
        default="strict_apriori",
        help="Selección de features.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=SUPPORTED_MODEL_NAMES,
        default=None,
        help="Modelos a entrenar. Si no se indica, ejecuta todos los disponibles.",
    )
    parser.add_argument(
        "--no-spark",
        action="store_true",
        help="Desactiva Spark (solo para datasets pequeños).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Número de jobs paralelos para sklearn.",
    )
    args = parser.parse_args()

    # Validar que los años sean razonables
    for year in args.years:
        if year < 2000 or year > 2100:
            console.print(f"[red]Error: Año {year} no es válido[/red]")
            sys.exit(1)

    console.print(
        Panel(
            f"[bold cyan]Entrenando modelos de propinas por año[/bold cyan]\n"
            f"Años: {', '.join(map(str, sorted(args.years)))}\n"
            f"Target: {args.target_col}\n"
            f"Modelos: {', '.join(args.models or ['todos'])}\n"
            f"Backend: {'Spark' if not args.no_spark else 'Pandas'}",
            expand=False,
        )
    )

    results = run_training_by_year(
        years=sorted(args.years),
        out_subdir=args.out_subdir,
        input_subdir=args.input_subdir,
        target_col=args.target_col,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        random_state=args.random_state,
        save_model_flag=not args.no_save_model,
        feature_scope=args.feature_scope,
        model_names=args.models,
        use_spark=not args.no_spark,
        n_jobs=args.n_jobs,
    )

    # Resumen final
    successful = sum(1 for r in results.values() if "error" not in r)
    console.print(f"\n[bold green]Resumen:[/bold green] {successful}/{len(args.years)} años procesados exitosamente")
    SparkManager.stop_session()


if __name__ == "__main__":
    main()
