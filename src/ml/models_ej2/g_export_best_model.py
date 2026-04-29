from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.ml.models_ej2.common.io import ensure_project_dir, resolve_project_path, save_json


COMPARISON_CSV = "outputs/ejercicio2/comparison/model_comparison.csv"
WEB_MODEL_DIR = "outputs/ejercicio2/web_model"

TARGET_COL = "target_stress_t24"
TARGET_CLF_COL = "target_is_stress_t24"


MODEL_EXPORT_CONFIG = {
    "xgboost": {
        "module": "src.ml.models_ej2.c_model_boosting",
        "args": [
            "--target-reg", TARGET_COL,
            "--target-clf", TARGET_CLF_COL,
            "--outputs-dir", "outputs/ejercicio2/final_training/xgboost",
            "--fit-all-data",
        ],
        "artifacts": {
            "regressor": "outputs/ejercicio2/final_training/xgboost/xgboost_regressor_target_stress_t24.joblib",
            "classifier": "outputs/ejercicio2/final_training/xgboost/xgboost_classifier_target_is_stress_t24.joblib",
            "features": "outputs/ejercicio2/final_training/xgboost/xgboost_feature_columns.json",
            "report": "outputs/ejercicio2/final_training/xgboost/xgboost_report.json",
        },
    },
    "random_forest_spark": {
        "module": "src.ml.models_ej2.b_model_random_forest",
        "args": [
            "--target-col", TARGET_COL,
            "--outputs-dir", "outputs/ejercicio2/final_training/random_forest",
            "--fit-all-data",
        ],
        "artifacts": {
            "model_dir": "outputs/ejercicio2/final_training/random_forest/random_forest_stress_spark_model",
            "features": "outputs/ejercicio2/final_training/random_forest/random_forest_stress_spark_feature_importance.csv",
            "report": "outputs/ejercicio2/final_training/random_forest/random_forest_stress_spark_report.json",
        },
    },
    "GBTRegressor_spark": {
        "module": "src.ml.models_ej2.e_model_spark",
        "args": [
            "--target-col", TARGET_COL,
            "--outputs-dir", "outputs/ejercicio2/final_training/gbt",
            "--fit-all-data",
        ],
        "artifacts": {
            "model_dir": "outputs/ejercicio2/final_training/gbt/gbt_stress_spark_model",
            "features": "outputs/ejercicio2/final_training/gbt/gbt_stress_spark_feature_importance.csv",
            "report": "outputs/ejercicio2/final_training/gbt/gbt_stress_spark_report.json",
        },
    },
    "EmbeddingMLPRegressor_torch": {
        "module": "src.ml.models_ej2.d_model_nn",
        "args": [
            "--target-col", TARGET_COL,
            "--outputs-dir", "outputs/ejercicio2/final_training/neural_network",
            "--fit-all-data",
        ],
        "artifacts": {
            "model_pt": "outputs/ejercicio2/final_training/neural_network/neural_network_stress_model_all_data.pt",
            "report": "outputs/ejercicio2/final_training/neural_network/neural_network_stress_report.json",
        },
    },
}


def copy_any(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"No existe artefacto esperado: {src}")

    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def load_best_model(comparison_csv: str | Path) -> dict:
    path = resolve_project_path(comparison_csv)
    if not path.exists():
        raise FileNotFoundError(f"No existe la comparativa: {path}")

    df = pd.read_csv(path)
    df = df.dropna(subset=["rmse_test"]).copy()
    df = df[df["target"] == TARGET_COL]

    if df.empty:
        raise ValueError(f"No hay modelos válidos comparables para target={TARGET_COL}")

    return df.sort_values("rmse_test", ascending=True).iloc[0].to_dict()


def run_retraining(module: str, args: list[str]) -> None:
    cmd = [sys.executable, "-m", module, *args]

    print("Reentrenando mejor modelo:")
    print(" ".join(cmd))

    completed = subprocess.run(cmd, cwd=resolve_project_path("."))

    if completed.returncode != 0:
        raise RuntimeError(f"Falló el reentrenamiento de {module}")


def export_best_model(best: dict, web_model_dir: str | Path) -> None:
    model_name = str(best["model"])

    if model_name not in MODEL_EXPORT_CONFIG:
        raise ValueError(
            f"El mejor modelo es '{model_name}', pero no hay configuración de exportación para él."
        )

    config = MODEL_EXPORT_CONFIG[model_name]

    run_retraining(config["module"], config["args"])

    out_dir = ensure_project_dir(web_model_dir)

    for item in out_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    copied_artifacts = {}

    for key, src_rel in config["artifacts"].items():
        src = resolve_project_path(src_rel)

        if key == "regressor":
            dst = out_dir / "stress_regressor.joblib"
        elif key == "classifier":
            dst = out_dir / "stress_classifier.joblib"
        elif key == "features":
            dst = out_dir / "feature_columns.json" if src.suffix == ".json" else out_dir / src.name
        elif key == "report":
            dst = out_dir / "model_report.json"
        elif key == "model_dir":
            dst = out_dir / "spark_model"
        elif key == "model_pt":
            dst = out_dir / "stress_model.pt"
        else:
            dst = out_dir / src.name

        copy_any(src, dst)
        copied_artifacts[key] = str(dst)

    metadata = {
        "selected_model": model_name,
        "selection_metric": "rmse_test",
        "target": TARGET_COL,
        "comparison_row": best,
        "retrained_module": config["module"],
        "retrained_args": config["args"],
        "exported_artifacts": copied_artifacts,
    }

    save_json(metadata, out_dir / "model_metadata.json")

    print(f"Modelo exportado correctamente en: {out_dir}")
    print(f"Modelo seleccionado: {model_name}")
    print(f"RMSE test: {best['rmse_test']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Selecciona automáticamente el mejor modelo, lo reentrena y lo exporta para la web."
    )
    parser.add_argument("--comparison-csv", default=COMPARISON_CSV)
    parser.add_argument("--web-model-dir", default=WEB_MODEL_DIR)

    args = parser.parse_args()

    best = load_best_model(args.comparison_csv)
    export_best_model(best, args.web_model_dir)


if __name__ == "__main__":
    main()