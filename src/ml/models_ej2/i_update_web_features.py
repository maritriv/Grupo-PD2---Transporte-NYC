from __future__ import annotations

import argparse

import requests

from src.ml.models_ej2.h_build_web_features import build_web_features


DEFAULT_DATASET_DIR = "data/aggregated/ex_stress/df_stress_zone_hour_day"
DEFAULT_WEB_MODEL_DIR = "outputs/ejercicio2/web_model"
DEFAULT_BACKEND_URL = "http://127.0.0.1:8000/api"
DEFAULT_ADMIN_TOKEN = "macbrides-admin-token"


def reload_backend_model(backend_url: str, admin_token: str) -> None:
    url = f"{backend_url.rstrip('/')}/admin/reload-model"

    response = requests.post(
        url,
        headers={"X-Admin-Token": admin_token},
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"No se pudo recargar el backend. "
            f"Status={response.status_code}, body={response.text}"
        )

    print("Backend recargado correctamente.")
    print(response.json())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenera web_features.parquet localmente y recarga el backend si está activo."
    )
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--web-model-dir", default=DEFAULT_WEB_MODEL_DIR)
    parser.add_argument("--backend-url", default=DEFAULT_BACKEND_URL)
    parser.add_argument("--admin-token", default=DEFAULT_ADMIN_TOKEN)
    parser.add_argument("--skip-backend-reload", action="store_true")

    args = parser.parse_args()

    print("Regenerando web_features.parquet local...")
    build_web_features(
        dataset_dir=args.dataset_dir,
        web_model_dir=args.web_model_dir,
    )

    if not args.skip_backend_reload:
        print("Recargando backend local...")
        reload_backend_model(
            backend_url=args.backend_url,
            admin_token=args.admin_token,
        )

    print("Actualización local completada.")


if __name__ == "__main__":
    main()