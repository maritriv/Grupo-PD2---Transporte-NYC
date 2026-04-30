from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ml.models_ej2.common.io import resolve_project_path


DEFAULT_DATASET_DIR = "data/aggregated/ex_stress/df_stress_zone_hour_day"
DEFAULT_WEB_MODEL_DIR = "outputs/ejercicio2/web_model"

TARGET_COLS = {
    "target_stress_t1",
    "target_stress_t3",
    "target_stress_t24",
    "target_is_stress_t1",
    "target_is_stress_t3",
    "target_is_stress_t24",
}

DROP_FEATURE_COLS = {
    "date",
    "datetime",
    "pickup_datetime",
    "dropoff_datetime",
    "timestamp",
    "timestamp_hour",
    "_sort_time",
}

ZONE_CANDIDATES = [
    "zone_id",
    "PULocationID",
    "pulocationid",
    "pu_location_id",
    "location_id",
    "LocationID",
]

HOUR_CANDIDATES = ["hour", "pickup_hour"]
DOW_CANDIDATES = ["day_of_week", "dayofweek", "weekday"]


def sanitize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.replace("[", "(", regex=False)
        .str.replace("]", ")", regex=False)
        .str.replace("<", "_lt_", regex=False)
        .str.replace(">", "_gt_", regex=False)
    )
    return out


def read_partitioned_parquet(dataset_dir: str | Path) -> pd.DataFrame:
    base = resolve_project_path(dataset_dir)
    if not base.exists():
        raise FileNotFoundError(f"No existe el dataset: {base}")

    files = sorted(base.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No se encontraron parquets en: {base}")

    print(f"Leyendo parquets -> {len(files):,} archivos")
    return pd.concat([pd.read_parquet(fp) for fp in files], ignore_index=True)


def load_feature_columns(web_model_dir: str | Path) -> list[str]:
    path = resolve_project_path(web_model_dir) / "feature_columns.json"
    if not path.exists():
        raise FileNotFoundError(f"No existe feature_columns.json: {path}")

    with path.open("r", encoding="utf-8") as f:
        data: Any = json.load(f)

    if isinstance(data, dict) and "feature_columns" in data:
        return list(data["feature_columns"])

    if isinstance(data, list):
        return list(data)

    raise ValueError("feature_columns.json debe ser una lista o contener la clave 'feature_columns'.")


def first_existing_col(df: pd.DataFrame, candidates: list[str], name: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"No se encontró columna para {name}. Candidatas: {candidates}")


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "hour" not in out.columns:
        if "timestamp_hour" in out.columns:
            out["hour"] = pd.to_datetime(out["timestamp_hour"], errors="coerce").dt.hour
        elif "datetime" in out.columns:
            out["hour"] = pd.to_datetime(out["datetime"], errors="coerce").dt.hour
        elif "timestamp" in out.columns:
            out["hour"] = pd.to_datetime(out["timestamp"], errors="coerce").dt.hour

    if "day_of_week" not in out.columns:
        if "timestamp_hour" in out.columns:
            out["day_of_week"] = pd.to_datetime(out["timestamp_hour"], errors="coerce").dt.dayofweek
        elif "date" in out.columns:
            out["day_of_week"] = pd.to_datetime(out["date"], errors="coerce").dt.dayofweek
        elif {"year", "month", "day"}.issubset(out.columns):
            dt = pd.to_datetime(
                dict(
                    year=pd.to_numeric(out["year"], errors="coerce"),
                    month=pd.to_numeric(out["month"], errors="coerce"),
                    day=pd.to_numeric(out["day"], errors="coerce"),
                ),
                errors="coerce",
            )
            out["day_of_week"] = dt.dt.dayofweek

    return out


def build_web_features(
    dataset_dir: str = DEFAULT_DATASET_DIR,
    web_model_dir: str = DEFAULT_WEB_MODEL_DIR,
) -> Path:
    web_model_path = resolve_project_path(web_model_dir)

    # Crear carpeta si no existe
    if not web_model_path.exists():
        print(f"Creando directorio: {web_model_path}")
        web_model_path.mkdir(parents=True, exist_ok=True)

    feature_columns = load_feature_columns(web_model_path)

    df = read_partitioned_parquet(dataset_dir)
    df = add_time_columns(df)

    zone_col = first_existing_col(df, ZONE_CANDIDATES, "zona")
    hour_col = first_existing_col(df, HOUR_CANDIDATES, "hora")
    dow_col = first_existing_col(df, DOW_CANDIDATES, "día de semana")

    df["_web_zone_id"] = pd.to_numeric(df[zone_col], errors="coerce")
    df["_web_hour"] = pd.to_numeric(df[hour_col], errors="coerce")
    df["_web_day_of_week"] = pd.to_numeric(df[dow_col], errors="coerce")

    df = df.dropna(subset=["_web_zone_id", "_web_hour", "_web_day_of_week"])
    df["_web_zone_id"] = df["_web_zone_id"].astype(int)
    df["_web_hour"] = df["_web_hour"].astype(int)
    df["_web_day_of_week"] = df["_web_day_of_week"].astype(int)

    drop_cols = set(TARGET_COLS) | set(DROP_FEATURE_COLS)
    x = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore").copy()

    key_cols = ["_web_zone_id", "_web_hour", "_web_day_of_week"]

    for col in x.select_dtypes(include=["bool"]).columns:
        x[col] = x[col].astype(int)

    cat_cols = x.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    x = pd.get_dummies(x, columns=cat_cols, dummy_na=True)

    for col in x.columns:
        if col not in key_cols:
            x[col] = pd.to_numeric(x[col], errors="coerce")

    x = sanitize_feature_names(x)

    for col in feature_columns:
        if col not in x.columns:
            x[col] = 0

    x = x[key_cols + feature_columns].copy()
    x[feature_columns] = x[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)

    web_features = (
        x.groupby(key_cols, as_index=False)[feature_columns]
        .median(numeric_only=True)
        .reset_index(drop=True)
    )

    out_path = web_model_path / "web_features.parquet"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    web_features.to_parquet(out_path, index=False)

    metadata = {
        "dataset_dir": str(resolve_project_path(dataset_dir)),
        "web_features": str(out_path),
        "n_rows": int(len(web_features)),
        "n_features": int(len(feature_columns)),
        "zone_col_used": zone_col,
        "hour_col_used": hour_col,
        "day_of_week_col_used": dow_col,
        "mode": "typical_zone_hour_dayofweek_features",
    }

    metadata_path = web_model_path / "web_features_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"web_features.parquet creado en: {out_path}")
    print(f"Filas web: {len(web_features):,}")
    print(f"Features modelo: {len(feature_columns):,}")

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera features para predicción web del modelo EX2.")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--web-model-dir", default=DEFAULT_WEB_MODEL_DIR)

    args = parser.parse_args()

    build_web_features(
        dataset_dir=args.dataset_dir,
        web_model_dir=args.web_model_dir,
    )


if __name__ == "__main__":
    main()