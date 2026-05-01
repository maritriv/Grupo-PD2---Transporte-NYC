from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import os
from config.minio_manager import MinioManager
import numpy as np
import pandas as pd

from backend.app.api.schemas.common import Meta
from backend.app.api.schemas.predict import PredictRequest, PredictResponse

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WEB_MODEL_DIR = PROJECT_ROOT / "outputs" / "ejercicio2" / "web_model"

REGRESSOR_PATH = WEB_MODEL_DIR / "stress_regressor.joblib"
CLASSIFIER_PATH = WEB_MODEL_DIR / "stress_classifier.joblib"
FEATURE_COLUMNS_PATH = WEB_MODEL_DIR / "feature_columns.json"
WEB_FEATURES_PATH = WEB_MODEL_DIR / "web_features.parquet"
METADATA_PATH = WEB_MODEL_DIR / "model_metadata.json"

MINIO_WEB_FEATURES_OBJECT = "outputs/ejercicio2/web_model/web_features.parquet"

def _score_to_level(score: float) -> str:
    if score <= 0.33:
        return "low"
    if score <= 0.66:
        return "medium"
    return "high"


def _read_feature_columns() -> list[str]:
    with FEATURE_COLUMNS_PATH.open("r", encoding="utf-8") as f:
        data: Any = json.load(f)

    if isinstance(data, dict) and "feature_columns" in data:
        return list(data["feature_columns"])

    if isinstance(data, list):
        return list(data)

    raise ValueError("feature_columns.json debe ser lista o contener 'feature_columns'.")


def _read_model_version() -> str:
    if not METADATA_PATH.exists():
        return "xgboost-web"

    try:
        with METADATA_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return str(data.get("selected_model", "xgboost-web"))
    except Exception:
        return "xgboost-web"


def _sync_web_features_from_minio() -> None:
    use_minio = os.environ.get("USE_MINIO_WEB_FEATURES", "false").lower() == "true"

    if not use_minio:
        return

    minio = MinioManager()
    minio.descargar_archivo(
        objeto_nombre=f"macbrides/{MINIO_WEB_FEATURES_OBJECT}",
        archivo_destino=WEB_FEATURES_PATH,
    )


@lru_cache(maxsize=1)
def _load_assets():

    _sync_web_features_from_minio()

    missing = [
        str(path)
        for path in [REGRESSOR_PATH, FEATURE_COLUMNS_PATH, WEB_FEATURES_PATH]
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Faltan artefactos del modelo web: " + ", ".join(missing)
        )

    regressor = joblib.load(REGRESSOR_PATH)

    classifier = None
    if CLASSIFIER_PATH.exists():
        classifier = joblib.load(CLASSIFIER_PATH)

    feature_columns = _read_feature_columns()
    web_features = pd.read_parquet(WEB_FEATURES_PATH)
    model_version = _read_model_version()

    x_all = web_features.reindex(columns=feature_columns, fill_value=0)
    x_all = x_all.apply(pd.to_numeric, errors="coerce").fillna(0)

    raw_all = regressor.predict(x_all)
    raw_low = float(np.percentile(raw_all, 5))
    raw_high = float(np.percentile(raw_all, 95))

    if raw_high <= raw_low:
        raw_high = raw_low + 1.0

    calibration = {
        "raw_low": raw_low,
        "raw_high": raw_high,
    }

    return regressor, classifier, feature_columns, web_features, model_version, calibration


def _raw_to_score(raw_stress: float, calibration: dict) -> float:
    raw_low = calibration["raw_low"]
    raw_high = calibration["raw_high"]

    score = (raw_stress - raw_low) / (raw_high - raw_low)
    return float(np.clip(score, 0.0, 1.0))


def _select_feature_row(
    web_features: pd.DataFrame,
    zone_id: int,
    hour: int,
    day_of_week: int,
) -> pd.DataFrame:
    exact = web_features[
        (web_features["_web_zone_id"] == zone_id)
        & (web_features["_web_hour"] == hour)
        & (web_features["_web_day_of_week"] == day_of_week)
    ]

    if not exact.empty:
        return exact.head(1)

    fallback = web_features[
        (web_features["_web_zone_id"] == zone_id)
        & (web_features["_web_hour"] == hour)
    ]
    if not fallback.empty:
        return fallback.head(1)

    fallback = web_features[web_features["_web_zone_id"] == zone_id]
    if not fallback.empty:
        return fallback.head(1)

    fallback = web_features[
        (web_features["_web_hour"] == hour)
        & (web_features["_web_day_of_week"] == day_of_week)
    ]
    if not fallback.empty:
        return fallback.head(1)

    return web_features.head(1)


def predict(req: PredictRequest) -> PredictResponse:
    regressor, classifier, feature_columns, web_features, model_version, calibration = _load_assets()

    row = _select_feature_row(
        web_features=web_features,
        zone_id=req.zone_id,
        hour=req.hour,
        day_of_week=req.day_of_week,
    )

    x = row.reindex(columns=feature_columns, fill_value=0)
    x = x.apply(pd.to_numeric, errors="coerce").fillna(0)

    raw_stress = float(regressor.predict(x)[0])
    score = _raw_to_score(raw_stress, calibration)

    is_stress = None
    if classifier is not None:
        is_stress = int(classifier.predict(x)[0])

    return PredictResponse(
        zone_id=req.zone_id,
        hour=req.hour,
        day_of_week=req.day_of_week,
        score=score,
        raw_stress=raw_stress,
        is_stress=is_stress,
        level=_score_to_level(score),
        meta=Meta(model_version=model_version),
    )


def predict_zone_at(zone_id: int, hour: int, day_of_week: int) -> dict:
    req = PredictRequest(
        zone_id=zone_id,
        hour=hour,
        day_of_week=day_of_week,
    )
    pred = predict(req)

    return {
        "zone_id": pred.zone_id,
        "hour": pred.hour,
        "day_of_week": pred.day_of_week,
        "score": pred.score,
        "raw_stress": pred.raw_stress,
        "is_stress": pred.is_stress,
        "level": pred.level,
    }


def build_zone_forecast(zone_id: int, hour: int, day_of_week: int) -> dict:
    offsets = [0, 2, 4, 6, 12, 24]

    forecast = []

    for offset in offsets:
        future_hour = (hour + offset) % 24
        future_day = (day_of_week + ((hour + offset) // 24)) % 7

        item = predict_zone_at(
            zone_id=zone_id,
            hour=future_hour,
            day_of_week=future_day,
        )

        item["hour_offset"] = offset
        item["time_label"] = "Ahora" if offset == 0 else f"+{offset}h"

        forecast.append(item)

    return {
        "zone_id": zone_id,
        "base_hour": hour,
        "base_day_of_week": day_of_week,
        "forecast": forecast,
        "meta": {
            "model_version": _read_model_version(),
        },
    }

def reload_assets() -> dict:
    _load_assets.cache_clear()
    _load_assets()

    return {
        "status": "ok",
        "message": "Modelo y features recargados correctamente",
        "model_dir": str(WEB_MODEL_DIR),
    }