from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
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


def _sigmoid(x: float) -> float:
    try:
        return float(1.0 / (1.0 + math.exp(-x)))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _to_level_from_raw(raw_stress: float) -> str:
    if raw_stress < 1.0:
        return "low"
    if raw_stress < 3.0:
        return "medium"
    return "high"


def _read_feature_columns() -> list[str]:
    with FEATURE_COLUMNS_PATH.open("r", encoding="utf-8") as f:
        data: Any = json.load(f)

    if isinstance(data, dict) and "feature_columns" in data:
        return list(data["feature_columns"])

    if isinstance(data, list):
        return list(data)

    raise ValueError("feature_columns.json debe ser una lista o contener 'feature_columns'.")


def _read_model_version() -> str:
    if not METADATA_PATH.exists():
        return "xgboost-web"

    try:
        with METADATA_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return str(data.get("selected_model", "xgboost-web"))
    except Exception:
        return "xgboost-web"


@lru_cache(maxsize=1)
def _load_assets():
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

    return regressor, classifier, feature_columns, web_features, model_version


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
    regressor, classifier, feature_columns, web_features, model_version = _load_assets()

    row = _select_feature_row(
        web_features=web_features,
        zone_id=req.zone_id,
        hour=req.hour,
        day_of_week=req.day_of_week,
    )

    x = row.reindex(columns=feature_columns, fill_value=0)
    x = x.apply(pd.to_numeric, errors="coerce").fillna(0)

    raw_stress = float(regressor.predict(x)[0])
    score = _sigmoid(raw_stress)

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
        level=_to_level_from_raw(raw_stress),
        meta=Meta(model_version=model_version),
    )