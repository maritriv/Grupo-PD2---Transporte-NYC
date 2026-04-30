from __future__ import annotations

import pandas as pd

from backend.app.api.schemas.common import Meta
from backend.app.api.schemas.map import MapResponse, ZoneScore
from backend.app.services.predict_service import _load_assets, _sigmoid, _to_level_from_raw


def build_map(day_of_week: int, hour: int) -> MapResponse:
    regressor, _classifier, feature_columns, web_features, model_version = _load_assets()

    df = web_features[
        (web_features["_web_day_of_week"] == day_of_week)
        & (web_features["_web_hour"] == hour)
    ].copy()

    # Fallback: si no hay combinación exacta, usar solo la hora
    if df.empty:
        df = web_features[web_features["_web_hour"] == hour].copy()

    # Fallback final: usar todo
    if df.empty:
        df = web_features.copy()

    x = df.reindex(columns=feature_columns, fill_value=0)
    x = x.apply(pd.to_numeric, errors="coerce").fillna(0)

    raw_preds = regressor.predict(x)

    zones = []
    for zone_id, raw_stress in zip(df["_web_zone_id"].astype(int), raw_preds):
        raw_stress = float(raw_stress)
        score = _sigmoid(raw_stress)

        zones.append(
            ZoneScore(
                zone_id=int(zone_id),
                score=float(score),
                level=_to_level_from_raw(raw_stress),
            )
        )

    return MapResponse(
        day_of_week=day_of_week,
        hour=hour,
        zones=zones,
        meta=Meta(model_version=model_version),
    )