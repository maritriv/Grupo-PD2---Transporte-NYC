from __future__ import annotations

import pandas as pd

from backend.app.api.schemas.common import Meta
from backend.app.api.schemas.map import MapResponse, ZoneScore
from backend.app.services.predict_service import _load_assets, _raw_to_score, _score_to_level


def build_map(day_of_week: int, hour: int) -> MapResponse:
    regressor, _classifier, feature_columns, web_features, model_version, calibration = _load_assets()

    df = web_features[
        (web_features["_web_day_of_week"] == day_of_week)
        & (web_features["_web_hour"] == hour)
    ].copy()

    if df.empty:
        df = web_features[web_features["_web_hour"] == hour].copy()

    if df.empty:
        df = web_features.copy()

    x = df.reindex(columns=feature_columns, fill_value=0)
    x = x.apply(pd.to_numeric, errors="coerce").fillna(0)

    raw_preds = regressor.predict(x)

    zones = []
    for zone_id, raw_stress in zip(df["_web_zone_id"].astype(int), raw_preds):
        raw_stress = float(raw_stress)
        score = _raw_to_score(raw_stress, calibration)

        zones.append(
            ZoneScore(
                zone_id=int(zone_id),
                score=float(score),
                raw_stress=float(raw_stress),
                level=_score_to_level(score),
            )
        )

    return MapResponse(
        day_of_week=day_of_week,
        hour=hour,
        zones=zones,
        meta=Meta(model_version=model_version),
    )