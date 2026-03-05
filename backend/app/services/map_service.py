from backend.app.api.schemas.map import MapResponse, ZoneScore
from backend.app.api.schemas.common import Meta

def _to_level(score: float) -> str:
    if score < 0.33:
        return "low"
    if score < 0.66:
        return "medium"
    return "high"

def build_map(day_of_week: int, hour: int) -> MapResponse:
    # MOCK: devuelve unas zonas “fake” para que el frontend pinte ya
    fake_zone_ids = [48, 79, 142, 161, 236, 262, 43, 170, 90, 151]

    zones = []
    for i, zid in enumerate(fake_zone_ids):
        score = (i / (len(fake_zone_ids) - 1))  # 0..1
        zones.append(ZoneScore(zone_id=zid, score=score, level=_to_level(score)))

    return MapResponse(
        day_of_week=day_of_week,
        hour=hour,
        zones=zones,
        meta=Meta(model_version="mock-v0"),
    )