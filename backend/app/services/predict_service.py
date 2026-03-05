from backend.app.api.schemas.predict import PredictRequest, PredictResponse
from backend.app.api.schemas.common import Meta

def _to_level(score: float) -> str:
    if score < 0.33:
        return "low"
    if score < 0.66:
        return "medium"
    return "high"

def predict(req: PredictRequest) -> PredictResponse:
    # MOCK: lógica determinista simple (luego se cambia por el modelo real)
    # Ejemplo: “sube score por la tarde”
    base = 0.25
    bump = 0.45 if 16 <= req.hour <= 20 else 0.15
    score = min(1.0, base + bump)

    return PredictResponse(
        zone_id=req.zone_id,
        hour=req.hour,
        day_of_week=req.day_of_week,
        score=score,
        level=_to_level(score),
        meta=Meta(model_version="mock-v0"),
    )