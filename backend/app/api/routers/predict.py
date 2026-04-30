from fastapi import APIRouter, Query

from backend.app.api.schemas.predict import PredictRequest, PredictResponse
from backend.app.services.predict_service import predict, build_zone_forecast

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    return predict(req)


@router.get("/predict/forecast")
def forecast_endpoint(
    zone_id: int = Query(..., ge=1),
    hour: int = Query(..., ge=0, le=23),
    day_of_week: int = Query(..., ge=0, le=6),
):
    return build_zone_forecast(
        zone_id=zone_id,
        hour=hour,
        day_of_week=day_of_week,
    )