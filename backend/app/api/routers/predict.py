from fastapi import APIRouter
from backend.app.api.schemas.predict import PredictRequest, PredictResponse
from backend.app.services.predict_service import predict

router = APIRouter(tags=["predict"])

@router.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    return predict(req)