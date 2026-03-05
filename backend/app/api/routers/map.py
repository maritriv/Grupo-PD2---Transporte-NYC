from fastapi import APIRouter, Query
from backend.app.api.schemas.map import MapResponse
from backend.app.services.map_service import build_map

router = APIRouter(tags=["map"])

@router.get("/map", response_model=MapResponse)
def get_map(
    day_of_week: int = Query(..., ge=0, le=6),
    hour: int = Query(..., ge=0, le=23),
):
    return build_map(day_of_week=day_of_week, hour=hour)