from typing import Literal, Optional

from pydantic import BaseModel, Field

from backend.app.api.schemas.common import Meta


class PredictRequest(BaseModel):
    zone_id: int = Field(..., ge=1, description="ID de zona, por ejemplo PULocationID")
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)

    rain: Optional[bool] = None
    events: Optional[int] = Field(default=None, ge=0)


class PredictResponse(BaseModel):
    zone_id: int
    hour: int
    day_of_week: int

    score: float = Field(..., ge=0.0, le=1.0)
    raw_stress: float
    is_stress: Optional[int] = None

    level: Literal["low", "medium", "high"]
    meta: Meta