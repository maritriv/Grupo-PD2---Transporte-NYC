from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from backend.app.api.schemas.common import Meta

class ZoneScore(BaseModel):
    zone_id: int = Field(..., ge=1)
    score: float = Field(..., ge=0.0, le=1.0)
    raw_stress: Optional[float] = None
    level: Literal["low", "medium", "high"]

class MapResponse(BaseModel):
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    zones: List[ZoneScore]
    meta: Meta