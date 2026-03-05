from pydantic import BaseModel, Field
from datetime import datetime, timezone

class Meta(BaseModel):
    model_version: str = Field(default="mock-v0")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))