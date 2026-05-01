from fastapi import APIRouter, Header, HTTPException

from backend.app.services.predict_service import reload_assets

router = APIRouter(tags=["admin"])


ADMIN_TOKEN = "macbrides-admin-token"


@router.post("/admin/reload-model")
def reload_model(x_admin_token: str | None = Header(default=None)):
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Token admin inválido")

    return reload_assets()