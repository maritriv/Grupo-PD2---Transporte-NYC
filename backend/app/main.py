from fastapi import FastAPI
from backend.app.core.config import settings
from backend.app.core.logging import setup_logging
from backend.app.api.routers import all_routers

def create_app() -> FastAPI:
    setup_logging()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
    )

    for r in all_routers:
        app.include_router(r, prefix=settings.api_prefix)

    @app.get("/")
    def root():
        return {"message": "Hola mundo", "api": settings.api_prefix}

    return app

app = create_app()