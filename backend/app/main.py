from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import settings
from backend.app.core.logging import setup_logging
from backend.app.api.routers import all_routers


def create_app() -> FastAPI:
    setup_logging()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    for r in all_routers:
        app.include_router(r, prefix=settings.api_prefix)

    @app.get("/")
    def root():
        return {"message": "Hola mundo", "api": settings.api_prefix}

    return app


app = create_app()