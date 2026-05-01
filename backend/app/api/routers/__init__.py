from backend.app.api.routers.health import router as health_router
from backend.app.api.routers.predict import router as predict_router
from backend.app.api.routers.map import router as map_router
from backend.app.api.routers.admin import router as admin_router

all_routers = [health_router, predict_router, map_router, admin_router]