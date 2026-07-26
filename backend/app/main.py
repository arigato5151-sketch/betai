from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from app.core.api_mode import get_api_mode
from app.core.config import SECRET_SOURCE_STATUS, settings
from app.core.logging_config import logger
from app.core.security import OriginValidationMiddleware
from app.core.observability import ObservabilityMiddleware, request_metrics
from app.services.cache import cache
from app.api.endpoints import router as api_router
from app.db.session import get_database_status
from app.prediction.ml.model import ml_pipeline


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    logger.info("Initializing platform startup hooks...")
    await cache.connect()
    ml_pipeline.load_active_model()
    try:
        yield
    finally:
        await cache.close()


app = FastAPI(title="Bet AI Pro Platform", version="3.0.0", lifespan=lifespan)

# Setup CORS with secure defaults
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "X-Requested-With",
        settings.CSRF_HEADER_NAME,
    ],
)
app.add_middleware(ObservabilityMiddleware)
app.add_middleware(
    OriginValidationMiddleware,
    allowed_origins=settings.BACKEND_CORS_ORIGINS,
    require_origin_header=settings.REQUIRE_ORIGIN_HEADER,
    access_cookie_name=settings.ACCESS_TOKEN_COOKIE_NAME,
    refresh_cookie_name=settings.REFRESH_TOKEN_COOKIE_NAME,
    csrf_cookie_name=settings.CSRF_COOKIE_NAME,
    csrf_header_name=settings.CSRF_HEADER_NAME,
)

FRONTEND_DIR = Path(settings.FRONTEND_DIST_DIR)


app.include_router(api_router, prefix="/api")


# Serve static frontend web interface
if FRONTEND_DIR.is_dir():
    assets_dir = FRONTEND_DIR / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "Bet AI Pro Platform",
        "api_mode": get_api_mode(settings.API_FOOTBALL_KEY),
        "frontend_url": settings.FRONTEND_URL,
        "ml_ready": ml_pipeline.is_ready,
        "ml_samples": ml_pipeline.metrics.get("samples", 0),
        "ml_min_samples": settings.MIN_TRAINING_SAMPLES,
        "active_model": ml_pipeline.active_model_name,
        "ml": ml_pipeline.status(),
        "allowed_origins": settings.BACKEND_CORS_ORIGINS,
        "cache": cache.status(),
        "secrets": SECRET_SOURCE_STATUS,
        "database": get_database_status(),
        "database_fallback_active": get_database_status()["fallback_active"],
    }


@app.get("/health/live", include_in_schema=False)
def liveness():
    return {"status": "ok"}


@app.get("/health/ready", include_in_schema=False)
def readiness(response: Response):
    database = get_database_status()
    ready = database["status"] in {"ready", "degraded"}
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ready" if ready else "not_ready",
        "database": database["status"],
        "cache": cache.status()["status"],
    }


@app.get("/metrics", include_in_schema=False)
def metrics():
    return PlainTextResponse(
        request_metrics.render_prometheus(),
        media_type="text/plain; version=0.0.4",
    )


@app.get("/ui")
def serve_frontend():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.is_file():
        return {"error": "Frontend build files not found."}
    return FileResponse(index_path)


@app.get("/ui/{path:path}", include_in_schema=False)
def serve_frontend_route(path: str):
    """Serve the Vite SPA entry point for client-side routes."""
    return serve_frontend()
