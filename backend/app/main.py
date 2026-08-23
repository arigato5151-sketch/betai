from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
import secrets

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
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
    }


@app.get("/health/live", include_in_schema=False)
def liveness():
    return {"status": "ok"}


@app.get("/health/ready", include_in_schema=False)
def readiness(response: Response):
    database = get_database_status()
    cache_state = cache.status()
    database_ready = database["status"] == "ready"
    cache_ready = (
        cache_state["status"] == "ready" or settings.ENVIRONMENT != "production"
    )
    ready = database_ready and cache_ready
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ready" if ready else "not_ready",
        "database": database["status"],
        "cache": cache_state["status"],
    }


@app.get("/metrics", include_in_schema=False)
def metrics(request: Request):
    if settings.ENVIRONMENT == "production":
        token = request.headers.get("Authorization", "").removeprefix("Bearer ")
        if not settings.METRICS_TOKEN or not secrets.compare_digest(
            token, settings.METRICS_TOKEN
        ):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
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
