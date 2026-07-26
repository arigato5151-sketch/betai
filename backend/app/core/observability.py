from __future__ import annotations

import threading
import time
import uuid
from collections import Counter, defaultdict

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from app.core.logging_config import logger


class RequestMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: Counter[tuple[str, str, int]] = Counter()
        self._duration_seconds: dict[tuple[str, str], float] = defaultdict(float)

    def record(
        self, method: str, path: str, status_code: int, duration_seconds: float
    ) -> None:
        with self._lock:
            self._requests[(method, path, status_code)] += 1
            self._duration_seconds[(method, path)] += duration_seconds

    def render_prometheus(self) -> str:
        lines = [
            "# HELP bet_ai_http_requests_total Total HTTP requests.",
            "# TYPE bet_ai_http_requests_total counter",
        ]
        with self._lock:
            requests = sorted(self._requests.items())
            durations = sorted(self._duration_seconds.items())
        for (method, path, status_code), count in requests:
            labels = f'method="{method}",path="{path}",status="{status_code}"'
            lines.append(f"bet_ai_http_requests_total{{{labels}}} {count}")
        lines.extend(
            [
                "# HELP bet_ai_http_request_duration_seconds_total Cumulative request duration.",
                "# TYPE bet_ai_http_request_duration_seconds_total counter",
            ]
        )
        for (method, path), duration in durations:
            labels = f'method="{method}",path="{path}"'
            lines.append(
                f"bet_ai_http_request_duration_seconds_total{{{labels}}} {duration:.6f}"
            )
        return "\n".join(lines) + "\n"


request_metrics = RequestMetrics()


class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
        finally:
            duration = time.perf_counter() - started
            route = request.scope.get("route")
            metric_path = getattr(route, "path", "<unmatched>")
            request_metrics.record(request.method, metric_path, status_code, duration)
            logger.info(
                "HTTP request completed",
                extra={
                    "request_id": request_id,
                    "http_method": request.method,
                    "http_path": request.url.path,
                    "http_status": status_code,
                    "duration_ms": round(duration * 1000.0, 2),
                },
            )
        response.headers["X-Request-ID"] = request_id
        return response
