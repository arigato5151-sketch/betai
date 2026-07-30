from collections.abc import Iterable
import secrets

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class OriginValidationMiddleware(BaseHTTPMiddleware):
    """Reject browser-originated state changes from untrusted origins."""

    UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

    def __init__(
        self,
        app,
        *,
        allowed_origins: Iterable[str],
        require_origin_header: bool,
        access_cookie_name: str,
        refresh_cookie_name: str,
        csrf_cookie_name: str,
        csrf_header_name: str,
    ) -> None:
        super().__init__(app)
        self.allowed_origins = frozenset(
            origin.rstrip("/") for origin in allowed_origins
        )
        self.require_origin_header = require_origin_header
        self.auth_cookie_names = (access_cookie_name, refresh_cookie_name)
        self.csrf_cookie_name = csrf_cookie_name
        self.csrf_header_name = csrf_header_name

    async def dispatch(self, request: Request, call_next):
        if request.method not in self.UNSAFE_METHODS:
            return await call_next(request)

        origin = request.headers.get("origin")
        normalized_origin = origin.rstrip("/") if origin else None
        if normalized_origin and normalized_origin not in self.allowed_origins:
            return JSONResponse(
                status_code=403, content={"detail": "İstek kaynağına güvenilmiyor."}
            )
        if not normalized_origin and self.require_origin_header:
            return JSONResponse(
                status_code=403, content={"detail": "Origin başlığı gereklidir."}
            )

        auth_cookie_present = any(
            request.cookies.get(name) for name in self.auth_cookie_names
        )
        csrf_exempt = request.url.path in {
            "/api/auth/login",
            "/api/auth/register",
        }
        if auth_cookie_present and not csrf_exempt:
            cookie_token = request.cookies.get(self.csrf_cookie_name, "")
            header_token = request.headers.get(self.csrf_header_name, "")
            if not cookie_token or not secrets.compare_digest(
                cookie_token, header_token
            ):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "CSRF belirteci eksik veya geçersiz."},
                )
        return await call_next(request)
