"""
app/api/middleware.py - Rate limiting, API key auth, and request logging middleware
"""
import time
import hashlib
from collections import defaultdict, deque
from typing import Optional
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger


# ── In-memory rate limiter (swap for Redis in production) ─────────────────────

class RateLimiter:
    """
    Sliding window rate limiter.
    Default: 60 requests/minute per IP, 10 fusion requests/minute.
    """

    def __init__(self):
        self._windows: dict = defaultdict(deque)

    def is_allowed(self, key: str, limit: int, window_seconds: int = 60) -> bool:
        now = time.time()
        window = self._windows[key]

        # Remove expired entries
        while window and window[0] < now - window_seconds:
            window.popleft()

        if len(window) >= limit:
            return False

        window.append(now)
        return True

    def remaining(self, key: str, limit: int, window_seconds: int = 60) -> int:
        now = time.time()
        window = self._windows[key]
        active = sum(1 for t in window if t >= now - window_seconds)
        return max(0, limit - active)


_rate_limiter = RateLimiter()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Apply per-IP rate limiting."""

    LIMITS = {
        "/api/v1/fuse": (10, 60),           # 10 fusions per minute
        "/api/v1/fuse/enhanced": (5, 60),    # 5 enhanced fusions per minute
        "/api/v1/fuse/batch": (2, 60),       # 2 batch jobs per minute
        "/api/v1/search": (30, 60),          # 30 searches per minute
        "default": (60, 60),                 # 60 general requests per minute
    }

    async def dispatch(self, request: Request, call_next):
        client_ip = self._get_client_ip(request)
        path = request.url.path

        # Find applicable limit
        limit, window = self.LIMITS.get(path, self.LIMITS["default"])
        key = f"{client_ip}:{path}"

        if not _rate_limiter.is_allowed(key, limit, window):
            remaining = _rate_limiter.remaining(key, limit, window)
            logger.warning(f"Rate limit exceeded: {client_ip} on {path}")
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded. Please slow down.",
                    "limit": limit,
                    "window_seconds": window,
                    "retry_after": window,
                },
                headers={
                    "Retry-After": str(window),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": str(remaining),
                },
            )

        response = await call_next(request)
        remaining = _rate_limiter.remaining(key, limit, window)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response

    def _get_client_ip(self, request: Request) -> str:
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"


# ── Request/Response Logging ──────────────────────────────────────────────────

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests with timing."""

    SKIP_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000

        logger.info(
            f"{request.method} {request.url.path} "
            f"→ {response.status_code} "
            f"[{elapsed:.1f}ms]"
        )
        return response


# ── Optional API Key Auth ─────────────────────────────────────────────────────

class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Optional API key authentication.
    Enable by setting API_KEY_ENABLED=true in .env.
    """

    PUBLIC_PATHS = {"/health", "/docs", "/redoc", "/openapi.json", "/ui"}

    def __init__(self, app, api_keys: set, enabled: bool = False):
        super().__init__(app)
        self.api_keys = api_keys
        self.enabled = enabled

    async def dispatch(self, request: Request, call_next):
        if not self.enabled:
            return await call_next(request)

        # Allow public paths
        if any(request.url.path.startswith(p) for p in self.PUBLIC_PATHS):
            return await call_next(request)

        # Check API key
        api_key = (
            request.headers.get("X-API-Key")
            or request.query_params.get("api_key")
        )

        if not api_key or not self._validate_key(api_key):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key. Include X-API-Key header."},
            )

        return await call_next(request)

    def _validate_key(self, key: str) -> bool:
        # Hash-based comparison to avoid timing attacks
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return key_hash in {
            hashlib.sha256(k.encode()).hexdigest()
            for k in self.api_keys
        }
