"""
app/main.py - FastAPI application entry point
"""
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.config import settings
from app.api.routes import fusion, search, analyze, jobs, metrics, advanced
from app.api.middleware import RateLimitMiddleware, RequestLoggingMiddleware
from app.core.database import init_db

# ── Import the SAME global vector store instance ──────────────────────────────
from app.vector.store import vector_store


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    # Database
    await init_db()
    logger.info("✅ Database initialized")

    # Vector store  (sync call — no await)
    vector_store.initialize()
    app.state.vector_store = vector_store
    logger.info("✅ Vector store initialized")

    # Ensure directories exist
    os.makedirs("./data/faiss_index", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./frontend", exist_ok=True)

    logger.info("✅ Application ready")
    yield

    logger.info("🛑 Shutting down...")
    vector_store.save()
    logger.info("✅ Shutdown complete")


# ─── App factory ──────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="## Semantic Code Fusion v2.0\nAI-powered code merging system.",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(RequestLoggingMiddleware)

    # Request timing header
    @app.middleware("http")
    async def add_process_time(request: Request, call_next):
        start    = time.perf_counter()
        response = await call_next(request)
        elapsed  = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed:.2f}"
        return response

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error": str(exc)},
        )

    # Routes
    app.include_router(fusion.router,   prefix="/api/v1", tags=["Fusion"])
    app.include_router(search.router,   prefix="/api/v1", tags=["Search"])
    app.include_router(analyze.router,  prefix="/api/v1", tags=["Analysis"])
    app.include_router(jobs.router,     prefix="/api/v1", tags=["Jobs"])
    app.include_router(metrics.router,  prefix="/api/v1", tags=["Metrics"])
    app.include_router(advanced.router, prefix="/api/v1", tags=["Advanced"])

    # Serve frontend at root
    @app.get("/", include_in_schema=False)
    async def serve_ui():
        if os.path.exists("frontend/index.html"):
            return FileResponse("frontend/index.html")
        return JSONResponse({"message": "Semantic Code Fusion API", "docs": "/docs"})

    # Static files
    if os.path.exists("frontend"):
        app.mount("/static", StaticFiles(directory="frontend"), name="frontend")

    # Health check
    @app.get("/health", tags=["Health"])
    async def health():
        return {
            "status":  "healthy",
            "version": settings.APP_VERSION,
            "service": settings.APP_NAME,
            "vectors": vector_store.stats()["total_vectors"],
        }

    return app


app = create_app()