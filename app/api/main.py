"""Canonical FastAPI application for CODEFINDER."""
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.middleware import setup_middleware
from app.api.routes import (
    analysis,
    auth,
    documents,
    patterns,
    relationships,
    reports,
    research,
    search,
    visualizations,
    websocket,
)
from app.core.config import settings
from app.core.project_identity import API_SERVICE_NAME, API_TITLE
from app.core.database import init_db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _resolve_frontend_build_dir(frontend_build_dir: Optional[Union[str, Path]] = None) -> Path:
    return Path(frontend_build_dir or settings.frontend_build_dir)


def _mount_frontend_static(app: FastAPI, build_dir: Path) -> None:
    static_assets_dir = build_dir / "static"
    if static_assets_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_assets_dir)), name="static")
    elif build_dir.exists():
        logger.info(
            "Frontend build directory exists without static assets at %s; "
            "continuing without mounted frontend assets.",
            static_assets_dir,
        )


def create_app(frontend_build_dir: Optional[Union[str, Path]] = None) -> FastAPI:
    build_dir = _resolve_frontend_build_dir(frontend_build_dir)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            init_db()
        except Exception as exc:
            logger.warning("Database init warning: %s", exc)
        yield

    app = FastAPI(
        title=API_TITLE,
        description="API for document ingestion, OCR-driven analysis, and historical-print research workflows",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )

    # Setup middleware
    setup_middleware(app)

    # Include canonical API routes.
    route_specs = [
        (auth.router, "/auth", "Authentication"),
        (documents.router, "/documents", "Documents"),
        (analysis.router, "/analysis", "Analysis"),
        (patterns.router, "/patterns", "Patterns"),
        (search.router, "/search", "Search"),
        (reports.router, "/reports", "Reports"),
        (visualizations.router, "/visualizations", "Visualizations"),
        (research.router, "/research", "Legacy Exploratory (Internal)"),
        (relationships.router, "/relationships", "Relationships"),
        (websocket.router, "/ws", "WebSocket"),
    ]

    for router, suffix, tag in route_specs:
        app.include_router(router, prefix=f"/api{suffix}", tags=[tag])

    # Backward compatibility for legacy clients that still call /api/v1/*.
    # Keep these out of the OpenAPI schema to avoid route duplication in docs.
    for router, suffix, _ in route_specs:
        app.include_router(router, prefix=f"/api/v1{suffix}", include_in_schema=False)

    _mount_frontend_static(app, build_dir)

    @app.get("/api/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": API_SERVICE_NAME,
            "version": "1.0.0",
        }

    @app.get("/health", include_in_schema=False)
    async def health_check_legacy():
        """Legacy health endpoint maintained for compatibility."""
        return await health_check()

    @app.get("/docs", include_in_schema=False)
    async def docs_redirect():
        """Legacy docs endpoint redirect."""
        return RedirectResponse(url="/api/docs")

    @app.get("/", response_class=HTMLResponse)
    async def serve_frontend():
        """Serve the React frontend application."""
        frontend_path = build_dir / "index.html"
        if frontend_path.is_file():
            return HTMLResponse(content=frontend_path.read_text(), status_code=200)

        return HTMLResponse(
            content=f"""
            <html>
                <head><title>{API_TITLE}</title></head>
                <body>
                    <h1>{API_TITLE}</h1>
                    <p>Frontend not built. Visit <a href="/api/docs">/api/docs</a> for API documentation.</p>
                </body>
            </html>
            """,
            status_code=200,
        )

    @app.get("/{path:path}", response_class=HTMLResponse)
    async def serve_frontend_routes(path: str):
        """Serve the React app for frontend routes."""
        if path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")

        return await serve_frontend()

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
