"""Canonical FastAPI application for CODEFINDER."""
import logging
from pathlib import Path

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
from app.core.database import init_db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Ancient Text Analysis API",
    description="Comprehensive API for analyzing ancient texts, detecting patterns, and generating reports",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Setup middleware
setup_middleware(app)

# Ensure database tables exist for local/test environments
try:
    init_db()
except Exception as e:
    logger.warning(f"Database init warning: {e}")

# Include canonical API routes.
route_specs = [
    (auth.router, "/auth", "Authentication"),
    (documents.router, "/documents", "Documents"),
    (analysis.router, "/analysis", "Analysis"),
    (patterns.router, "/patterns", "Patterns"),
    (search.router, "/search", "Search"),
    (reports.router, "/reports", "Reports"),
    (visualizations.router, "/visualizations", "Visualizations"),
    (research.router, "/research", "Research"),
    (relationships.router, "/relationships", "Relationships"),
    (websocket.router, "/ws", "WebSocket"),
]

for router, suffix, tag in route_specs:
    app.include_router(router, prefix=f"/api{suffix}", tags=[tag])

# Backward compatibility for legacy clients that still call /api/v1/*.
# Keep these out of the OpenAPI schema to avoid route duplication in docs.
for router, suffix, _ in route_specs:
    app.include_router(router, prefix=f"/api/v1{suffix}", include_in_schema=False)

# Mount static files for frontend
static_dir = Path("frontend/build")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir / "static")), name="static")

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Ancient Text Analysis API",
        "version": "1.0.0"
    }

@app.get("/health", include_in_schema=False)
async def health_check_legacy():
    """Legacy health endpoint maintained for compatibility."""
    return await health_check()

@app.get("/docs", include_in_schema=False)
async def docs_redirect():
    """Legacy docs endpoint redirect."""
    return RedirectResponse(url="/api/docs")

# Root endpoint - serve React app
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the React frontend application"""
    frontend_path = static_dir / "index.html"
    if frontend_path.exists():
        return HTMLResponse(content=frontend_path.read_text(), status_code=200)
    else:
        return HTMLResponse(
            content="""
            <html>
                <head><title>Ancient Text Analysis</title></head>
                <body>
                    <h1>Ancient Text Analysis API</h1>
                    <p>Frontend not built. Visit <a href="/api/docs">/api/docs</a> for API documentation.</p>
                </body>
            </html>
            """,
            status_code=200
        )

# Catch-all route for React Router
@app.get("/{path:path}", response_class=HTMLResponse)
async def serve_frontend_routes(path: str):
    """Serve React app for all frontend routes"""
    # Don't serve frontend for API routes
    if path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API endpoint not found")
    
    return await serve_frontend()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
