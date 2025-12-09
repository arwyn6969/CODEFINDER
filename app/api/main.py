"""
FastAPI Main Application
Comprehensive API for Ancient Text Analysis System
"""
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import asyncio
import json
import logging
from pathlib import Path

from app.core.database import get_db, init_db
from app.core.config import settings
from app.core.exceptions import CodeFinderException
from app.api.routes import (
    documents, analysis, patterns, search, reports, 
    visualizations, auth, websocket
)
from app.api.middleware import setup_middleware
from app.api.dependencies import get_current_user
from app.models.database_models import Document

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

# Global exception handler for custom exceptions
@app.exception_handler(CodeFinderException)
async def codefinder_exception_handler(request: Request, exc: CodeFinderException):
    """
    Global exception handler for CodeFinderException and subclasses.
    
    Returns standardized error responses with proper status codes.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "details": exc.details
            }
        }
    )

# Ensure database tables exist for local/test environments
try:
    init_db()
except Exception as e:
    logger.warning(f"Database init warning: {e}")

# Include API routes
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["Analysis"])
app.include_router(patterns.router, prefix="/api/patterns", tags=["Patterns"])
app.include_router(search.router, prefix="/api/search", tags=["Search"])
app.include_router(reports.router, prefix="/api/reports", tags=["Reports"])
app.include_router(visualizations.router, prefix="/api/visualizations", tags=["Visualizations"])
app.include_router(websocket.router, prefix="/api/ws", tags=["WebSocket"])

# Mount static files for frontend
static_dir = Path("frontend/build")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir / "static")), name="static")

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """
    Enhanced health check endpoint with component status.
    
    Returns health status of all system components including:
    - Database connectivity
    - Redis connectivity (if configured)
    - Tesseract OCR availability
    - Storage accessibility
    """
    from datetime import datetime, timezone
    from app.core.config import settings
    
    checks = {}
    overall_status = "healthy"
    
    # Check database
    try:
        db = next(get_db())
        db.execute("SELECT 1")
        db.close()
        checks["database"] = {"status": "healthy", "message": "Database connection successful"}
    except Exception as e:
        checks["database"] = {"status": "unhealthy", "message": str(e)}
        overall_status = "degraded"
    
    # Check Redis (optional)
    if settings.redis_url:
        try:
            import redis
            redis_client = redis.from_url(settings.redis_url)
            redis_client.ping()
            checks["redis"] = {"status": "healthy", "message": "Redis connection successful"}
        except Exception as e:
            checks["redis"] = {"status": "unhealthy", "message": str(e)}
            # Redis is optional, so don't fail overall health if it's down
    else:
        checks["redis"] = {"status": "not_configured", "message": "Redis not configured"}
    
    # Check Tesseract
    try:
        import pytesseract
        # Try to get version
        version = pytesseract.get_tesseract_version()
        checks["tesseract"] = {"status": "healthy", "message": f"Tesseract version {version} available"}
    except Exception as e:
        checks["tesseract"] = {"status": "unhealthy", "message": str(e)}
        overall_status = "degraded"
    
    # Check storage
    try:
        upload_dir = Path(settings.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        # Test write access
        test_file = upload_dir / ".health_check"
        test_file.write_text("test")
        test_file.unlink()
        checks["storage"] = {"status": "healthy", "message": f"Storage accessible at {upload_dir}"}
    except Exception as e:
        checks["storage"] = {"status": "unhealthy", "message": str(e)}
        overall_status = "degraded"
    
    return {
        "status": overall_status,
        "service": "Ancient Text Analysis API",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": checks
    }

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