"""
API routes package - Main router configuration
"""
from fastapi import APIRouter

# Create main router
router = APIRouter()

# Import and include all route modules
try:
    from .analysis import router as analysis_router
    router.include_router(analysis_router, prefix="/analysis", tags=["analysis"])
except ImportError:
    pass

try:
    from .auth import router as auth_router
    router.include_router(auth_router, prefix="/auth", tags=["auth"])
except ImportError:
    pass

try:
    from .documents import router as documents_router
    router.include_router(documents_router, prefix="/documents", tags=["documents"])
except ImportError:
    pass

try:
    from .visualizations import router as visualizations_router
    router.include_router(visualizations_router, prefix="/visualizations", tags=["visualizations"])
except ImportError:
    pass

try:
    from .websocket import router as websocket_router
    router.include_router(websocket_router, prefix="/ws", tags=["websocket"])
except ImportError:
    pass

# Include other route modules if they exist
try:
    from .patterns import router as patterns_router
    router.include_router(patterns_router, prefix="/patterns", tags=["patterns"])
except ImportError:
    pass

try:
    from .reports import router as reports_router
    router.include_router(reports_router, prefix="/reports", tags=["reports"])
except ImportError:
    pass

try:
    from .search import router as search_router
    router.include_router(search_router, prefix="/search", tags=["search"])
except ImportError:
    pass

try:
    from .relationships import router as relationships_router
    router.include_router(relationships_router, prefix="/relationships", tags=["relationships"])
except ImportError:
    pass

# Research routes
from .research import router as research_router
router.include_router(research_router, prefix="/research", tags=["research"])