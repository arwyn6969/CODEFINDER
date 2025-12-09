"""
FastAPI Middleware Configuration
"""
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import time
import logging
from app.core.config import get_allowed_origins

logger = logging.getLogger(__name__)

def setup_middleware(app: FastAPI):
    """
    Setup all middleware for the FastAPI application.
    
    NOTE: CORS and TrustedHost settings are security-critical.
    In production, ensure ALLOWED_ORIGINS is properly configured.
    """
    from app.core.config import settings
    
    # Get allowed origins (validated in config)
    allowed_origins = get_allowed_origins()
    
    # CORS middleware - allow frontend to communicate with API
    # NOTE: In production, restrict allow_headers to specific headers needed
    allowed_headers = ["Content-Type", "Authorization", "X-Requested-With"]
    if settings.debug:
        # In development, allow all headers for flexibility
        allowed_headers = ["*"]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=allowed_headers,
        expose_headers=["X-Process-Time", "X-Request-ID"],  # Expose custom headers
    )
    
    # Trusted host middleware for security
    # NOTE: In production, add your actual domain names
    trusted_hosts = ["localhost", "127.0.0.1", "*.localhost", "testserver"]
    if not settings.debug:
        # In production, require explicit configuration
        import os
        prod_hosts = os.getenv("TRUSTED_HOSTS", "")
        if prod_hosts:
            trusted_hosts.extend([h.strip() for h in prod_hosts.split(",") if h.strip()])
        else:
            import warnings
            warnings.warn(
                "TRUSTED_HOSTS not set in production. Using default localhost hosts. "
                "Set TRUSTED_HOSTS environment variable with your domain names.",
                UserWarning
            )
    
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=trusted_hosts
    )
    
    # GZip compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Request ID middleware for tracing
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        """Add unique request ID for tracing"""
        import uuid
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    
    # Custom logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all requests with timing information"""
        start_time = time.time()
        request_id = getattr(request.state, "request_id", "unknown")
        
        # Log request
        logger.info(
            f"[{request_id}] Request: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        # Process request
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(
            f"[{request_id}] Response: {response.status_code} - {process_time:.4f}s"
        )
        
        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
    
    # Error handling middleware
    @app.middleware("http")
    async def catch_exceptions(request: Request, call_next):
        """
        Global exception handler.
        
        NOTE: In production, sanitize error messages to avoid exposing
        internal implementation details.
        """
        from app.core.config import settings
        try:
            return await call_next(request)
        except Exception as e:
            request_id = getattr(request.state, "request_id", "unknown")
            logger.error(
                f"[{request_id}] Unhandled exception: {str(e)}",
                exc_info=True
            )
            
            # In production, don't expose internal error details
            if settings.debug:
                error_message = f"Internal server error: {str(e)}"
            else:
                error_message = "Internal server error. Please contact support."
                # Log full error details server-side
                logger.error(f"[{request_id}] Full error: {repr(e)}", exc_info=True)
            
            return Response(
                content=error_message,
                status_code=500,
                media_type="text/plain"
            )