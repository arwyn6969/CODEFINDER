"""
FastAPI Dependencies
Authentication, database sessions, and other shared dependencies
"""
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import Dict, Any, Generator, Optional
import jwt
from datetime import datetime, timedelta, timezone
import logging

from app.core.database import get_db
from app.core.config import settings

logger = logging.getLogger(__name__)

# Security scheme. Missing credentials are handled explicitly so the API
# returns a consistent 401 instead of a framework-level 403.
security = HTTPBearer(auto_error=False)

# JWT settings
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

class User:
    """Simple user model for authentication"""
    def __init__(self, username: str, email: str, is_active: bool = True):
        self.username = username
        self.email = email
        self.is_active = is_active


def _unauthorized(detail: str = "Not authenticated") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify JWT token and return payload"""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        return payload
    except jwt.PyJWTError as e:
        logger.warning(f"Token verification failed: {str(e)}")
        return None

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> User:
    """Get current authenticated user"""
    if not credentials:
        raise _unauthorized()
    
    # Verify token
    payload = verify_token(credentials.credentials)
    if not payload:
        raise _unauthorized("Invalid authentication credentials")
    
    username = payload.get("sub")
    if not username:
        raise _unauthorized("Invalid token payload")
    
    # In a real application, you would fetch user from database
    # For now, return user from token payload
    return User(
        username=username,
        email=payload.get("email", f"{username}@example.com")
    )

async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user

def get_database() -> Generator[Session, None, None]:
    """Get database session dependency with proper lifecycle management."""
    yield from get_db()

# Rate limiting (simple in-memory implementation)
request_counts = {}

async def rate_limit_dependency(request: Request) -> bool:
    """Simple rate limiting dependency"""
    current_time = datetime.now(timezone.utc)
    client_host = request.client.host if request.client else "unknown"
    request_id = f"{client_host}:{request.url.path}"
    
    # Clean old entries (older than 1 minute)
    cutoff_time = current_time - timedelta(minutes=1)
    request_counts[request_id] = [
        timestamp for timestamp in request_counts.get(request_id, [])
        if timestamp > cutoff_time
    ]
    
    # Check rate limit (max 100 requests per minute)
    if len(request_counts.get(request_id, [])) >= 100:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    
    # Add current request
    if request_id not in request_counts:
        request_counts[request_id] = []
    request_counts[request_id].append(current_time)
    
    return True
