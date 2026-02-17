"""
FastAPI Dependencies
Authentication, database sessions, and other shared dependencies
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any, Generator
import jwt
from datetime import datetime, timedelta, timezone
import logging

from app.core.database import get_db
from app.core.config import settings

logger = logging.getLogger(__name__)

# Security scheme
# In non-debug environments, enforce credentials strictly
security = HTTPBearer(auto_error=not getattr(settings, 'debug', True))

# JWT settings
SECRET_KEY = (
    getattr(settings, 'secret_key', None)
    or getattr(settings, 'SECRET_KEY', None)
    or "ancient-text-analysis-secret-key"
)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

class User:
    """Simple user model for authentication"""
    def __init__(self, username: str, email: str, is_active: bool = True):
        self.username = username
        self.email = email
        self.is_active = is_active

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify JWT token and return payload"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.PyJWTError as e:
        logger.warning(f"Token verification failed: {str(e)}")
        return None

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[User]:
    """Get current authenticated user"""
    
    # For development, allow anonymous access when debug is True
    if not credentials:
        if getattr(settings, 'debug', True):
            return User(username="anonymous", email="anonymous@example.com")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify token
    payload = verify_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    username = payload.get("sub")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
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

async def rate_limit_dependency(request_id: str = "default") -> bool:
    """Simple rate limiting dependency"""
    current_time = datetime.now(timezone.utc)
    
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
