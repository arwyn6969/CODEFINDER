"""
Authentication API Routes
User login, logout, and session management
"""
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from typing import Optional
from datetime import timedelta
import logging

from app.api.dependencies import (
    create_access_token, get_current_user, get_current_active_user,
    User, ACCESS_TOKEN_EXPIRE_MINUTES
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response
class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    user: dict

class UserResponse(BaseModel):
    username: str
    email: str
    is_active: bool

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str

# Demo users for development (in production, use proper user database)
DEMO_USERS = {
    "admin": {
        "username": "admin",
        "email": "admin@ancienttext.com",
        "password": "admin123",  # In production, use hashed passwords
        "is_active": True
    },
    "researcher": {
        "username": "researcher",
        "email": "researcher@ancienttext.com", 
        "password": "research123",
        "is_active": True
    },
    "demo": {
        "username": "demo",
        "email": "demo@ancienttext.com",
        "password": "demo123",
        "is_active": True
    }
}

@router.post("/login", response_model=LoginResponse)
async def login(login_data: LoginRequest):
    """
    User login endpoint
    Returns JWT access token for authenticated requests
    """
    try:
        # Validate credentials (in production, check against database with hashed passwords)
        user_data = DEMO_USERS.get(login_data.username)
        
        if not user_data or user_data["password"] != login_data.password:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not user_data["is_active"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Inactive user account"
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={
                "sub": user_data["username"],
                "email": user_data["email"]
            },
            expires_delta=access_token_expires
        )
        
        logger.info(f"User {login_data.username} logged in successfully")
        
        return LoginResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # Convert to seconds
            user={
                "username": user_data["username"],
                "email": user_data["email"],
                "is_active": user_data["is_active"]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed due to server error"
        )

@router.post("/register", response_model=UserResponse)
async def register(register_data: RegisterRequest):
    """
    User registration endpoint
    Creates new user account (demo implementation)
    """
    try:
        # Check if user already exists
        if register_data.username in DEMO_USERS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        # In production, hash the password and store in database
        DEMO_USERS[register_data.username] = {
            "username": register_data.username,
            "email": register_data.email,
            "password": register_data.password,  # Hash this in production!
            "is_active": True
        }
        
        logger.info(f"New user registered: {register_data.username}")
        
        return UserResponse(
            username=register_data.username,
            email=register_data.email,
            is_active=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed due to server error"
        )

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """
    Get current user information
    Requires valid authentication token
    """
    return UserResponse(
        username=current_user.username,
        email=current_user.email,
        is_active=current_user.is_active
    )

@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)):
    """
    User logout endpoint
    In a stateless JWT system, logout is handled client-side by discarding the token
    """
    logger.info(f"User {current_user.username} logged out")
    return {"message": "Successfully logged out"}

@router.post("/refresh")
async def refresh_token(current_user: User = Depends(get_current_user)):
    """
    Refresh access token
    Issues a new token with extended expiration
    """
    try:
        # Create new access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={
                "sub": current_user.username,
                "email": current_user.email
            },
            expires_delta=access_token_expires
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }
        
    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )

@router.get("/demo-users")
async def get_demo_users():
    """
    Get list of demo users for development/testing
    Remove this endpoint in production!
    """
    return {
        "demo_users": [
            {
                "username": user_data["username"],
                "email": user_data["email"],
                "password": user_data["password"]  # Only for demo!
            }
            for user_data in DEMO_USERS.values()
        ],
        "note": "These are demo credentials for development only"
    }