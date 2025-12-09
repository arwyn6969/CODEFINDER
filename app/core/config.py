"""
Configuration settings for Ancient Text Analyzer

NOTE: All sensitive settings should be provided via environment variables.
Never commit secrets to the repository. See .env.example for required variables.
"""
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import Optional
import os
try:
    import secrets
except ImportError:
    # Python < 3.6 fallback (unlikely but safe)
    import random
    import string
    class secrets:
        @staticmethod
        def token_urlsafe(nbytes=32):
            return ''.join(random.choices(string.ascii_letters + string.digits + '-_', k=nbytes))

class Settings(BaseSettings):
    # Database settings
    # NOTE: In production, use PostgreSQL. SQLite is only for development.
    database_url: str = Field(
        default="sqlite:///./test_ancient_text_analyzer.db",
        description="Database connection URL. Use PostgreSQL in production."
    )
    
    # Redis settings (optional - for caching and rate limiting)
    redis_url: Optional[str] = Field(
        default=None,
        description="Redis connection URL. Optional, used for caching and distributed rate limiting."
    )
    
    # OCR settings
    tesseract_cmd: Optional[str] = Field(
        default=None,
        description="Path to tesseract executable. Auto-detected if not provided."
    )
    ocr_confidence_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum OCR confidence threshold (0.0-1.0)"
    )
    
    # File processing settings
    max_file_size: int = Field(
        default=2 * 1024 * 1024 * 1024,  # 2GB
        description="Maximum file upload size in bytes"
    )
    upload_dir: str = Field(
        default="uploads",
        description="Directory for uploaded files"
    )
    temp_dir: str = Field(
        default="temp",
        description="Directory for temporary files"
    )
    
    # Processing settings
    chunk_size: int = Field(
        default=1000,
        description="Chunk size for large file processing"
    )
    max_workers: int = Field(
        default=4,
        ge=1,
        description="Maximum number of worker threads"
    )
    
    # Security settings
    # NOTE: SECRET_KEY is REQUIRED in production. Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"
    secret_key: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32) if os.getenv("PRODUCTION") else "dev-secret-key-change-in-production",
        description="Secret key for JWT tokens and encryption. MUST be set via environment variable in production."
    )
    
    # NOTE: DEBUG should be False in production
    debug: bool = Field(
        default=os.getenv("DEBUG", "False").lower() == "true",
        description="Debug mode. Set to False in production."
    )
    
    # CORS settings
    # NOTE: In production, specify exact origins. Comma-separated list.
    allowed_origins: Optional[str] = Field(
        default=None,
        description="Comma-separated list of allowed CORS origins. Defaults to localhost in dev mode."
    )
    
    # Logging settings
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL"
    )
    log_file: str = Field(
        default="logs/app.log",
        description="Path to log file"
    )
    
    @field_validator('secret_key')
    @classmethod
    def validate_secret_key(cls, v: str, info) -> str:
        """Validate that secret key is not the default production value"""
        if v == "your-secret-key-change-in-production":
            import warnings
            warnings.warn(
                "Using default secret key! This is insecure. "
                "Set SECRET_KEY environment variable in production.",
                UserWarning
            )
        if len(v) < 32 and not os.getenv("DEBUG"):
            raise ValueError("SECRET_KEY must be at least 32 characters in production")
        return v
    
    @field_validator('database_url')
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """Warn if using SQLite in production"""
        if v.startswith("sqlite") and os.getenv("PRODUCTION"):
            import warnings
            warnings.warn(
                "SQLite is not recommended for production. Use PostgreSQL instead.",
                UserWarning
            )
        return v
    
    model_config = {
        'env_file': '.env',
        'env_file_encoding': 'utf-8',
        'case_sensitive': False
    }

# Global settings instance
settings = Settings()

# Set tesseract command path if specified
if settings.tesseract_cmd:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

# Normalize allowed origins into list form for CORS
def get_allowed_origins() -> list[str]:
    """
    Get list of allowed CORS origins.
    
    In production, explicitly set ALLOWED_ORIGINS environment variable.
    In development, defaults to localhost origins.
    """
    if settings.allowed_origins:
        origins = [o.strip() for o in settings.allowed_origins.split(',') if o.strip()]
        # In production, don't allow wildcards or overly permissive origins
        if not settings.debug:
            for origin in origins:
                if origin == "*" or origin == "null":
                    import warnings
                    warnings.warn(
                        f"Insecure CORS origin '{origin}' detected. Use specific origins in production.",
                        UserWarning
                    )
        return origins
    
    # Sensible defaults for development only
    if settings.debug:
        return [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ]
    
    # In production, require explicit configuration
    import warnings
    warnings.warn(
        "ALLOWED_ORIGINS not set in production. Defaulting to empty list. "
        "Set ALLOWED_ORIGINS environment variable.",
        UserWarning
    )
    return []