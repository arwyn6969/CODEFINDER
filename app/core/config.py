"""Application settings for CODEFINDER."""

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Database settings
    database_url: str = "sqlite:///./data/codefinder_app.db"
    
    # Redis settings
    redis_url: str = "redis://localhost:6379"
    
    # OCR settings
    tesseract_cmd: Optional[str] = None
    ocr_confidence_threshold: float = 0.85
    
    # File processing settings
    max_file_size: int = 2 * 1024 * 1024 * 1024  # 2GB
    upload_dir: str = "uploads"
    temp_dir: str = "temp"
    
    # Processing settings
    chunk_size: int = 1000  # For large file processing
    max_workers: int = 4
    
    # Security settings
    secret_key: str = "codefinder-change-me"
    debug: bool = False
    enable_demo_auth: bool = False
    allowed_origins: Optional[str] = None  # Comma-separated list
    frontend_build_dir: str = "frontend/build"
    
    # Logging settings
    log_level: str = "INFO"
    log_file: str = "logs/app.log"
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

# Global settings instance
settings = Settings()

# Set tesseract command path if specified
if settings.tesseract_cmd:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

# Normalize allowed origins into list form for CORS
def get_allowed_origins() -> list[str]:
    if settings.allowed_origins:
        return [o.strip() for o in settings.allowed_origins.split(',') if o.strip()]
    # sensible defaults for dev
    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]
