"""
Database configuration and session management
"""
from pathlib import Path

from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import declarative_base, sessionmaker
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


def _ensure_database_directory() -> None:
    """Create the local SQLite parent directory when needed."""
    if not settings.database_url.startswith("sqlite:///"):
        return

    db_path = settings.database_url.removeprefix("sqlite:///")
    if not db_path or db_path == ":memory:":
        return

    Path(db_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def _connect_args() -> dict:
    if settings.database_url.startswith("sqlite"):
        return {"check_same_thread": False}
    return {}


_ensure_database_directory()

# Create database engine
engine = create_engine(
    settings.database_url,
    connect_args=_connect_args(),
    pool_pre_ping=True,
    pool_recycle=300,
    echo=False  # Set to True for SQL debugging
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

# Metadata for database operations
metadata = MetaData()

_db_initialized = False


def ensure_db_initialized() -> None:
    """Create the schema on first real DB use.

    Test clients in this repo do not consistently enter FastAPI lifespan
    events, so runtime startup alone is not enough to guarantee tables exist.
    This remains non-destructive because `init_db()` only calls `create_all()`.
    """
    global _db_initialized
    if _db_initialized:
        return
    init_db()

def get_db():
    """Database dependency for FastAPI"""
    ensure_db_initialized()
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def init_db():
    """Initialize database tables.

    This is intentionally non-destructive. Schema resets should happen via
    explicit maintenance workflows, not on application import/startup.
    """
    global _db_initialized
    try:
        Base.metadata.create_all(bind=engine)
        _db_initialized = True
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise
