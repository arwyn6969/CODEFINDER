"""
Database configuration and session management
"""
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import declarative_base, sessionmaker
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

# Create database engine
engine = create_engine(
    settings.database_url,
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

def get_db():
    """Database dependency for FastAPI"""
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

    In debug/dev mode with SQLite, drop and recreate all tables to ensure
    new columns are materialized when models change. This avoids stale
    schemas during rapid iteration and ensures tests see the latest models.
    """
    try:
        if settings.debug and settings.database_url.startswith("sqlite"):
            # For fast local/dev cycles ensure schema matches models
            Base.metadata.drop_all(bind=engine)
            logger.info("Dropped all tables (debug mode)")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise