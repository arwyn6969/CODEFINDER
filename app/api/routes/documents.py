"""
Documents API Routes
Document upload, management, and metadata operations
"""
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, status, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
import logging
from pathlib import Path
import shutil
from datetime import datetime, timezone
import uuid

from app.api.dependencies import get_current_active_user, get_database, User, rate_limit_dependency
from app.models.database_models import Document
from app.services.processing_pipeline import ProcessingPipeline

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models
class DocumentResponse(BaseModel):
    id: int
    filename: str
    file_size: int
    upload_date: datetime
    processing_status: str
    total_pages: Optional[int] = None
    analysis_complete: bool = False
    
    class Config:
        from_attributes = True

class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int
    page: int
    per_page: int

class DocumentUploadResponse(BaseModel):
    document_id: int
    filename: str
    message: str
    processing_started: bool

class ProcessingStatusResponse(BaseModel):
    document_id: int
    status: str
    progress_percentage: float
    current_step: str
    estimated_completion: Optional[datetime] = None
    error_message: Optional[str] = None

# File upload settings
# NOTE: These should match settings in app/core/config.py
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB (can be overridden by config)
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".png", ".jpg", ".jpeg", ".tiff"}

# File type magic numbers for content validation
# Format: {extension: [magic_number_bytes]}
FILE_SIGNATURES = {
    ".pdf": [b"%PDF"],
    ".png": [b"\x89PNG\r\n\x1a\n"],
    ".jpg": [b"\xff\xd8\xff"],
    ".jpeg": [b"\xff\xd8\xff"],
    ".tiff": [b"II*\x00", b"MM\x00*"],
    # Text files don't have magic numbers, validate by content
    ".txt": None,
    ".docx": [b"PK\x03\x04"],  # DOCX is a ZIP file
}

async def validate_file(file: UploadFile) -> tuple[bool, str]:
    """
    Comprehensive file validation including extension, size, and content.
    
    Returns:
        (is_valid, error_message)
    """
    # Check filename is provided
    if not file.filename:
        return False, "Filename is required"
    
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return False, f"File extension '{file_ext}' not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    
    # Check filename for path traversal attempts
    if ".." in file.filename or "/" in file.filename or "\\" in file.filename:
        return False, "Invalid filename: path traversal detected"
    
    # Read first few bytes for content validation
    try:
        # Save current position
        current_pos = await file.seek(0, 1) if hasattr(file, 'seek') else 0
        
        # Read first 16 bytes for magic number checking
        content = await file.read(16)
        
        # Reset file position
        if hasattr(file, 'seek'):
            await file.seek(0)
        
        # Check file size (read content length)
        if len(content) == 0:
            return False, "File is empty"
        
        # Validate content signature if available
        if file_ext in FILE_SIGNATURES:
            signatures = FILE_SIGNATURES[file_ext]
            if signatures:  # None means no signature check (e.g., .txt)
                # Check if content matches any expected signature
                matches = any(content.startswith(sig) for sig in signatures)
                if not matches:
                    return False, f"File content does not match expected format for {file_ext}"
        
        # For text files, validate it's actually text
        if file_ext == ".txt":
            try:
                content.decode('utf-8')
            except UnicodeDecodeError:
                return False, "File does not appear to be valid UTF-8 text"
        
    except Exception as e:
        logger.error(f"Error validating file content: {str(e)}")
        return False, f"Error reading file: {str(e)}"
    
    return True, ""

def validate_file_size(file_size: int, max_size: Optional[int] = None) -> tuple[bool, str]:
    """
    Validate file size.
    
    Args:
        file_size: Size of file in bytes
        max_size: Maximum allowed size (uses MAX_FILE_SIZE if not provided)
    
    Returns:
        (is_valid, error_message)
    """
    max_allowed = max_size or MAX_FILE_SIZE
    if file_size > max_allowed:
        size_mb = file_size / (1024 * 1024)
        max_mb = max_allowed / (1024 * 1024)
        return False, f"File size ({size_mb:.2f}MB) exceeds maximum allowed size ({max_mb:.2f}MB)"
    
    if file_size == 0:
        return False, "File is empty"
    
    return True, ""

async def process_document_background(document_id: int, file_path: str):
    """Background task to process uploaded document"""
    try:
        # Initialize processing pipeline
        db = next(get_database())
        pipeline = ProcessingPipeline(db)
        
        # Update document status
        document = db.query(Document).filter(Document.id == document_id).first()
        if document:
            document.processing_status = "processing"
            db.commit()
        
        # Process the document
        result = await pipeline.process_document_async(file_path, document_id)
        
        # Update final status
        if document:
            document.processing_status = "completed" if result.success else "failed"
            document.total_pages = result.total_pages
            document.analysis_complete = result.success
            db.commit()
        
        logger.info(f"Document {document_id} processing completed: {result.success}")
        
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {str(e)}")
        
        # Update error status
        try:
            db = next(get_database())
            document = db.query(Document).filter(Document.id == document_id).first()
            if document:
                document.processing_status = "failed"
                db.commit()
        except Exception as db_error:
            logger.error(f"Error updating document status: {str(db_error)}")

@router.post("/upload", response_model=DocumentUploadResponse, dependencies=[Depends(rate_limit_dependency)])
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_database)
):
    """
    Upload a document for analysis.
    
    Supports PDF, text, and image files.
    Files are validated for type, size, and content before processing.
    """
    from app.core.config import settings
    
    try:
        # Validate file extension and content
        is_valid, error_message = await validate_file(file)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_message or f"Invalid file. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Generate unique filename to prevent conflicts and path traversal
        file_ext = Path(file.filename).suffix.lower()
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = UPLOAD_DIR / unique_filename
        
        # Ensure upload directory exists and is secure
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file with size validation
        content = await file.read()
        
        # Validate file size using config setting
        max_size = getattr(settings, 'max_file_size', MAX_FILE_SIZE)
        is_size_valid, size_error = validate_file_size(len(content), max_size)
        if not is_size_valid:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=size_error
            )
        
        # Write file to disk
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        # Verify file was written correctly
        if not file_path.exists() or file_path.stat().st_size != len(content):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error saving file. Please try again."
            )
        
        # Create document record
        document = Document(
            filename=file.filename,
            file_path=str(file_path),
            file_size=len(content),
            upload_date=datetime.now(timezone.utc),
            processing_status="uploaded",
            uploaded_by=current_user.username
        )
        
        db.add(document)
        db.commit()
        db.refresh(document)
        
        # Start background processing
        background_tasks.add_task(
            process_document_background,
            document.id,
            str(file_path)
        )
        
        logger.info(f"Document uploaded: {file.filename} (ID: {document.id}) by {current_user.username}")
        
        return DocumentUploadResponse(
            document_id=document.id,
            filename=file.filename,
            message="Document uploaded successfully. Processing started.",
            processing_started=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload error: {str(e)}")
        
        # Clean up file if it was created
        if 'file_path' in locals() and file_path.exists():
            file_path.unlink()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document upload failed due to server error"
        )

@router.get("/", response_model=DocumentListResponse, dependencies=[Depends(rate_limit_dependency)])
async def list_documents(
    page: int = 1,
    per_page: int = 20,
    status_filter: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_database)
):
    """
    List all documents with pagination and filtering
    """
    try:
        # Build query
        query = db.query(Document)
        
        # Apply status filter
        if status_filter:
            query = query.filter(Document.processing_status == status_filter)
        
        # Get total count
        total = query.count()
        
        # Apply pagination
        offset = (page - 1) * per_page
        documents = query.offset(offset).limit(per_page).all()
        
        return DocumentListResponse(
            documents=[DocumentResponse.from_orm(doc) for doc in documents],
            total=total,
            page=page,
            per_page=per_page
        )
        
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve documents"
        )

@router.get("/{document_id}", response_model=DocumentResponse, dependencies=[Depends(rate_limit_dependency)])
async def get_document(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_database)
):
    """
    Get specific document details
    """
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        return DocumentResponse.from_orm(document)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document"
        )

@router.get("/{document_id}/status", response_model=ProcessingStatusResponse, dependencies=[Depends(rate_limit_dependency)])
async def get_processing_status(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_database)
):
    """
    Get document processing status and progress
    """
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Calculate progress percentage based on status
        progress_map = {
            "uploaded": 0.0,
            "processing": 50.0,
            "completed": 100.0,
            "failed": 0.0
        }
        
        # Get current step description
        step_map = {
            "uploaded": "Queued for processing",
            "processing": "Analyzing document content",
            "completed": "Analysis complete",
            "failed": "Processing failed"
        }
        
        return ProcessingStatusResponse(
            document_id=document_id,
            status=document.processing_status,
            progress_percentage=progress_map.get(document.processing_status, 0.0),
            current_step=step_map.get(document.processing_status, "Unknown"),
            estimated_completion=None,  # Could implement ETA calculation
            error_message=None  # Could store error details in database
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting processing status for document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve processing status"
        )

@router.delete("/{document_id}", dependencies=[Depends(rate_limit_dependency)])
async def delete_document(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_database)
):
    """
    Delete a document and its associated data
    """
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Delete associated file
        if document.file_path and Path(document.file_path).exists():
            Path(document.file_path).unlink()
        
        # Delete database record (cascade will handle related records)
        db.delete(document)
        db.commit()
        
        logger.info(f"Document {document_id} deleted by {current_user.username}")
        
        return {"message": "Document deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document"
        )

@router.post("/{document_id}/reprocess", dependencies=[Depends(rate_limit_dependency)])
async def reprocess_document(
    document_id: int,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_database)
):
    """
    Reprocess a document (useful if processing failed or new analysis methods available)
    """
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        if not Path(document.file_path).exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Original file no longer exists"
            )
        
        # Reset processing status
        document.processing_status = "uploaded"
        document.analysis_complete = False
        db.commit()
        
        # Start background processing
        background_tasks.add_task(
            process_document_background,
            document.id,
            document.file_path
        )
        
        logger.info(f"Document {document_id} reprocessing started by {current_user.username}")
        
        return {"message": "Document reprocessing started"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reprocessing document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start reprocessing"
        )