"""
Custom Exception Hierarchy for CODEFINDER

This module defines a custom exception hierarchy for consistent error handling
across the application. All exceptions inherit from CodeFinderException for
easy catching and standardized error responses.
"""
from typing import Optional, Dict, Any


class CodeFinderException(Exception):
    """
    Base exception for all CODEFINDER-specific exceptions.
    
    All custom exceptions should inherit from this class to enable
    consistent error handling and response formatting.
    """
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(CodeFinderException):
    """Raised when input validation fails"""
    def __init__(self, message: str, field: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=400,
            error_code="VALIDATION_ERROR",
            details={"field": field, **(details or {})}
        )
        self.field = field


class AuthenticationError(CodeFinderException):
    """Raised when authentication fails"""
    def __init__(self, message: str = "Authentication failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=401,
            error_code="AUTHENTICATION_ERROR",
            details=details or {}
        )


class AuthorizationError(CodeFinderException):
    """Raised when user lacks permission for an operation"""
    def __init__(self, message: str = "Insufficient permissions", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=403,
            error_code="AUTHORIZATION_ERROR",
            details=details or {}
        )


class NotFoundError(CodeFinderException):
    """Raised when a requested resource is not found"""
    def __init__(self, resource_type: str, resource_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        message = f"{resource_type} not found"
        if resource_id:
            message += f": {resource_id}"
        super().__init__(
            message=message,
            status_code=404,
            error_code="NOT_FOUND",
            details={"resource_type": resource_type, "resource_id": resource_id, **(details or {})}
        )
        self.resource_type = resource_type
        self.resource_id = resource_id


class ConflictError(CodeFinderException):
    """Raised when a resource conflict occurs (e.g., duplicate entry)"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=409,
            error_code="CONFLICT",
            details=details or {}
        )


class ProcessingError(CodeFinderException):
    """Raised when document processing fails"""
    def __init__(
        self,
        message: str,
        stage: Optional[str] = None,
        document_id: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            status_code=500,
            error_code="PROCESSING_ERROR",
            details={
                "stage": stage,
                "document_id": document_id,
                **(details or {})
            }
        )
        self.stage = stage
        self.document_id = document_id


class OCRError(ProcessingError):
    """Raised when OCR processing fails"""
    def __init__(
        self,
        message: str = "OCR processing failed",
        page_number: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            stage="ocr",
            details={"page_number": page_number, **(details or {})}
        )
        self.error_code = "OCR_ERROR"
        self.page_number = page_number


class DatabaseError(CodeFinderException):
    """Raised when database operations fail"""
    def __init__(self, message: str = "Database operation failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=500,
            error_code="DATABASE_ERROR",
            details=details or {}
        )


class FileError(CodeFinderException):
    """Raised when file operations fail"""
    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            status_code=400,
            error_code="FILE_ERROR",
            details={
                "file_path": file_path,
                "operation": operation,
                **(details or {})
            }
        )
        self.file_path = file_path
        self.operation = operation


class ConfigurationError(CodeFinderException):
    """Raised when configuration is invalid or missing"""
    def __init__(self, message: str, setting: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=500,
            error_code="CONFIGURATION_ERROR",
            details={"setting": setting, **(details or {})}
        )
        self.setting = setting
