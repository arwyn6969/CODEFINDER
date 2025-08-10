"""
PDF Processing Service for Ancient Text Analyzer
Handles large PDF files with chunked processing and metadata preservation
"""
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image
import logging
from typing import Iterator, Optional, Dict, Any
from pathlib import Path
import tempfile
import os
from dataclasses import dataclass
import time

from app.core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class PageImage:
    """Container for page image data"""
    image: Image.Image
    page_number: int
    width: int
    height: int
    dpi: int
    file_path: Optional[str] = None

@dataclass
class PageMetadata:
    """Container for page metadata"""
    page_number: int
    width: float
    height: float
    rotation: int
    media_box: tuple
    crop_box: tuple
    text_length: int
    has_images: bool
    creation_date: Optional[str] = None

@dataclass
class ProcessedImage:
    """Container for processed image data"""
    image: Image.Image
    original: PageImage
    preprocessing_steps: list
    quality_score: float

class PDFProcessor:
    """
    Advanced PDF processor for handling large historical documents
    Implements chunked processing and memory management
    """
    
    def __init__(self, chunk_size: int = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.temp_dir = Path(settings.temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
    
    def get_page_count(self, pdf_path: str) -> int:
        """Get the total number of pages in a PDF"""
        try:
            doc = fitz.open(str(pdf_path))
            page_count = len(doc)
            doc.close()
            return page_count
        except Exception as e:
            logger.error(f"Error getting page count: {e}")
            raise
    
    def get_pdf_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Get PDF metadata"""
        try:
            doc = fitz.open(str(pdf_path))
            metadata = doc.metadata
            doc.close()
            return metadata
        except Exception as e:
            logger.error(f"Error getting PDF metadata: {e}")
            raise
    
    def extract_page_range(self, pdf_path: str, output_path: str, start_page: int, end_page: int) -> bool:
        """Extract a range of pages to a new PDF"""
        try:
            doc = fitz.open(str(pdf_path))
            
            # Create new document with selected pages
            new_doc = fitz.open()
            
            # Insert pages (fitz uses 0-based indexing)
            new_doc.insert_pdf(doc, from_page=start_page-1, to_page=end_page-1)
            
            # Save new document
            new_doc.save(output_path)
            
            # Clean up
            new_doc.close()
            doc.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error extracting page range: {e}")
            return False
        
    def extract_pages(self, pdf_path: str, start_page: int = 0, end_page: int = None) -> Iterator[PageImage]:
        """
        Extract pages from PDF with memory-efficient streaming
        
        Args:
            pdf_path: Path to PDF file
            start_page: Starting page number (0-indexed)
            end_page: Ending page number (None for all pages)
            
        Yields:
            PageImage objects for each page
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Starting PDF extraction: {pdf_path}")
        start_time = time.time()
        
        try:
            # Open PDF document
            doc = fitz.open(str(pdf_path))
            total_pages = len(doc)
            
            if end_page is None:
                end_page = total_pages
            else:
                end_page = min(end_page, total_pages)
            
            logger.info(f"Processing pages {start_page} to {end_page-1} of {total_pages}")
            
            # Process pages in chunks to manage memory
            for chunk_start in range(start_page, end_page, self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, end_page)
                
                logger.info(f"Processing chunk: pages {chunk_start} to {chunk_end-1}")
                
                # Convert chunk to images
                try:
                    images = convert_from_path(
                        str(pdf_path),
                        first_page=chunk_start + 1,  # pdf2image uses 1-indexed
                        last_page=chunk_end,
                        dpi=300,  # High DPI for OCR accuracy
                        fmt='PNG',
                        thread_count=2  # Limit threads to manage memory
                    )
                    
                    # Yield each page image
                    for i, image in enumerate(images):
                        page_num = chunk_start + i
                        
                        # Create temporary file for the image
                        temp_file = self.temp_dir / f"page_{page_num:04d}.png"
                        image.save(temp_file, "PNG")
                        
                        page_image = PageImage(
                            image=image,
                            page_number=page_num,
                            width=image.width,
                            height=image.height,
                            dpi=300,
                            file_path=str(temp_file)
                        )
                        
                        yield page_image
                        
                        # Clean up image from memory
                        del image
                        
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_start}-{chunk_end}: {e}")
                    continue
            
            doc.close()
            
            processing_time = time.time() - start_time
            logger.info(f"PDF extraction completed in {processing_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise
    
    def get_page_metadata(self, pdf_path: str, page_num: int) -> PageMetadata:
        """
        Extract metadata for a specific page
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)
            
        Returns:
            PageMetadata object
        """
        try:
            doc = fitz.open(pdf_path)
            
            if page_num >= len(doc):
                raise ValueError(f"Page {page_num} does not exist in document")
            
            page = doc[page_num]
            
            # Extract page properties
            rect = page.rect
            media_box = (rect.x0, rect.y0, rect.x1, rect.y1)
            crop_box = media_box  # Simplified for now
            
            # Get text length
            text = page.get_text()
            text_length = len(text.strip())
            
            # Check for images
            image_list = page.get_images()
            has_images = len(image_list) > 0
            
            # Get document metadata
            metadata = doc.metadata
            creation_date = metadata.get('creationDate')
            
            page_metadata = PageMetadata(
                page_number=page_num,
                width=rect.width,
                height=rect.height,
                rotation=page.rotation,
                media_box=media_box,
                crop_box=crop_box,
                text_length=text_length,
                has_images=has_images,
                creation_date=creation_date
            )
            
            doc.close()
            return page_metadata
            
        except Exception as e:
            logger.error(f"Failed to extract metadata for page {page_num}: {e}")
            raise
    
    def get_document_info(self, pdf_path: str) -> Dict[str, Any]:
        """
        Get comprehensive document information
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with document information
        """
        try:
            doc = fitz.open(pdf_path)
            
            # Basic document info
            info = {
                'page_count': len(doc),
                'metadata': doc.metadata,
                'is_pdf': doc.is_pdf,
                'is_encrypted': doc.needs_pass,
                'file_size': Path(pdf_path).stat().st_size
            }
            
            # Calculate total text length
            total_text_length = 0
            total_images = 0
            
            for page_num in range(min(10, len(doc))):  # Sample first 10 pages
                page = doc[page_num]
                text = page.get_text()
                total_text_length += len(text.strip())
                total_images += len(page.get_images())
            
            info['estimated_text_length'] = total_text_length * (len(doc) / min(10, len(doc)))
            info['estimated_images'] = total_images * (len(doc) / min(10, len(doc)))
            
            doc.close()
            return info
            
        except Exception as e:
            logger.error(f"Failed to get document info: {e}")
            raise
    
    def cleanup_temp_files(self, page_numbers: list = None):
        """
        Clean up temporary files
        
        Args:
            page_numbers: List of page numbers to clean up (None for all)
        """
        try:
            if page_numbers is None:
                # Clean up all temp files
                for temp_file in self.temp_dir.glob("page_*.png"):
                    temp_file.unlink()
            else:
                # Clean up specific pages
                for page_num in page_numbers:
                    temp_file = self.temp_dir / f"page_{page_num:04d}.png"
                    if temp_file.exists():
                        temp_file.unlink()
                        
            logger.info("Temporary files cleaned up")
            
        except Exception as e:
            logger.error(f"Failed to clean up temp files: {e}")
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.cleanup_temp_files()
        except:
            pass