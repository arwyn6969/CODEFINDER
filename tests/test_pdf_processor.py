"""
Tests for PDF processor
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
from PIL import Image

from app.services.pdf_processor import PDFProcessor, PageImage, PageMetadata

class TestPDFProcessor:
    
    def setup_method(self):
        """Set up test fixtures"""
        self.processor = PDFProcessor(chunk_size=2)
    
    def test_init(self):
        """Test PDFProcessor initialization"""
        assert self.processor.chunk_size == 2
        assert self.processor.temp_dir.exists()
    
    @patch('app.services.pdf_processor.fitz.open')
    @patch('app.services.pdf_processor.convert_from_path')
    def test_extract_pages_success(self, mock_convert, mock_fitz_open):
        """Test successful page extraction"""
        # Mock PDF document
        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=3)
        mock_fitz_open.return_value = mock_doc
        
        # Mock images
        mock_image1 = Mock(spec=Image.Image)
        mock_image1.width = 800
        mock_image1.height = 1200
        mock_image1.save = Mock()
        
        mock_image2 = Mock(spec=Image.Image)
        mock_image2.width = 800
        mock_image2.height = 1200
        mock_image2.save = Mock()
        
        mock_convert.return_value = [mock_image1, mock_image2]
        
        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Test extraction
            pages = list(self.processor.extract_pages(tmp_path, 0, 2))
            
            assert len(pages) == 2
            assert all(isinstance(page, PageImage) for page in pages)
            assert pages[0].page_number == 0
            assert pages[1].page_number == 1
            assert pages[0].width == 800
            assert pages[0].height == 1200
            
        finally:
            Path(tmp_path).unlink()
    
    def test_extract_pages_file_not_found(self):
        """Test extraction with non-existent file"""
        with pytest.raises(FileNotFoundError):
            list(self.processor.extract_pages("nonexistent.pdf"))
    
    @patch('app.services.pdf_processor.fitz.open')
    def test_get_page_metadata(self, mock_fitz_open):
        """Test page metadata extraction"""
        # Mock PDF document and page
        mock_page = Mock()
        mock_page.rect = Mock()
        mock_page.rect.x0, mock_page.rect.y0 = 0, 0
        mock_page.rect.x1, mock_page.rect.y1 = 612, 792
        mock_page.rect.width, mock_page.rect.height = 612, 792
        mock_page.rotation = 0
        mock_page.get_text.return_value = "Sample text content"
        mock_page.get_images.return_value = []
        
        mock_doc = Mock()
        mock_doc.__getitem__ = Mock(return_value=mock_page)
        mock_doc.__len__ = Mock(return_value=5)
        mock_doc.metadata = {'creationDate': '2023-01-01'}
        mock_fitz_open.return_value = mock_doc
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            metadata = self.processor.get_page_metadata(tmp_path, 0)
            
            assert isinstance(metadata, PageMetadata)
            assert metadata.page_number == 0
            assert metadata.width == 612
            assert metadata.height == 792
            assert metadata.rotation == 0
            assert metadata.text_length > 0
            assert metadata.has_images == False
            
        finally:
            Path(tmp_path).unlink()
    
    @patch('app.services.pdf_processor.fitz.open')
    def test_get_document_info(self, mock_fitz_open):
        """Test document info extraction"""
        # Mock PDF document
        mock_page = Mock()
        mock_page.get_text.return_value = "Sample text"
        mock_page.get_images.return_value = []
        
        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=10)
        mock_doc.__getitem__ = Mock(return_value=mock_page)
        mock_doc.metadata = {'title': 'Test Document'}
        mock_doc.is_pdf = True
        mock_doc.needs_pass = False
        mock_fitz_open.return_value = mock_doc
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            info = self.processor.get_document_info(tmp_path)
            
            assert info['page_count'] == 10
            assert info['is_pdf'] == True
            assert info['is_encrypted'] == False
            assert 'file_size' in info
            assert 'estimated_text_length' in info
            
        finally:
            Path(tmp_path).unlink()
    
    def test_cleanup_temp_files(self):
        """Test temporary file cleanup"""
        # Create some temporary files
        temp_files = []
        for i in range(3):
            temp_file = self.processor.temp_dir / f"page_{i:04d}.png"
            temp_file.touch()
            temp_files.append(temp_file)
        
        # Verify files exist
        assert all(f.exists() for f in temp_files)
        
        # Clean up specific files
        self.processor.cleanup_temp_files([0, 1])
        
        # Verify specific files are gone
        assert not temp_files[0].exists()
        assert not temp_files[1].exists()
        assert temp_files[2].exists()
        
        # Clean up all files
        self.processor.cleanup_temp_files()
        
        # Verify all files are gone
        assert not temp_files[2].exists()
    
    def teardown_method(self):
        """Clean up after tests"""
        self.processor.cleanup_temp_files()