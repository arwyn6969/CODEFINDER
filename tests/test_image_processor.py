"""
Tests for image processor
"""
import pytest
import numpy as np
import cv2
from PIL import Image
from unittest.mock import Mock, patch

from app.services.image_processor import ImageProcessor, PreprocessingStep, QualityMetrics
from app.services.pdf_processor import PageImage, ProcessedImage

class TestImageProcessor:
    
    def setup_method(self):
        """Set up test fixtures"""
        self.processor = ImageProcessor()
        
        # Create a test image
        self.test_image_array = np.ones((100, 100, 3), dtype=np.uint8) * 128
        self.test_pil_image = Image.fromarray(self.test_image_array)
        
        self.test_page_image = PageImage(
            image=self.test_pil_image,
            page_number=0,
            width=100,
            height=100,
            dpi=300
        )
    
    def test_init(self):
        """Test ImageProcessor initialization"""
        assert isinstance(self.processor.preprocessing_steps, list)
    
    def test_pil_to_cv2_conversion(self):
        """Test PIL to OpenCV conversion"""
        cv_image = self.processor._pil_to_cv2(self.test_pil_image)
        assert isinstance(cv_image, np.ndarray)
        assert cv_image.shape == (100, 100, 3)
        assert cv_image.dtype == np.uint8
    
    def test_cv2_to_pil_conversion(self):
        """Test OpenCV to PIL conversion"""
        cv_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        pil_image = self.processor._cv2_to_pil(cv_image)
        assert isinstance(pil_image, Image.Image)
        assert pil_image.size == (100, 100)
    
    def test_cv2_to_pil_grayscale(self):
        """Test OpenCV to PIL conversion for grayscale"""
        cv_image = np.ones((100, 100), dtype=np.uint8) * 128
        pil_image = self.processor._cv2_to_pil(cv_image)
        assert isinstance(pil_image, Image.Image)
        assert pil_image.size == (100, 100)
    
    def test_reduce_noise(self):
        """Test noise reduction"""
        # Create noisy image
        noisy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        denoised, step_info = self.processor._reduce_noise(noisy_image)
        
        assert isinstance(denoised, np.ndarray)
        assert denoised.shape == noisy_image.shape
        assert step_info["step"] == PreprocessingStep.NOISE_REDUCTION.value
        assert step_info["success"] == True
    
    def test_enhance_contrast(self):
        """Test contrast enhancement"""
        # Create low contrast image
        low_contrast = np.ones((100, 100, 3), dtype=np.uint8) * 100
        
        enhanced, step_info = self.processor._enhance_contrast(low_contrast)
        
        assert isinstance(enhanced, np.ndarray)
        assert enhanced.shape == low_contrast.shape
        assert step_info["step"] == PreprocessingStep.CONTRAST_ENHANCEMENT.value
        assert step_info["success"] == True
    
    def test_adjust_brightness_dark_image(self):
        """Test brightness adjustment for dark image"""
        # Create dark image
        dark_image = np.ones((100, 100, 3), dtype=np.uint8) * 50
        
        adjusted, step_info = self.processor._adjust_brightness(dark_image)
        
        assert isinstance(adjusted, np.ndarray)
        assert step_info["step"] == PreprocessingStep.BRIGHTNESS_ADJUSTMENT.value
        assert step_info["success"] == True
        assert step_info["parameters"]["adjustment"] > 0  # Should brighten
    
    def test_adjust_brightness_bright_image(self):
        """Test brightness adjustment for bright image"""
        # Create bright image
        bright_image = np.ones((100, 100, 3), dtype=np.uint8) * 200
        
        adjusted, step_info = self.processor._adjust_brightness(bright_image)
        
        assert isinstance(adjusted, np.ndarray)
        assert step_info["step"] == PreprocessingStep.BRIGHTNESS_ADJUSTMENT.value
        assert step_info["success"] == True
        assert step_info["parameters"]["adjustment"] < 0  # Should darken
    
    def test_sharpen_image(self):
        """Test image sharpening"""
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        sharpened, step_info = self.processor._sharpen_image(test_image)
        
        assert isinstance(sharpened, np.ndarray)
        assert sharpened.shape == test_image.shape
        assert step_info["step"] == PreprocessingStep.SHARPENING.value
        assert step_info["success"] == True
    
    def test_binarize_image(self):
        """Test image binarization"""
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        binary, step_info = self.processor._binarize_image(test_image)
        
        assert isinstance(binary, np.ndarray)
        assert binary.shape == test_image.shape
        assert step_info["step"] == PreprocessingStep.BINARIZATION.value
        assert step_info["success"] == True
    
    def test_morphological_operations(self):
        """Test morphological operations"""
        # Create binary-like image
        binary_image = np.zeros((100, 100, 3), dtype=np.uint8)
        binary_image[40:60, 40:60] = 255  # White square
        
        cleaned, step_info = self.processor._apply_morphological_operations(binary_image)
        
        assert isinstance(cleaned, np.ndarray)
        assert cleaned.shape == binary_image.shape
        assert step_info["step"] == PreprocessingStep.MORPHOLOGICAL_OPERATIONS.value
        assert step_info["success"] == True
    
    def test_calculate_noise_level(self):
        """Test noise level calculation"""
        # Clean image should have low noise
        clean_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        noise_level = self.processor._calculate_noise_level(clean_image)
        assert 0 <= noise_level <= 1
        
        # Noisy image should have higher noise
        noisy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        noisy_level = self.processor._calculate_noise_level(noisy_image)
        assert noisy_level > noise_level
    
    def test_assess_image_quality(self):
        """Test image quality assessment"""
        original = np.ones((100, 100, 3), dtype=np.uint8) * 128
        processed = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        quality_score = self.processor._assess_image_quality(original, processed)
        
        assert 0 <= quality_score <= 1
        assert isinstance(quality_score, float)
    
    def test_preprocess_image_success(self):
        """Test successful image preprocessing"""
        result = self.processor.preprocess_image(self.test_page_image)
        
        assert isinstance(result, ProcessedImage)
        assert isinstance(result.image, Image.Image)
        assert result.original == self.test_page_image
        assert len(result.preprocessing_steps) > 0
        assert 0 <= result.quality_score <= 1
    
    @patch.object(ImageProcessor, '_reduce_noise')
    def test_preprocess_image_with_error(self, mock_reduce_noise):
        """Test preprocessing with error handling"""
        # Mock an error in noise reduction
        mock_reduce_noise.side_effect = Exception("Test error")
        
        result = self.processor.preprocess_image(self.test_page_image)
        
        # Should return original image when preprocessing fails
        assert isinstance(result, ProcessedImage)
        assert result.quality_score == 0.0
        assert any("error" in step.get("step", "") for step in result.preprocessing_steps)
    
    def test_deskew_image_no_lines(self):
        """Test deskewing when no lines are detected"""
        # Create image without clear lines
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        deskewed, step_info = self.processor._deskew_image(test_image)
        
        assert isinstance(deskewed, np.ndarray)
        assert step_info["step"] == PreprocessingStep.DESKEWING.value
        assert step_info["success"] == True
        assert step_info["parameters"]["angle_corrected"] == 0