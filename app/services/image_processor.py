"""
Image Processing Service for Ancient Text Analyzer
Handles image enhancement and preprocessing for historical documents
"""
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import logging
from typing import Tuple, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import time

from app.services.pdf_processor import PageImage, ProcessedImage

logger = logging.getLogger(__name__)

class PreprocessingStep(Enum):
    """Enumeration of preprocessing steps"""
    NOISE_REDUCTION = "noise_reduction"
    CONTRAST_ENHANCEMENT = "contrast_enhancement"
    BRIGHTNESS_ADJUSTMENT = "brightness_adjustment"
    SHARPENING = "sharpening"
    DESKEWING = "deskewing"
    BINARIZATION = "binarization"
    MORPHOLOGICAL_OPERATIONS = "morphological_operations"

@dataclass
class QualityMetrics:
    """Container for image quality assessment metrics"""
    contrast_score: float
    sharpness_score: float
    noise_level: float
    text_clarity_score: float
    overall_score: float

class ImageProcessor:
    """
    Advanced image processor for historical document enhancement
    Optimized for ornate fonts and aged paper
    """
    
    def __init__(self):
        self.preprocessing_steps = []
        
    def preprocess_image(self, page_image: PageImage) -> ProcessedImage:
        """
        Apply comprehensive preprocessing to enhance OCR accuracy
        
        Args:
            page_image: Input page image
            
        Returns:
            ProcessedImage with enhanced image and metadata
        """
        logger.info(f"Starting image preprocessing for page {page_image.page_number}")
        start_time = time.time()
        
        # Convert PIL image to OpenCV format
        cv_image = self._pil_to_cv2(page_image.image)
        original_image = cv_image.copy()
        
        preprocessing_steps = []
        
        try:
            # Step 1: Noise reduction
            cv_image, step_info = self._reduce_noise(cv_image)
            preprocessing_steps.append(step_info)
            
            # Step 2: Contrast enhancement
            cv_image, step_info = self._enhance_contrast(cv_image)
            preprocessing_steps.append(step_info)
            
            # Step 3: Brightness adjustment
            cv_image, step_info = self._adjust_brightness(cv_image)
            preprocessing_steps.append(step_info)
            
            # Step 4: Sharpening for text clarity
            cv_image, step_info = self._sharpen_image(cv_image)
            preprocessing_steps.append(step_info)
            
            # Step 5: Deskewing (straighten rotated text)
            cv_image, step_info = self._deskew_image(cv_image)
            preprocessing_steps.append(step_info)
            
            # Step 6: Binarization (convert to black and white)
            cv_image, step_info = self._binarize_image(cv_image)
            preprocessing_steps.append(step_info)
            
            # Step 7: Morphological operations (clean up artifacts)
            cv_image, step_info = self._apply_morphological_operations(cv_image)
            preprocessing_steps.append(step_info)
            
            # Convert back to PIL format
            processed_pil = self._cv2_to_pil(cv_image)
            
            # Calculate quality score
            quality_score = self._assess_image_quality(original_image, cv_image)
            
            processing_time = time.time() - start_time
            logger.info(f"Image preprocessing completed in {processing_time:.2f}s, quality score: {quality_score:.2f}")
            
            return ProcessedImage(
                image=processed_pil,
                original=page_image,
                preprocessing_steps=preprocessing_steps,
                quality_score=quality_score
            )
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            # Return original image if preprocessing fails
            return ProcessedImage(
                image=page_image.image,
                original=page_image,
                preprocessing_steps=[{"step": "error", "message": str(e)}],
                quality_score=0.0
            )
    
    def _pil_to_cv2(self, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL image to OpenCV format"""
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def _cv2_to_pil(self, cv_image: np.ndarray) -> Image.Image:
        """Convert OpenCV image to PIL format"""
        if len(cv_image.shape) == 3:
            return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        else:
            return Image.fromarray(cv_image)
    
    def _reduce_noise(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply noise reduction using bilateral filtering
        Preserves edges while reducing noise
        """
        try:
            # Apply bilateral filter to reduce noise while preserving edges
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
            
            # Additional Gaussian blur for very noisy images
            if self._calculate_noise_level(image) > 0.3:
                denoised = cv2.GaussianBlur(denoised, (3, 3), 0)
            
            step_info = {
                "step": PreprocessingStep.NOISE_REDUCTION.value,
                "parameters": {"bilateral_d": 9, "sigma_color": 75, "sigma_space": 75},
                "success": True
            }
            
            return denoised, step_info
            
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return image, {"step": PreprocessingStep.NOISE_REDUCTION.value, "success": False, "error": str(e)}
    
    def _enhance_contrast(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        """
        try:
            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            
            # Merge channels back
            enhanced = cv2.merge([l_channel, a_channel, b_channel])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            step_info = {
                "step": PreprocessingStep.CONTRAST_ENHANCEMENT.value,
                "parameters": {"clip_limit": 2.0, "tile_grid_size": (8, 8)},
                "success": True
            }
            
            return enhanced, step_info
            
        except Exception as e:
            logger.warning(f"Contrast enhancement failed: {e}")
            return image, {"step": PreprocessingStep.CONTRAST_ENHANCEMENT.value, "success": False, "error": str(e)}
    
    def _adjust_brightness(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Adjust brightness based on image histogram
        """
        try:
            # Calculate mean brightness
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            
            # Adjust brightness if too dark or too bright
            adjustment = 0
            if mean_brightness < 100:  # Too dark
                adjustment = 100 - mean_brightness
            elif mean_brightness > 180:  # Too bright
                adjustment = 180 - mean_brightness
            
            if abs(adjustment) > 10:
                adjusted = cv2.convertScaleAbs(image, alpha=1.0, beta=adjustment)
            else:
                adjusted = image
            
            step_info = {
                "step": PreprocessingStep.BRIGHTNESS_ADJUSTMENT.value,
                "parameters": {"mean_brightness": mean_brightness, "adjustment": adjustment},
                "success": True
            }
            
            return adjusted, step_info
            
        except Exception as e:
            logger.warning(f"Brightness adjustment failed: {e}")
            return image, {"step": PreprocessingStep.BRIGHTNESS_ADJUSTMENT.value, "success": False, "error": str(e)}
    
    def _sharpen_image(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply sharpening filter to enhance text clarity
        """
        try:
            # Create sharpening kernel
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
            
            # Apply sharpening
            sharpened = cv2.filter2D(image, -1, kernel)
            
            # Blend with original to avoid over-sharpening
            alpha = 0.7  # Sharpening strength
            result = cv2.addWeighted(image, 1 - alpha, sharpened, alpha, 0)
            
            step_info = {
                "step": PreprocessingStep.SHARPENING.value,
                "parameters": {"alpha": alpha},
                "success": True
            }
            
            return result, step_info
            
        except Exception as e:
            logger.warning(f"Sharpening failed: {e}")
            return image, {"step": PreprocessingStep.SHARPENING.value, "success": False, "error": str(e)}
    
    def _deskew_image(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Detect and correct skew in the image
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                # Calculate average angle
                angles = []
                for rho, theta in lines[:20]:  # Use first 20 lines
                    angle = theta * 180 / np.pi
                    if angle > 90:
                        angle = angle - 180
                    angles.append(angle)
                
                if angles:
                    median_angle = np.median(angles)
                    
                    # Only correct if angle is significant
                    if abs(median_angle) > 0.5:
                        # Get image center
                        (h, w) = image.shape[:2]
                        center = (w // 2, h // 2)
                        
                        # Create rotation matrix
                        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                        
                        # Apply rotation
                        deskewed = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                                flags=cv2.INTER_CUBIC, 
                                                borderMode=cv2.BORDER_REPLICATE)
                        
                        step_info = {
                            "step": PreprocessingStep.DESKEWING.value,
                            "parameters": {"angle_corrected": median_angle},
                            "success": True
                        }
                        
                        return deskewed, step_info
            
            # No significant skew detected
            step_info = {
                "step": PreprocessingStep.DESKEWING.value,
                "parameters": {"angle_corrected": 0},
                "success": True
            }
            
            return image, step_info
            
        except Exception as e:
            logger.warning(f"Deskewing failed: {e}")
            return image, {"step": PreprocessingStep.DESKEWING.value, "success": False, "error": str(e)}
    
    def _binarize_image(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Convert image to binary (black and white) using adaptive thresholding
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Convert back to 3-channel for consistency
            binary_3ch = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            
            step_info = {
                "step": PreprocessingStep.BINARIZATION.value,
                "parameters": {"method": "adaptive_gaussian", "block_size": 11, "C": 2},
                "success": True
            }
            
            return binary_3ch, step_info
            
        except Exception as e:
            logger.warning(f"Binarization failed: {e}")
            return image, {"step": PreprocessingStep.BINARIZATION.value, "success": False, "error": str(e)}
    
    def _apply_morphological_operations(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply morphological operations to clean up the binary image
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Define kernels for morphological operations
            kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            kernel_medium = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            
            # Remove small noise
            cleaned = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_small)
            
            # Fill small gaps in characters
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_medium)
            
            # Convert back to 3-channel
            cleaned_3ch = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
            
            step_info = {
                "step": PreprocessingStep.MORPHOLOGICAL_OPERATIONS.value,
                "parameters": {"operations": ["opening", "closing"]},
                "success": True
            }
            
            return cleaned_3ch, step_info
            
        except Exception as e:
            logger.warning(f"Morphological operations failed: {e}")
            return image, {"step": PreprocessingStep.MORPHOLOGICAL_OPERATIONS.value, "success": False, "error": str(e)}
    
    def _calculate_noise_level(self, image: np.ndarray) -> float:
        """
        Calculate noise level in the image using Laplacian variance
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            # Normalize to 0-1 range (higher values indicate more noise)
            return min(laplacian_var / 1000.0, 1.0)
        except:
            return 0.0
    
    def _assess_image_quality(self, original: np.ndarray, processed: np.ndarray) -> float:
        """
        Assess the quality of the processed image
        
        Returns:
            Quality score between 0 and 1 (higher is better)
        """
        try:
            # Convert to grayscale for analysis
            orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            proc_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            
            # Calculate contrast (standard deviation of pixel intensities)
            contrast_score = np.std(proc_gray) / 255.0
            
            # Calculate sharpness using Laplacian variance
            sharpness_score = min(cv2.Laplacian(proc_gray, cv2.CV_64F).var() / 1000.0, 1.0)
            
            # Calculate noise level (lower is better)
            noise_level = self._calculate_noise_level(processed)
            noise_score = 1.0 - noise_level
            
            # Text clarity score based on edge strength
            edges = cv2.Canny(proc_gray, 50, 150)
            text_clarity_score = np.sum(edges > 0) / edges.size
            
            # Weighted overall score
            overall_score = (
                contrast_score * 0.3 +
                sharpness_score * 0.3 +
                noise_score * 0.2 +
                text_clarity_score * 0.2
            )
            
            return min(overall_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return 0.5  # Default middle score