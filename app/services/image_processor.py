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


@dataclass
class LayoutRegion:
    """Represents a detected layout region (column, line, etc.)"""
    x: int
    y: int
    width: int
    height: int
    region_type: str  # 'column', 'line', 'block'
    confidence: float


class LayoutAnalyzer:
    """
    Analyzes document layout using projection profiles.
    Integrated from BIBLE OCR project for enhanced column/line detection.
    """

    def __init__(self):
        self.min_column_width = 100
        self.min_line_height = 20
        self.text_threshold = 50

    def detect_columns(self, image: Image.Image) -> List[LayoutRegion]:
        """Detect columns using vertical projection profile"""
        gray = np.array(image.convert('L'))
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        vertical_proj = np.sum(binary, axis=0)
        kernel = np.ones(10) / 10
        smoothed_proj = np.convolve(vertical_proj, kernel, mode='same')

        height, width = binary.shape
        threshold = np.mean(smoothed_proj) * 0.3

        columns = []
        in_column = False
        start_x = 0

        for x in range(width):
            has_content = smoothed_proj[x] > threshold
            if has_content and not in_column:
                start_x = x
                in_column = True
            elif not has_content and in_column:
                col_width = x - start_x
                if col_width > self.min_column_width:
                    columns.append(LayoutRegion(
                        x=start_x, y=0, width=col_width, height=height,
                        region_type='column',
                        confidence=smoothed_proj[start_x:x].mean() / 255.0
                    ))
                in_column = False

        if in_column:
            col_width = width - start_x
            if col_width > self.min_column_width:
                columns.append(LayoutRegion(
                    x=start_x, y=0, width=col_width, height=height,
                    region_type='column',
                    confidence=smoothed_proj[start_x:].mean() / 255.0
                ))

        return columns

    def detect_lines(self, image: Image.Image) -> List[LayoutRegion]:
        """Detect text lines using horizontal projection profile"""
        gray = np.array(image.convert('L'))
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        horizontal_proj = np.sum(binary, axis=1)
        kernel = np.ones(5) / 5
        smoothed_proj = np.convolve(horizontal_proj, kernel, mode='same')

        height, width = binary.shape
        threshold = np.mean(smoothed_proj) * 0.2

        lines = []
        in_line = False
        start_y = 0

        for y in range(height):
            has_content = smoothed_proj[y] > threshold
            if has_content and not in_line:
                start_y = y
                in_line = True
            elif not has_content and in_line:
                line_height = y - start_y
                if line_height > self.min_line_height:
                    lines.append(LayoutRegion(
                        x=0, y=start_y, width=width, height=line_height,
                        region_type='line',
                        confidence=smoothed_proj[start_y:y].mean() / 255.0
                    ))
                in_line = False

        if in_line:
            line_height = height - start_y
            if line_height > self.min_line_height:
                lines.append(LayoutRegion(
                    x=0, y=start_y, width=width, height=line_height,
                    region_type='line',
                    confidence=smoothed_proj[start_y:].mean() / 255.0
                ))

        return lines

    def segment_layout(self, image: Image.Image) -> Dict[str, List[LayoutRegion]]:
        """Perform complete layout segmentation"""
        logger.info("Starting layout segmentation...")
        columns = self.detect_columns(image)
        logger.info(f"Detected {len(columns)} columns")

        all_lines = []
        for col in columns:
            col_image = image.crop((col.x, col.y, col.x + col.width, col.y + col.height))
            lines = self.detect_lines(col_image)
            for line in lines:
                all_lines.append(LayoutRegion(
                    x=col.x + line.x, y=col.y + line.y,
                    width=line.width, height=line.height,
                    region_type='line', confidence=line.confidence
                ))

        logger.info(f"Detected {len(all_lines)} lines across all columns")
        return {'columns': columns, 'lines': all_lines}

class ImageProcessor:
    """
    Advanced image processor for historical document enhancement
    Optimized for ornate fonts and aged paper
    """
    
    def __init__(self):
        self.preprocessing_steps = []
        self.layout_analyzer = LayoutAnalyzer()
        
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

    def ocr_with_layout(self, image: Image.Image, timeout_seconds: int = 15) -> Dict[str, Any]:
        """
        Perform layout-aware OCR using detected columns and lines.
        Returns detailed results with confidence metrics.
        Integrated from BIBLE OCR project.
        """
        import pytesseract
        from pytesseract import Output
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

        logger.info("Starting layout-aware OCR...")
        layout = self.layout_analyzer.segment_layout(image)

        all_text = []
        all_confidences = []
        line_results = []

        for i, line_region in enumerate(layout['lines']):
            if i >= 20:  # Limit for performance
                break

            line_img = image.crop((
                line_region.x, line_region.y,
                line_region.x + line_region.width,
                line_region.y + line_region.height
            ))

            def ocr_line():
                config = "--oem 3 --psm 7"
                data = pytesseract.image_to_data(line_img, config=config, output_type=Output.DICT)
                text = pytesseract.image_to_string(line_img, config=config).strip()
                return text, data

            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(ocr_line)
                    text, data = future.result(timeout=timeout_seconds)

                    confs = [int(c) for c in data.get('conf', []) if str(c).isdigit() and int(c) >= 0]
                    words = [w for w in data.get('text', []) if w and w.strip()]

                    if text and words:
                        line_results.append({
                            'line_number': i + 1,
                            'region': {
                                'x': line_region.x, 'y': line_region.y,
                                'width': line_region.width, 'height': line_region.height
                            },
                            'text': text,
                            'words': words,
                            'confidences': confs,
                            'avg_conf': sum(confs) / len(confs) if confs else 0,
                            'word_count': len(words)
                        })
                        all_text.append(text)
                        all_confidences.extend(confs)

            except FutureTimeoutError:
                logger.warning(f"Line {i+1} OCR timed out")
            except Exception as e:
                logger.warning(f"Line {i+1} OCR failed: {e}")

        combined_text = '\n'.join(all_text)
        avg_conf = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        median_conf = sorted(all_confidences)[len(all_confidences)//2] if all_confidences else 0

        return {
            'combined_text': combined_text,
            'total_words': sum(r['word_count'] for r in line_results),
            'total_lines': len(line_results),
            'avg_confidence': avg_conf,
            'median_confidence': median_conf,
            'line_results': line_results,
            'layout_info': {
                'columns_detected': len(layout['columns']),
                'lines_detected': len(layout['lines'])
            }
        }