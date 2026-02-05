
# Sample implementation for improved OCR processing

import cv2
import numpy as np
import pytesseract
from PIL import Image

class ImprovedOCRProcessor:
    """Enhanced OCR processor for KJV Bible text"""
    
    def __init__(self):
        self.tesseract_config = r'--oem 3 --psm 3 -c preserve_interword_spaces=1'
        
    def preprocess_image(self, image_path):
        """Apply preprocessing for better OCR accuracy"""
        # Read image
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Detect and correct skew
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        if angle > 0:
            (h, w) = thresh.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            thresh = cv2.warpAffine(thresh, M, (w, h),
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return thresh
    
    def detect_print_blocks(self, preprocessed_image):
        """Detect print blocks and columns"""
        # Find contours
        contours, _ = cv2.findContours(
            preprocessed_image, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter and sort contours by area
        blocks = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum block size
                x, y, w, h = cv2.boundingRect(contour)
                blocks.append({
                    'x': x, 'y': y, 
                    'width': w, 'height': h,
                    'area': area
                })
        
        # Sort blocks by position (top to bottom, left to right)
        blocks.sort(key=lambda b: (b['y'], b['x']))
        
        return blocks
    
    def extract_text_with_confidence(self, preprocessed_image):
        """Extract text with confidence scores"""
        # Get detailed OCR data
        data = pytesseract.image_to_data(
            preprocessed_image, 
            output_type=pytesseract.Output.DICT,
            config=self.tesseract_config
        )
        
        # Process results
        results = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            if int(data['conf'][i]) > 0:  # Has confidence score
                results.append({
                    'text': data['text'][i],
                    'confidence': data['conf'][i],
                    'x': data['left'][i],
                    'y': data['top'][i],
                    'width': data['width'][i],
                    'height': data['height'][i]
                })
        
        return results
    
    def validate_kjv_text(self, extracted_text):
        """Validate extracted text against KJV patterns"""
        validations = {
            'verse_patterns': 0,
            'book_names': 0,
            'common_words': 0,
            'total_confidence': 0
        }
        
        # Check for verse patterns (e.g., "3:16")
        import re
        verse_pattern = r'\d+:\d+'
        validations['verse_patterns'] = len(
            re.findall(verse_pattern, extracted_text)
        )
        
        # Check for book names
        kjv_books = ['Genesis', 'Exodus', 'Matthew', 'John', 'Revelation']
        for book in kjv_books:
            if book in extracted_text:
                validations['book_names'] += 1
        
        # Check common KJV words
        common_words = ['the', 'and', 'of', 'to', 'that', 'in', 'he', 
                       'shall', 'unto', 'for', 'LORD', 'God', 'Jesus']
        for word in common_words:
            validations['common_words'] += extracted_text.count(word)
        
        return validations

# Usage example:
processor = ImprovedOCRProcessor()
image_path = "bible_page.png"

# Preprocess
preprocessed = processor.preprocess_image(image_path)

# Detect blocks
blocks = processor.detect_print_blocks(preprocessed)
print(f"Found {len(blocks)} print blocks")

# Extract text
text_data = processor.extract_text_with_confidence(preprocessed)
avg_confidence = np.mean([t['confidence'] for t in text_data if t['confidence'] > 0])
print(f"Average OCR confidence: {avg_confidence:.2f}%")

# Validate
full_text = ' '.join([t['text'] for t in text_data])
validation = processor.validate_kjv_text(full_text)
print(f"Validation results: {validation}")
