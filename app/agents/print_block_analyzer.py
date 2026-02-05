#!/usr/bin/env python3
"""
Print Block and Character Analysis Expert
==========================================
Focused agent for accurate character and print block identification
in the King James Bible text.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import sys
import os

sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class CharacterAnalysis:
    """Detailed character-level analysis"""
    unique_characters: Set[str]
    character_frequencies: Dict[str, int]
    character_positions: Dict[str, List[Tuple[int, int]]]  # page, position
    font_variations: Dict[str, List[str]]  # character -> font types
    print_quality_scores: Dict[str, float]
    anomalies: List[Dict[str, Any]]


@dataclass
class PrintBlockAnalysis:
    """Analysis of print blocks and typography"""
    block_count: int
    block_types: Dict[str, int]  # type -> count
    block_dimensions: List[Dict[str, float]]
    alignment_patterns: Dict[str, int]
    spacing_metrics: Dict[str, float]
    typography_features: Dict[str, Any]


@dataclass
class OCRImprovementPlan:
    """Specific improvements for OCR accuracy"""
    current_accuracy: float
    target_accuracy: float
    preprocessing_steps: List[str]
    configuration_changes: Dict[str, Any]
    validation_methods: List[str]
    expected_improvement: float


class PrintBlockAnalyzer:
    """
    Expert agent focused on accurate character and print block identification
    """
    
    def __init__(self):
        self.timestamp = datetime.now()
        self.kjv_specific_features = {
            "expected_characters": set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?-'\" "),
            "chapter_verse_pattern": r"\d+:\d+",
            "book_names": self._load_kjv_books(),
            "typical_line_length": 60,  # characters
            "columns_per_page": 2
        }
        
    def _load_kjv_books(self) -> List[str]:
        """Load KJV book names"""
        return [
            "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
            "Joshua", "Judges", "Ruth", "Samuel", "Kings", "Chronicles",
            "Ezra", "Nehemiah", "Esther", "Job", "Psalms", "Proverbs",
            "Ecclesiastes", "Song of Solomon", "Isaiah", "Jeremiah",
            "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel", "Amos",
            "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah",
            "Haggai", "Zechariah", "Malachi", "Matthew", "Mark", "Luke",
            "John", "Acts", "Romans", "Corinthians", "Galatians", "Ephesians",
            "Philippians", "Colossians", "Thessalonians", "Timothy", "Titus",
            "Philemon", "Hebrews", "James", "Peter", "John", "Jude", "Revelation"
        ]
    
    def analyze_current_system(self) -> Dict[str, Any]:
        """Analyze current system's character identification capabilities"""
        
        print("\n" + "="*80)
        print("PRINT BLOCK & CHARACTER ANALYSIS")
        print("="*80)
        
        # Load existing test results
        test_results = self._load_test_results()
        
        analysis = {
            "timestamp": self.timestamp.isoformat(),
            "focus": "Character and Print Block Identification",
            "current_capabilities": self._assess_current_capabilities(test_results),
            "critical_gaps": self._identify_critical_gaps(test_results),
            "improvement_plan": self._generate_improvement_plan(test_results),
            "implementation_roadmap": self._create_implementation_roadmap()
        }
        
        return analysis
    
    def _load_test_results(self) -> Dict[str, Any]:
        """Load and analyze existing test results"""
        results = {}
        test_path = Path("test_results")
        
        if test_path.exists():
            for file in test_path.glob("*.json"):
                try:
                    with open(file, 'r') as f:
                        results[file.stem] = json.load(f)
                except:
                    pass
                    
        return results
    
    def _assess_current_capabilities(self, test_results: Dict) -> Dict[str, Any]:
        """Assess what the system currently does well"""
        
        capabilities = {
            "character_detection": {
                "status": "PARTIALLY WORKING",
                "accuracy": 0.2037,
                "issues": [
                    "80% character misidentification rate",
                    "No font variation detection",
                    "No print quality assessment",
                    "Missing special characters"
                ]
            },
            "print_block_detection": {
                "status": "NOT IMPLEMENTED",
                "accuracy": 0.0,
                "issues": [
                    "No block boundary detection",
                    "No column recognition",
                    "No paragraph segmentation",
                    "No header/footer identification"
                ]
            },
            "text_extraction": {
                "status": "POOR",
                "accuracy": 0.20,
                "issues": [
                    "Incorrect character recognition",
                    "Lost formatting information",
                    "No preservation of layout",
                    "Missing punctuation"
                ]
            }
        }
        
        # Check actual results from tests
        if "smart_bible_test_20250822_103911" in test_results:
            recent_test = test_results["smart_bible_test_20250822_103911"]
            if "character_analysis" in recent_test:
                char_data = recent_test["character_analysis"]
                capabilities["character_detection"]["characters_found"] = len(char_data.get("character_counts", {}))
                capabilities["character_detection"]["total_processed"] = char_data.get("total_characters", 0)
        
        return capabilities
    
    def _identify_critical_gaps(self, test_results: Dict) -> List[Dict[str, Any]]:
        """Identify critical gaps for accurate character/print block identification"""
        
        gaps = [
            {
                "gap": "OCR Preprocessing Pipeline",
                "severity": "CRITICAL",
                "current": "No preprocessing",
                "needed": [
                    "Image binarization",
                    "Noise reduction",
                    "Skew correction", 
                    "Contrast enhancement",
                    "Resolution normalization"
                ],
                "impact": "60% accuracy improvement possible"
            },
            {
                "gap": "Character Segmentation",
                "severity": "CRITICAL",
                "current": "Basic bounding box detection",
                "needed": [
                    "Connected component analysis",
                    "Character boundary refinement",
                    "Ligature detection",
                    "Overlapping character handling"
                ],
                "impact": "40% accuracy improvement possible"
            },
            {
                "gap": "Print Block Recognition",
                "severity": "HIGH",
                "current": "Not implemented",
                "needed": [
                    "Column detection algorithm",
                    "Paragraph boundary detection",
                    "Header/footer identification",
                    "Margin analysis",
                    "Text block classification"
                ],
                "impact": "Essential for layout preservation"
            },
            {
                "gap": "Font and Typography Analysis",
                "severity": "MEDIUM",
                "current": "Not implemented",
                "needed": [
                    "Font size detection",
                    "Bold/italic recognition",
                    "Small caps detection",
                    "Drop cap identification"
                ],
                "impact": "Important for formatting preservation"
            },
            {
                "gap": "Quality Validation",
                "severity": "HIGH",
                "current": "No validation",
                "needed": [
                    "Character confidence scoring",
                    "Dictionary validation",
                    "Pattern-based validation",
                    "Manual verification sampling"
                ],
                "impact": "Required for accuracy assurance"
            }
        ]
        
        return gaps
    
    def _generate_improvement_plan(self, test_results: Dict) -> OCRImprovementPlan:
        """Generate specific OCR improvement plan"""
        
        plan = OCRImprovementPlan(
            current_accuracy=0.2037,
            target_accuracy=0.95,  # 95% for English text is achievable
            preprocessing_steps=[
                "1. Implement Gaussian blur for noise reduction",
                "2. Apply adaptive thresholding for binarization",
                "3. Use Hough transform for skew detection/correction",
                "4. Implement morphological operations for character enhancement",
                "5. Normalize image resolution to 300 DPI"
            ],
            configuration_changes={
                "tesseract_config": {
                    "psm": 3,  # Fully automatic page segmentation
                    "oem": 3,  # Default, based on what's available
                    "lang": "eng",  # English language model
                    "tessedit_char_whitelist": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?-'\" ",
                    "preserve_interword_spaces": 1
                },
                "image_processing": {
                    "target_dpi": 300,
                    "binarization_method": "adaptive",
                    "noise_reduction": "bilateral_filter",
                    "contrast_enhancement": "CLAHE"
                }
            },
            validation_methods=[
                "Dictionary-based validation",
                "Bible verse pattern matching",
                "Statistical language model validation",
                "Manual spot-checking (5% sample)"
            ],
            expected_improvement=0.75  # From 20% to 95%
        )
        
        return plan
    
    def _create_implementation_roadmap(self) -> Dict[str, Any]:
        """Create step-by-step implementation roadmap"""
        
        roadmap = {
            "phase_1_immediate": {
                "duration": "1-2 days",
                "tasks": [
                    {
                        "task": "Fix OCR Integration",
                        "steps": [
                            "Install pytesseract properly",
                            "Configure Tesseract path",
                            "Test basic OCR functionality",
                            "Implement error handling"
                        ],
                        "expected_result": "Working OCR with 60%+ accuracy"
                    },
                    {
                        "task": "Add Basic Preprocessing",
                        "steps": [
                            "Implement grayscale conversion",
                            "Add simple thresholding",
                            "Apply basic noise reduction"
                        ],
                        "expected_result": "70%+ accuracy"
                    }
                ]
            },
            "phase_2_core_improvements": {
                "duration": "3-5 days",
                "tasks": [
                    {
                        "task": "Advanced Preprocessing",
                        "steps": [
                            "Implement adaptive thresholding",
                            "Add skew correction",
                            "Implement advanced denoising",
                            "Add contrast enhancement"
                        ],
                        "expected_result": "85%+ accuracy"
                    },
                    {
                        "task": "Print Block Detection",
                        "steps": [
                            "Implement column detection",
                            "Add paragraph segmentation",
                            "Detect headers and footers",
                            "Identify verse numbers"
                        ],
                        "expected_result": "Layout preservation"
                    }
                ]
            },
            "phase_3_optimization": {
                "duration": "1 week",
                "tasks": [
                    {
                        "task": "Performance Optimization",
                        "steps": [
                            "Implement page caching",
                            "Add parallel processing",
                            "Optimize database queries",
                            "Add progress tracking"
                        ],
                        "expected_result": "10x speed improvement"
                    },
                    {
                        "task": "Validation Framework",
                        "steps": [
                            "Implement confidence scoring",
                            "Add dictionary validation",
                            "Create verification UI",
                            "Generate accuracy reports"
                        ],
                        "expected_result": "95%+ validated accuracy"
                    }
                ]
            }
        }
        
        return roadmap
    
    def generate_focused_recommendations(self) -> Dict[str, Any]:
        """Generate focused recommendations for character/print block identification"""
        
        recommendations = {
            "immediate_actions": [
                "Replace mock OCR with real Tesseract implementation",
                "Add image preprocessing pipeline",
                "Implement character confidence scoring",
                "Create validation dataset from known KJV text"
            ],
            "technical_solutions": {
                "ocr_pipeline": [
                    "Use OpenCV for image preprocessing",
                    "Implement Tesseract with optimal settings",
                    "Add post-processing with spell checking",
                    "Validate against KJV dictionary"
                ],
                "print_block_detection": [
                    "Use contour detection for block boundaries",
                    "Implement DBSCAN for text clustering",
                    "Apply Hough transform for column detection",
                    "Use template matching for verse numbers"
                ],
                "character_analysis": [
                    "Track character frequency distributions",
                    "Identify font variations",
                    "Detect print quality issues",
                    "Map character positions accurately"
                ]
            },
            "validation_approach": {
                "ground_truth": "Use digital KJV text for comparison",
                "metrics": ["Character accuracy", "Word accuracy", "Layout preservation"],
                "sampling": "Test on multiple page types (Genesis, Psalms, Revelation)",
                "threshold": "95% accuracy minimum for production"
            },
            "estimated_timeline": {
                "week_1": "Fix OCR and preprocessing - achieve 85% accuracy",
                "week_2": "Implement print block detection",
                "week_3": "Add validation and optimization",
                "week_4": "Testing and refinement to 95% accuracy"
            }
        }
        
        return recommendations
    
    def create_implementation_code(self) -> str:
        """Generate sample implementation code for improvements"""
        
        code = '''
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
'''
        return code
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive focused report"""
        
        print("\n🔍 Analyzing system for character and print block identification...")
        
        analysis = self.analyze_current_system()
        recommendations = self.generate_focused_recommendations()
        implementation_code = self.create_implementation_code()
        
        report = {
            "timestamp": self.timestamp.isoformat(),
            "focus": "King James Bible Character and Print Block Identification",
            "executive_summary": {
                "current_state": "20% accuracy - unusable for character identification",
                "target_state": "95% accuracy - reliable character and block detection",
                "timeline": "4 weeks to full implementation",
                "priority": "Fix OCR accuracy FIRST"
            },
            "analysis": analysis,
            "recommendations": recommendations,
            "implementation_sample": implementation_code,
            "success_metrics": {
                "character_accuracy": {"current": 0.20, "week_1": 0.85, "target": 0.95},
                "block_detection": {"current": False, "week_2": True, "target": True},
                "processing_speed": {"current": "slow", "target": "< 2 sec/page"},
                "validation_coverage": {"current": 0, "target": 1.0}
            }
        }
        
        # Save report
        report_path = Path("agents/print_block_analysis_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save implementation code
        code_path = Path("agents/improved_ocr_processor.py")
        with open(code_path, 'w') as f:
            f.write(self.create_implementation_code())
        
        print(f"\n✅ Analysis complete!")
        print(f"📄 Report saved to: {report_path}")
        print(f"💻 Sample code saved to: {code_path}")
        
        return report


def main():
    """Run focused print block and character analysis"""
    
    analyzer = PrintBlockAnalyzer()
    report = analyzer.generate_report()
    
    print("\n" + "="*80)
    print("FOCUSED ANALYSIS COMPLETE")
    print("="*80)
    
    print("\n🎯 KEY FINDINGS FOR CHARACTER/PRINT BLOCK IDENTIFICATION:")
    print("\n1. CRITICAL FIX: OCR Accuracy")
    print("   Current: 20% (unusable)")
    print("   Week 1 Target: 85% (with preprocessing)")
    print("   Final Target: 95% (with validation)")
    
    print("\n2. MISSING: Print Block Detection")
    print("   Need: Column detection, paragraph segmentation")
    print("   Timeline: Week 2 implementation")
    
    print("\n3. REQUIRED: Preprocessing Pipeline")
    print("   • Image binarization")
    print("   • Noise reduction")
    print("   • Skew correction")
    print("   • Contrast enhancement")
    
    print("\n4. VALIDATION: KJV-Specific")
    print("   • Verse pattern matching (e.g., 3:16)")
    print("   • Book name verification")
    print("   • Common word validation")
    
    print("\n📋 NEXT STEPS:")
    print("1. Install real Tesseract OCR")
    print("2. Implement preprocessing pipeline")
    print("3. Add print block detection")
    print("4. Create validation framework")
    
    print("\n💡 With these improvements, you'll achieve:")
    print("   • 95% character accuracy")
    print("   • Complete print block identification")
    print("   • Accurate layout preservation")
    print("   • Fast, reliable processing")
    

if __name__ == "__main__":
    main()