#!/usr/bin/env python3
"""
Gemini Vision OCR Comparison
============================
Compare Tesseract OCR with Gemini 1.5 Pro Vision for historical document analysis.

This prototype tests whether modern VLMs can better handle:
- Long-s (ſ) recognition
- Ligature detection (ct, st, ff, fi, fl)
- Historical typography understanding
- Semantic text extraction
"""

import os
import json
import base64
from pathlib import Path
from datetime import datetime

# Check for Gemini API
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  google-generativeai not installed. Run: pip install google-generativeai")


def load_image_as_base64(image_path: Path) -> str:
    """Load image and convert to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_with_gemini(image_path: Path, api_key: str = None) -> dict:
    """
    Analyze a page image using Gemini 1.5 Pro Vision.
    
    Returns structured analysis including:
    - Full transcription
    - Long-s (ſ) instances detected
    - Ligatures found
    - Confidence assessment
    """
    if not GEMINI_AVAILABLE:
        return {"error": "google-generativeai not installed"}
    
    # Configure API
    api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return {"error": "No API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable."}
    
    genai.configure(api_key=api_key)
    
    # Use Gemini 2.5 Flash (latest with vision capabilities)
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    # Load image
    image_data = load_image_as_base64(image_path)
    
    # Craft prompt for historical document analysis
    prompt = """You are an expert paleographer analyzing a page from Shakespeare's Sonnets (1609 Quarto).

Please analyze this page and provide:

1. **TRANSCRIPTION**: Transcribe the text exactly as printed, preserving:
   - Long-s (ſ) — do NOT modernize to 's'
   - Ligatures (ct, st, ff, fi, fl, etc.) — note them with [lig:XX]
   - Original spelling and punctuation
   - Line breaks

2. **LONG-S COUNT**: How many instances of the long-s (ſ) character appear?

3. **LIGATURES FOUND**: List each ligature type and count:
   - ct: X
   - st: X
   - ff: X
   - etc.

4. **ANOMALIES**: Note any unusual marks, marginalia, or printing artifacts.

5. **CONFIDENCE**: Rate your confidence in the transcription (0-100%).

Respond in JSON format:
```json
{
  "transcription": "...",
  "long_s_count": N,
  "ligatures": {"ct": N, "st": N, ...},
  "anomalies": ["..."],
  "confidence": N,
  "notes": "..."
}
```"""
    
    try:
        # Send to Gemini
        response = model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": image_data}
        ])
        
        # Parse response
        response_text = response.text
        
        # Try to extract JSON from response
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            json_str = response_text[json_start:json_end].strip()
        elif "{" in response_text:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            json_str = response_text[json_start:json_end]
        else:
            return {
                "error": "Could not parse JSON from response",
                "raw_response": response_text
            }
        
        result = json.loads(json_str)
        result["raw_response"] = response_text
        result["model"] = "gemini-1.5-pro"
        result["timestamp"] = datetime.now().isoformat()
        
        return result
        
    except Exception as e:
        return {"error": str(e)}


def compare_with_tesseract(image_path: Path) -> dict:
    """Get Tesseract analysis for comparison."""
    try:
        import pytesseract
        from PIL import Image
        
        img = Image.open(image_path)
        
        # Get OCR text
        text = pytesseract.image_to_string(img, config='--psm 6')
        
        # Get confidence data
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config='--psm 6')
        
        # Calculate average confidence
        confidences = [int(c) for c in data['conf'] if c != '-1' and int(c) > 0]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        
        # Count Long-s (Tesseract typically misses these)
        long_s_count = text.count('ſ')
        
        # Count potential ligatures (will likely be 0)
        ligatures = {
            "ct": text.count("ct"),  # Tesseract won't preserve these
            "st": text.count("st"),
            "ff": text.count("ff"),
            "fi": text.count("fi"),
            "fl": text.count("fl"),
        }
        
        return {
            "transcription": text,
            "long_s_count": long_s_count,
            "ligatures": ligatures,
            "confidence": avg_conf,
            "model": "tesseract",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"error": str(e)}


def run_comparison(image_path: Path, output_dir: Path = None) -> dict:
    """Run side-by-side comparison of Tesseract vs Gemini."""
    
    output_dir = output_dir or Path("reports/ocr_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📄 Analyzing: {image_path.name}")
    print("=" * 60)
    
    # Tesseract analysis
    print("\n🔍 Running Tesseract OCR...")
    tesseract_result = compare_with_tesseract(image_path)
    
    if "error" not in tesseract_result:
        print(f"   ✓ Confidence: {tesseract_result['confidence']:.1f}%")
        print(f"   ✓ Long-s found: {tesseract_result['long_s_count']}")
        print(f"   ✓ Characters: {len(tesseract_result['transcription'])}")
    else:
        print(f"   ✗ Error: {tesseract_result['error']}")
    
    # Gemini analysis
    print("\n🤖 Running Gemini Vision...")
    gemini_result = analyze_with_gemini(image_path)
    
    if "error" not in gemini_result:
        print(f"   ✓ Confidence: {gemini_result.get('confidence', 'N/A')}%")
        print(f"   ✓ Long-s found: {gemini_result.get('long_s_count', 'N/A')}")
        if 'transcription' in gemini_result:
            print(f"   ✓ Characters: {len(gemini_result['transcription'])}")
    else:
        print(f"   ✗ Error: {gemini_result['error']}")
    
    # Comparison summary
    comparison = {
        "image": str(image_path),
        "tesseract": tesseract_result,
        "gemini": gemini_result,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save results
    output_file = output_dir / f"comparison_{image_path.stem}.json"
    with open(output_file, "w") as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Print comparison table
    print("\n" + "=" * 60)
    print("📊 COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<20} {'Tesseract':<20} {'Gemini':<20}")
    print("-" * 60)
    
    tess_conf = tesseract_result.get('confidence', 0)
    gem_conf = gemini_result.get('confidence', 0)
    print(f"{'Confidence':<20} {tess_conf:>6.1f}%{'':<12} {gem_conf:>6}%")
    
    tess_long_s = tesseract_result.get('long_s_count', 0)
    gem_long_s = gemini_result.get('long_s_count', 0)
    print(f"{'Long-s (ſ) count':<20} {tess_long_s:>6}{'':<13} {gem_long_s:>6}")
    
    print("-" * 60)
    
    return comparison


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare Tesseract vs Gemini Vision for historical document OCR"
    )
    parser.add_argument(
        "--image", "-i",
        type=Path,
        default=Path("data/sources/folger_sonnets_1609/page_010.jpg"),
        help="Image file to analyze"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("reports/ocr_comparison"),
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    if not args.image.exists():
        # Try to find a sample page
        sonnets_dir = Path("data/sources/folger_sonnets_1609")
        if sonnets_dir.exists():
            images = list(sonnets_dir.glob("*.jpg"))
            if images:
                args.image = images[9] if len(images) > 9 else images[0]  # Page 10 has sonnets
                print(f"Using: {args.image}")
        
    if not args.image.exists():
        print(f"❌ Image not found: {args.image}")
        return 1
    
    run_comparison(args.image, args.output)
    return 0


if __name__ == "__main__":
    exit(main())
