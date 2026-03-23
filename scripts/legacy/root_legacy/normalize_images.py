#!/usr/bin/env python3
"""
Image Normalization Utility
===========================
Rescales Aspley images to match Wright DPI for consistent per-character analysis.

Usage:
    python3 normalize_images.py
"""

import logging
from pathlib import Path
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Known scale factor from RANSAC alignment
# Aspley is ~36% larger, so we need to scale DOWN by 0.64
SCALE_FACTOR = 0.64

def get_average_dimensions(image_dir: Path) -> tuple[int, int]:
    """Calculate average image dimensions for a directory."""
    widths = []
    heights = []
    
    for img_path in image_dir.glob("*.jpg"):
        try:
            with Image.open(img_path) as img:
                widths.append(img.width)
                heights.append(img.height)
        except Exception as e:
            logger.warning(f"Could not read {img_path}: {e}")
    
    if not widths:
        return 0, 0
    
    return int(sum(widths) / len(widths)), int(sum(heights) / len(heights))


def normalize_aspley_images(
    wright_dir: Path,
    aspley_dir: Path,
    output_dir: Path,
    scale_factor: float = SCALE_FACTOR
) -> int:
    """
    Normalize Aspley images to match Wright dimensions.
    
    Args:
        wright_dir: Directory containing Wright images
        aspley_dir: Directory containing Aspley images
        output_dir: Directory to save normalized images
        scale_factor: Scale factor to apply (default from RANSAC)
        
    Returns:
        Number of images processed
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get Wright baseline dimensions
    wright_avg_w, wright_avg_h = get_average_dimensions(wright_dir)
    logger.info(f"Wright average dimensions: {wright_avg_w} x {wright_avg_h}")
    
    # Get Aspley dimensions for verification
    aspley_avg_w, aspley_avg_h = get_average_dimensions(aspley_dir)
    logger.info(f"Aspley average dimensions: {aspley_avg_w} x {aspley_avg_h}")
    
    # Calculate actual scale
    actual_scale = wright_avg_w / aspley_avg_w if aspley_avg_w > 0 else scale_factor
    logger.info(f"Calculated scale: {actual_scale:.4f} (expected: {scale_factor:.4f})")
    
    # Use calculated scale for precision
    processed = 0
    
    for img_path in sorted(aspley_dir.glob("*.jpg")):
        try:
            with Image.open(img_path) as img:
                # Calculate new dimensions
                new_w = int(img.width * actual_scale)
                new_h = int(img.height * actual_scale)
                
                # Resize with high-quality resampling
                resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
                # Save to output directory
                output_path = output_dir / img_path.name
                resized.save(output_path, quality=95)
                
                processed += 1
                if processed % 10 == 0:
                    logger.info(f"Processed {processed} images...")
                    
        except Exception as e:
            logger.error(f"Failed to process {img_path}: {e}")
    
    logger.info(f"Normalization complete: {processed} images saved to {output_dir}")
    return processed


def main():
    wright_dir = Path("data/sources/folger_sonnets_1609")
    aspley_dir = Path("data/sources/folger_sonnets_1609_aspley")
    output_dir = Path("data/sources/folger_sonnets_1609_aspley_normalized")
    
    logger.info("Starting Aspley image normalization...")
    count = normalize_aspley_images(wright_dir, aspley_dir, output_dir)
    logger.info(f"Done. {count} images normalized.")


if __name__ == "__main__":
    main()
