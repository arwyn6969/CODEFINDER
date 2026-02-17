#!/usr/bin/env python3
"""
Ink Density Measurement (Task 3)

Measures "percent black" within character bounding boxes to compare
ink saturation between editions.

This tests H4: Ink density difference < 5%
"""

import sqlite3
import csv
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict
import random

# Configuration
DB_PATH = Path("data/codefinder.db")
REPORT_DIR = Path("reports/ink_density")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Sample size (analyzing all 155k chars would take too long)
SAMPLE_SIZE = 500  # Per edition

def get_connection():
    return sqlite3.connect(DB_PATH)

def get_sample_instances(source, sample_size=500):
    """Get a random sample of character instances with image paths."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            ci.id,
            ci.character,
            ci.x, ci.y, ci.width, ci.height,
            p.image_path,
            s.name as source
        FROM character_instances ci
        JOIN pages p ON ci.page_id = p.id
        JOIN sources s ON p.source_id = s.id
        WHERE s.name = ? 
          AND ci.confidence > 0.6
          AND ci.width > 5 AND ci.height > 5
        ORDER BY RANDOM()
        LIMIT ?
    """, (source, sample_size))
    
    instances = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return instances

def calculate_ink_density(instance, image_cache):
    """Calculate percent black pixels within the character bbox."""
    
    img_path = instance['image_path']
    
    # Load image (with caching)
    if img_path not in image_cache:
        try:
            image_cache[img_path] = Image.open(img_path).convert('L')  # Grayscale
        except Exception as e:
            return None
    
    img = image_cache[img_path]
    
    # Crop character region
    x, y, w, h = instance['x'], instance['y'], instance['width'], instance['height']
    
    try:
        crop = img.crop((x, y, x + w, y + h))
    except Exception:
        return None
    
    # Count black pixels (< 128 threshold)
    pixels = np.array(crop)
    total_pixels = pixels.size
    
    if total_pixels == 0:
        return None
    
    black_pixels = np.sum(pixels < 128)
    density = black_pixels / total_pixels
    
    return density

def analyze_ink_density():
    """Task 3.1-3.4: Sample chars, calculate density, compare."""
    
    print("="*60)
    print("INK DENSITY MEASUREMENT")
    print("="*60)
    
    results = {}
    image_cache = {}
    
    for source in ['aspley', 'wright']:
        print(f"\nProcessing {source.upper()}...")
        instances = get_sample_instances(source, SAMPLE_SIZE)
        print(f"  Got {len(instances)} sample instances.")
        
        densities = []
        for inst in instances:
            density = calculate_ink_density(inst, image_cache)
            if density is not None:
                densities.append(density)
                inst['ink_density'] = density
        
        if densities:
            results[source] = {
                'count': len(densities),
                'mean': np.mean(densities),
                'std_dev': np.std(densities),
                'median': np.median(densities),
                'min': np.min(densities),
                'max': np.max(densities)
            }
            
            print(f"  Analyzed {len(densities)} characters.")
            print(f"  Mean Ink Density: {results[source]['mean']*100:.1f}%")
            print(f"  Std. Dev.: {results[source]['std_dev']*100:.1f}%")
    
    # Hypothesis Test H4
    print("\n" + "="*60)
    print("HYPOTHESIS TESTING")
    print("="*60)
    
    if 'aspley' in results and 'wright' in results:
        mean_diff = abs(results['aspley']['mean'] - results['wright']['mean'])
        h4_threshold = 0.05  # 5%
        h4_pass = mean_diff < h4_threshold
        
        print(f"\nH4: Ink density difference < 5%")
        print(f"  Aspley Mean: {results['aspley']['mean']*100:.1f}%")
        print(f"  Wright Mean: {results['wright']['mean']*100:.1f}%")
        print(f"  Difference: {mean_diff*100:.1f}%")
        print(f"  Result: {'PASS ✓' if h4_pass else 'FAIL ✗'}")
        
        return results, h4_pass, mean_diff
    
    return results, None, None

def save_results(results, h4_pass, mean_diff):
    """Save summary."""
    summary_path = REPORT_DIR / "ink_density_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("INK DENSITY ANALYSIS SUMMARY\n")
        f.write("="*40 + "\n\n")
        
        for source, stats in results.items():
            f.write(f"{source.upper()}:\n")
            for k, v in stats.items():
                if isinstance(v, float):
                    f.write(f"  {k}: {v*100:.2f}%\n")
                else:
                    f.write(f"  {k}: {v}\n")
            f.write("\n")
        
        if h4_pass is not None:
            f.write(f"H4 Result: {'PASS' if h4_pass else 'FAIL'}\n")
            f.write(f"Mean Difference: {mean_diff*100:.2f}%\n")
    
    print(f"\nSaved summary to: {summary_path}")

def main():
    results, h4_pass, mean_diff = analyze_ink_density()
    save_results(results, h4_pass, mean_diff)
    
    print("\n" + "="*60)
    print("INK DENSITY ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
