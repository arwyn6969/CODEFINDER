#!/usr/bin/env python3
"""
Character Image Extractor

Extracts individual character images from source scans based on 
bounding boxes stored in the database. Generates side-by-side
comparison sheets for forensic analysis.
"""

import sqlite3
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Optional
import json

from db_persistence import get_connection, get_character_instances, init_database


class CharacterExtractor:
    """Extract and compare character images from scans."""
    
    def __init__(self, output_dir: str = "reports/character_extraction"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        init_database()
        self.conn = get_connection()
    
    def get_source_image_path(self, source_name: str, page_number: int) -> Optional[Path]:
        """Get the path to a source image."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT p.image_path 
            FROM pages p
            JOIN sources s ON p.source_id = s.id
            WHERE s.name = ? AND p.page_number = ?
        """, (source_name, page_number))
        
        row = cursor.fetchone()
        if row:
            return Path(row['image_path'])
        
        # Fallback: construct path from source name
        if 'wright' in source_name.lower():
            base = Path("data/sources/folger_sonnets_1609")
        else:
            base = Path("data/sources/folger_sonnets_1609_aspley")
        
        # Find matching image
        for img in sorted(base.glob("*.jpg")):
            if f"{page_number:03d}" in img.name or f"_{page_number:02d}_" in img.name:
                return img
        
        return None
    
    def extract_character_crop(self, image: Image.Image, x: int, y: int, 
                                width: int, height: int, padding: int = 5) -> Image.Image:
        """Extract a character crop from an image with padding."""
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.width, x + width + padding)
        y2 = min(image.height, y + height + padding)
        
        return image.crop((x1, y1, x2, y2))
    
    def extract_character_images(self, character: str, source_name: str = None,
                                  limit: int = 20) -> List[Tuple[Image.Image, Dict]]:
        """Extract character images from stored bounding boxes."""
        instances = get_character_instances(
            self.conn,
            source_name=source_name,
            character=character,
            min_confidence=60,
            limit=limit
        )
        
        results = []
        current_page_image = None
        current_page_path = None
        
        for inst in instances:
            # Load page image if needed
            page_path = self.get_source_image_path(inst['source_name'], inst['page_number'])
            if page_path and page_path != current_page_path:
                current_page_path = page_path
                try:
                    current_page_image = Image.open(page_path).convert('RGB')
                except Exception as e:
                    print(f"Error loading {page_path}: {e}")
                    continue
            
            if current_page_image is None:
                continue
            
            # Extract crop
            crop = self.extract_character_crop(
                current_page_image,
                inst['x'], inst['y'],
                inst['width'], inst['height']
            )
            results.append((crop, inst))
        
        return results
    
    def create_comparison_sheet(self, character: str, 
                                 sources: List[str] = None,
                                 samples_per_source: int = 10) -> Path:
        """Create a side-by-side comparison sheet for a character."""
        
        if sources is None:
            sources = ['wright', 'aspley']
        
        # Cell dimensions
        cell_size = 80
        padding = 10
        header_height = 50
        
        # Get samples from each source
        source_samples = {}
        for source in sources:
            samples = self.extract_character_images(character, source, samples_per_source)
            source_samples[source] = samples
        
        # Calculate dimensions
        max_samples = max(len(s) for s in source_samples.values()) if source_samples else 0
        if max_samples == 0:
            print(f"No samples found for '{character}'")
            return None
        
        width = len(sources) * (cell_size + padding) + padding
        height = header_height + max_samples * (cell_size + padding) + padding
        
        # Create image
        sheet = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(sheet)
        
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
            small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        except:
            font = ImageFont.load_default()
            small_font = font
        
        # Draw title
        char_display = character if character.isprintable() else f"U+{ord(character):04X}"
        draw.text((padding, 5), f"Character: '{char_display}'", fill='black', font=font)
        
        # Draw headers and samples
        for col, source in enumerate(sources):
            x = padding + col * (cell_size + padding)
            
            # Header
            draw.text((x, 30), source.upper()[:8], fill='#333', font=small_font)
            
            # Samples
            samples = source_samples.get(source, [])
            for row, (crop, inst) in enumerate(samples):
                y = header_height + row * (cell_size + padding)
                
                # Resize crop to fit cell
                crop.thumbnail((cell_size - 4, cell_size - 4), Image.Resampling.LANCZOS)
                
                # Center in cell
                paste_x = x + (cell_size - crop.width) // 2
                paste_y = y + (cell_size - crop.height) // 2
                
                # Draw cell border
                draw.rectangle([(x, y), (x + cell_size, y + cell_size)], 
                              outline='#ddd', width=1)
                
                # Paste crop
                sheet.paste(crop, (paste_x, paste_y))
        
        # Save
        safe_char = character.replace('/', 'SLASH').replace('\\', 'BSLASH')
        if ord(character) > 127:
            safe_char = f"U{ord(character):04X}"
        
        output_path = self.output_dir / f"compare_{safe_char}.png"
        sheet.save(output_path)
        print(f"Saved comparison sheet: {output_path}")
        
        return output_path
    
    def generate_outlier_comparisons(self):
        """Generate comparison sheets for statistically significant outliers."""
        
        # Characters identified as significant in our analysis
        outliers = [
            ('J', 'Capital J - 6.5x ratio'),
            ('O', 'Capital O - 2.2x ratio'),
            ("'", 'Straight quote - major outlier'),
            ('f', 'Lowercase f - significant'),
            ('0', 'Digit zero'),
            ('1', 'Digit one'),
            ('2', 'Digit two'),
            ('-', 'Hyphen - 2.5x ratio'),
            (';', 'Semicolon - 3.08x ratio'),
        ]
        
        print("="*60)
        print("GENERATING OUTLIER CHARACTER COMPARISONS")
        print("="*60)
        
        for char, description in outliers:
            print(f"\nProcessing: {description}")
            self.create_comparison_sheet(char)
        
        print("\n✅ Outlier comparison sheets generated!")
        return self.output_dir


def main():
    """Main entry point."""
    extractor = CharacterExtractor()
    
    # Check if database has data
    cursor = extractor.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM character_instances")
    count = cursor.fetchone()[0]
    
    if count == 0:
        print("⚠️  Database is empty! Run database population script first.")
        print("   python3 populate_database.py")
        return
    
    print(f"Database has {count:,} character instances")
    
    # Generate outlier comparisons
    extractor.generate_outlier_comparisons()


if __name__ == "__main__":
    main()
