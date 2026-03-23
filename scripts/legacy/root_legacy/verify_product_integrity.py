import json
import csv
from pathlib import Path
import os
import sys

def check_integrity():
    base_dir = Path("reports")
    type_case_dir = base_dir / "digital_type_case"
    sonnet_dir = base_dir / "sonnet_print_block_analysis"
    
    print("=== Quality Assurance: Product Integrity Check ===\n")
    
    # 1. Digital Type Case DB
    db_path = type_case_dir / "type_case_db.json"
    if not db_path.exists():
        print(f"[FAIL] Type Case DB not found: {db_path}")
        return
    
    try:
        with open(db_path) as f:
            db_data = json.load(f)
        print(f"[PASS] Type Case DB loaded. Total records: {len(db_data)}")
    except Exception as e:
        print(f"[FAIL] Type Case DB corrupted: {e}")
        return

    # 2. Image Asset Verification
    # Check a sample or count total files
    images_dir = type_case_dir / "images"
    if not images_dir.exists():
        print(f"[FAIL] Images directory not found: {images_dir}")
    else:
        # Recursively count images
        image_count = 0
        for root, dirs, files in os.walk(images_dir):
            image_count += len([f for f in files if f.endswith('.png')])
            
        print(f"[INFO] Total Image Assets on Disk: {image_count}")
        
        # Cross-reference
        if abs(len(db_data) - image_count) > 1000: # allow some leniency for anomalies or multi-block
             print(f"[WARN] Integrity Mismatch: DB has {len(db_data)} records but found {image_count} images.")
        else:
             print(f"[PASS] Image count aligns with Database.")

    # 3. Anomaly Map
    anomaly_path = sonnet_dir / "anomalies_map.csv"
    if not anomaly_path.exists():
        print(f"[FAIL] Anomaly Map CSV not found: {anomaly_path}")
    else:
        try:
            with open(anomaly_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            print(f"[PASS] Anomaly Map loaded. Total anomalies: {len(rows)}")
        except Exception as e:
            print(f"[FAIL] Anomaly Map CSV corrupted: {e}")

    # 4. HTML Index
    html_path = type_case_dir / "index.html"
    if html_path.exists() and html_path.stat().st_size > 1000:
        print(f"[PASS] HTML Contact Sheet exists ({html_path.stat().st_size // 1024} KB).")
    else:
        print(f"[FAIL] HTML Contact Sheet missing or empty.")

    print("\n=== QA Check Complete ===")

if __name__ == "__main__":
    check_integrity()
