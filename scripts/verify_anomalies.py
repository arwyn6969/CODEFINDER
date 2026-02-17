#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import json
import os
from app.services.flash_verifier import FlashVerifier

def main():
    print("=== Gemini Flash Anomaly Verification ===\n")
    
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY not found in environment.")
        print("Please export your API key to run this script:")
        print("  export GOOGLE_API_KEY='your_key_here'")
        sys.exit(1)
        
    try:
        verifier = FlashVerifier(api_key=api_key)
    except Exception as e:
        print(f"❌ Failed to initialize verifier: {e}")
        sys.exit(1)
        
    # Load filtered anomalies
    input_path = project_root / 'reports/ink_dot_analysis_v2/filtered_analysis.json'
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        print("Please run the filtering step first.")
        sys.exit(1)
        
    with open(input_path) as f:
        data = json.load(f)
        
    candidates = data.get('anomalies', [])
    print(f"Loaded {len(candidates)} candidates from filtered analysis.")
    
    # Run verification
    output_path = project_root / 'reports/ink_dot_analysis_v2/verified_anomalies.json'
    results = verifier.verify_batch(candidates, str(output_path))
    
    # Summary
    true_anomalies = [r for r in results if r.get('is_anomaly')]
    print("\n=== Verification Complete ===")
    print(f"Total processed: {len(results)}")
    print(f"Verified True Anomalies: {len(true_anomalies)}")
    print(f"Rejected as False Positives: {len(results) - len(true_anomalies)}")
    print(f"Results saved to: {output_path}")
    
    if true_anomalies:
        print("\nTop Verified Anomalies:")
        for a in true_anomalies[:5]:
            print(f"- {Path(a['image_path']).name}: {a.get('classification')} (Conf: {a.get('confidence')})")

if __name__ == "__main__":
    main()
