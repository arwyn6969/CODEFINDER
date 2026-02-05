#!/usr/bin/env python3
"""
Verify Prophetic Services
=========================
Smoke test for the new PropheticAnalyzerService and ELSValidator.
"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from app.services.prophetic_analyzer import PropheticAnalyzerService
from app.services.els_validator import ELSValidator

def main():
    print("Initializing services...")
    analyzer = PropheticAnalyzerService()
    validator = ELSValidator()
    print("Services initialized.")

    # 1. Test Triple Convergence (Dummy Data with Skip 2)
    print("\n1. Testing Triple Convergence logic...")
    
    # Construct text with Skip 2 patterns
    # Text length 100
    chars = ["X"] * 100
    
    # G1: "ABC" at start 10, skip 2 -> indices 10, 12, 14
    chars[10] = "A"
    chars[12] = "B"
    chars[14] = "C"
    
    # G2: "DEF" at start 15, skip 2 -> indices 15, 17, 19
    chars[15] = "D"
    chars[17] = "E"
    chars[19] = "F"
    
    # G3: "HIJ" at start 20, skip 2 -> indices 20, 22, 24
    chars[20] = "H"
    chars[22] = "I"
    chars[24] = "J"
    
    text = "".join(chars)
    
    groups = [
        {"name": "G1", "terms": ["ABC"]},
        {"name": "G2", "terms": ["DEF"]},
        {"name": "G3", "terms": ["HIJ"]}
    ]
    
    # Expected convergence:
    # A=10, D=15, H=20.
    # Center approx 15. Spread = 20-10 = 10.
    
    # Analyzer default min_skip=2, so it should find these.
    zones = analyzer.find_triple_convergence(text, groups, max_spread=50)
    print(f"Found {len(zones)} zones.")
    if zones:
        print(f"Best zone center: {zones[0].center_index}")
        print(f"Spread: {zones[0].spread}")
        print(f"Terms found: {[r.term for r in zones[0].terms.values()]}")
        
        if zones[0].spread <= 15:
            print("SUCCESS: Found tight convergence as expected.")
        else:
            print("WARNING: Spread larger than expected.")
    else:
        print("FAILURE: No zones found.")
    
    # 2. Test Monte Carlo Plumbing
    print("\n2. Testing Monte Carlo plumbing (5 trials)...")
    term_structure = [
        {"name": "G1", "length": 3},
        {"name": "G2", "length": 3},
        {"name": "G3", "length": 3}
    ]
    
    result = validator.run_monte_carlo_simulation(text, observed_best_spread=10, term_structure=term_structure, num_trials=5)
    print(f"Simulation complete.")
    print(f"P-value: {result.p_value}")
    print(f"Stats: Mean {result.mean_best_spread}, StdDev {result.std_dev_spread}")

if __name__ == "__main__":
    main()
