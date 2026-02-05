#!/usr/bin/env python3
"""
RUN PEPE MONTE CARLO
====================
Executes the statistical validation for the PEPE + MEME + FROG convergence.
"""
import sys
import os
import time
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from app.services.els_validator import ELSValidator

def load_torah():
    path = Path("app/data/torah.txt")
    if not path.exists():
        # Try full path if running from strange context
        path = Path("/Users/arwynhughes/Documents/CODEFINDER_PUBLISH/app/data/torah.txt")
    
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def main():
    print("Loading Torah...")
    text = load_torah()
    print(f"Torah loaded: {len(text)} letters")
    
    validator = ELSValidator()
    
    # Configuration based on actual findings
    # PEPE (פפי) = 3 letters
    # MEME (מימי) = 4 letters
    # FROG (צפר) = 3 letters
    term_structure = [
        {"name": "PEPE", "length": 3},
        {"name": "MEME", "length": 4},
        {"name": "FROG", "length": 3}
    ]
    
    observed_best_spread = 5
    num_trials = 100
    
    print(f"\nRunning {num_trials} Monte Carlo simulations...")
    print(f"Goal: Find random triplets (len 3,4,3) converging within {observed_best_spread} letters.")
    
    start_time = time.time()
    result = validator.run_monte_carlo_simulation(
        text, 
        observed_best_spread, 
        term_structure, 
        num_trials
    )
    duration = time.time() - start_time
    
    print("\n" + "="*40)
    print("RESULTS")
    print("="*40)
    print(f"Trials: {result.num_trials}")
    print(f"Better Matches (Spread <= 5): {result.better_matches}")
    print(f"P-Value: {result.p_value:.6f}")
    print(f"Z-Score: {result.z_score:.2f}")
    print(f"Average Random Best Spread: {result.mean_best_spread:.1f}")
    print(f"Time Taken: {duration:.1f}s")
    
    # Save results
    output = {
        "p_value": result.p_value,
        "z_score": result.z_score,
        "better_matches": result.better_matches,
        "trials": result.num_trials,
        "observed_spread": result.observed_spread,
        "mean_random_best": result.mean_best_spread
    }
    
    with open("pepe_monte_carlo_results.json", "w") as f:
        json.dump(output, f, indent=2)
        
    print("\nResults saved to pepe_monte_carlo_results.json")

if __name__ == "__main__":
    main()
