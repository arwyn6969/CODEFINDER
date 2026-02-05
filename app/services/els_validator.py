"""
ELS Validator Service
=====================
Statistical validation for ELS findings using Monte Carlo simulations.
Calculates P-values and Z-scores to determine if findings are statistically significant
or likely due to random chance.
"""

import random
import statistics
from typing import List, Dict, Any
from dataclasses import dataclass
from app.services.prophetic_analyzer import PropheticAnalyzerService

@dataclass
class ValidationResult:
    p_value: float
    z_score: float
    num_trials: int
    better_matches: int
    mean_best_spread: float
    std_dev_spread: float
    observed_spread: int

class ELSValidator:
    def __init__(self):
        # Hebrew alphabet for random generation
        self.alphabet = list("אבגדהוזחטיכלמנסעפצקרשת")
        self.analyzer = PropheticAnalyzerService()

    def _generate_random_term(self, length: int) -> str:
        return "".join(random.choices(self.alphabet, k=length))

    def run_monte_carlo_simulation(
        self, 
        text: str, 
        observed_best_spread: int,
        term_structure: List[Dict[str, Any]],
        num_trials: int = 1000
    ) -> ValidationResult:
        """
        Run Monte Carlo simulation to validate a triple convergence finding.
        
        Args:
            text: Source text (Torah)
            observed_best_spread: The spread of the actual finding (e.g., 5)
            term_structure: Structure of terms to simulate, e.g.:
                [
                    {"name": "PEPE", "length": 3},
                    {"name": "MEME", "length": 4},
                    {"name": "FROG", "length": 3}
                ]
            num_trials: Number of random simulations to run.
            
        Returns:
            ValidationResult with P-value and stats.
        """
        best_spreads = []
        better_matches = 0
        
        print(f"Starting Monte Carlo simulation ({num_trials} trials)...")
        
        for i in range(num_trials):
            # 1. Generate random terms matching the structure
            random_groups = []
            for item in term_structure:
                # Generate 1 random term per group to keep it fair/comparable
                # (Or match the number of variants used in original search?)
                # To be conservative, let's generate 1 term of same length.
                rand_term = self._generate_random_term(item["length"])
                random_groups.append({
                    "name": item["name"],
                    "terms": [rand_term]
                })
            
            # 2. Find convergences for random terms
            # We look for ANY convergence within reasonable bound (e.g. 100)
            # If nothing found, spread is infinite
            zones = self.analyzer.find_triple_convergence(
                text, 
                random_groups, 
                max_spread=100 # Look for tight ones
            )
            
            if zones:
                best = zones[0].spread
            else:
                best = 999999 # No convergence found
            
            best_spreads.append(best)
            
            if best <= observed_best_spread:
                better_matches += 1
                
            if (i + 1) % 100 == 0:
                print(f"  Trial {i+1}/{num_trials}: Best spread so far: {min(best_spreads)}")

        # 3. Calculate statistics
        valid_spreads = [s for s in best_spreads if s < 999999]
        if not valid_spreads:
            mean = 999999
            std_dev = 0
        else:
            mean = statistics.mean(valid_spreads)
            std_dev = statistics.stdev(valid_spreads) if len(valid_spreads) > 1 else 0
            
        # P-value: Probability of finding a result as good or better by chance
        # Add 1 to numerator and denominator for Laplace smoothing? 
        # Standard definition: better_matches / num_trials
        p_value = better_matches / num_trials
        
        # Z-score: (X - µ) / σ
        # (Observed - Mean Random) / StdDev
        z_score = 0.0
        if std_dev > 0:
            z_score = (observed_best_spread - mean) / std_dev
            
        return ValidationResult(
            p_value=p_value,
            z_score=z_score,
            num_trials=num_trials,
            better_matches=better_matches,
            mean_best_spread=mean,
            std_dev_spread=std_dev,
            observed_spread=observed_best_spread
        )
