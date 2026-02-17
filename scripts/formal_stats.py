#!/usr/bin/env python3
"""
Formal Statistical Tests for Print Block Attribution
=====================================================

Runs rigorous statistical tests on the extracted forensic data:

1. Kolmogorov-Smirnov (KS) test — Are character dimension distributions the same?
2. Chi-Squared test — Are character frequency distributions independent?
3. Mann-Whitney U test — Non-parametric comparison of fingerprint features
4. Bootstrap confidence intervals — How robust is the similarity score?
5. Type measurement analysis — Body height, x-height, set width ratios

Usage:
    python scripts/formal_stats.py

Output:
    reports/statistical_analysis/
        formal_stats.json
        stats_report.html
"""

import sys
import json
import sqlite3
import numpy as np
import logging
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from app.services.block_fingerprinter import BlockFingerprinter

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("formal_stats")


class ForensicStatistics:
    """Formal statistical tests for print attribution."""
    
    def __init__(self, db_path: str = "data/forensic.db"):
        self.db_path = db_path
        self.fingerprinter = BlockFingerprinter()
    
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def gather_data(self) -> Dict[str, Dict]:
        """Gather per-source character data."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        sources = {}
        all_sources = cursor.execute("SELECT id, name FROM sources").fetchall()
        for row in all_sources:
            src_id = row['id']
            src_name = row['name']
            
            # Get all character instances
            instances = cursor.execute("""
                SELECT ci.character, ci.x, ci.y, ci.width, ci.height, ci.confidence,
                       p.page_number
                FROM character_instances ci
                JOIN pages p ON ci.page_id = p.id
                WHERE p.source_id = ?
            """, (src_id,)).fetchall()
            
            sources[src_name] = {
                'instances': [dict(r) for r in instances],
                'widths': [r['width'] for r in instances if r['width'] and r['width'] > 0],
                'heights': [r['height'] for r in instances if r['height'] and r['height'] > 0],
                'chars': [r['character'] for r in instances],
            }
        
        conn.close()
        return sources
    
    # =========================================================================
    # Test 1: Kolmogorov-Smirnov Test
    # =========================================================================
    
    def ks_test_dimensions(self, sources: Dict) -> Dict[str, Any]:
        """
        KS test: Are character dimension distributions from different sources 
        drawn from the same underlying distribution?
        
        If same printer → same type → dimensions should be similar → high p-value
        """
        results = {}
        source_names = list(sources.keys())
        
        for i in range(len(source_names)):
            for j in range(i+1, len(source_names)):
                s1, s2 = source_names[i], source_names[j]
                
                w1, w2 = np.array(sources[s1]['widths']), np.array(sources[s2]['widths'])
                h1, h2 = np.array(sources[s1]['heights']), np.array(sources[s2]['heights'])
                
                if len(w1) < 5 or len(w2) < 5:
                    continue
                
                # Manual KS statistic (avoid scipy dependency)
                ks_w = self._ks_statistic(w1, w2)
                ks_h = self._ks_statistic(h1, h2)
                
                # Approximate p-value using KS distribution
                n_eff = (len(w1) * len(w2)) / (len(w1) + len(w2))
                p_w = self._ks_pvalue(ks_w, n_eff)
                p_h = self._ks_pvalue(ks_h, n_eff)
                
                pair = f"{s1} vs {s2}"
                results[pair] = {
                    'ks_width': float(ks_w),
                    'ks_height': float(ks_h),
                    'p_width': float(p_w),
                    'p_height': float(p_h),
                    'n1': len(w1), 'n2': len(w2),
                    'verdict_width': 'SAME' if p_w > 0.05 else 'DIFFERENT',
                    'verdict_height': 'SAME' if p_h > 0.05 else 'DIFFERENT',
                }
        
        return results
    
    def _ks_statistic(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Compute two-sample KS statistic."""
        all_data = np.concatenate([data1, data2])
        all_data.sort()
        
        cdf1 = np.searchsorted(np.sort(data1), all_data, side='right') / len(data1)
        cdf2 = np.searchsorted(np.sort(data2), all_data, side='right') / len(data2)
        
        return float(np.max(np.abs(cdf1 - cdf2)))
    
    def _ks_pvalue(self, ks_stat: float, n_eff: float) -> float:
        """Approximate KS p-value using asymptotic formula."""
        if ks_stat == 0:
            return 1.0
        lam = (np.sqrt(n_eff) + 0.12 + 0.11 / np.sqrt(n_eff)) * ks_stat
        # Kolmogorov distribution approximation
        if lam < 0.2:
            return 1.0
        p = 2 * np.exp(-2 * lam * lam)
        return min(1.0, max(0.0, float(p)))
    
    # =========================================================================
    # Test 2: Chi-Squared Test on Character Frequencies
    # =========================================================================
    
    def chi_squared_frequency(self, sources: Dict) -> Dict[str, Any]:
        """
        Chi-squared test: Are character frequency distributions independent
        of the source?
        
        If same compositorial practice → similar character distributions.
        """
        results = {}
        source_names = list(sources.keys())
        
        for i in range(len(source_names)):
            for j in range(i+1, len(source_names)):
                s1, s2 = source_names[i], source_names[j]
                
                freq1 = Counter(sources[s1]['chars'])
                freq2 = Counter(sources[s2]['chars'])
                
                # Only characters present in both
                common = set(freq1.keys()) & set(freq2.keys())
                if len(common) < 3:
                    continue
                
                obs1 = [freq1[c] for c in sorted(common)]
                obs2 = [freq2[c] for c in sorted(common)]
                
                # Chi-squared test (manual)
                total1 = sum(obs1)
                total2 = sum(obs2)
                grand_total = total1 + total2
                
                chi2 = 0.0
                for o1, o2 in zip(obs1, obs2):
                    row_total = o1 + o2
                    e1 = row_total * total1 / grand_total
                    e2 = row_total * total2 / grand_total
                    if e1 > 0:
                        chi2 += (o1 - e1) ** 2 / e1
                    if e2 > 0:
                        chi2 += (o2 - e2) ** 2 / e2
                
                df = len(common) - 1
                # Approximate p-value using Wilson-Hilferty
                if df > 0:
                    z = ((chi2 / df) ** (1/3) - (1 - 2/(9*df))) / np.sqrt(2/(9*df))
                    # Standard normal CDF approximation
                    p = 0.5 * (1 + np.tanh(z * 0.7978845608))  # erfc approx
                    p = 1.0 - p  # Upper tail
                else:
                    p = 1.0
                
                pair = f"{s1} vs {s2}"
                results[pair] = {
                    'chi2': float(chi2),
                    'degrees_freedom': df,
                    'p_value': float(p),
                    'common_chars': len(common),
                    'verdict': 'SIMILAR' if p > 0.05 else 'DIFFERENT',
                }
        
        return results
    
    # =========================================================================
    # Test 3: Mann-Whitney U Test
    # =========================================================================
    
    def mann_whitney_dimensions(self, sources: Dict) -> Dict[str, Any]:
        """
        Mann-Whitney U test on character widths — non-parametric alternative
        to t-test. Good for non-normal distributions.
        """
        results = {}
        source_names = list(sources.keys())
        
        for i in range(len(source_names)):
            for j in range(i+1, len(source_names)):
                s1, s2 = source_names[i], source_names[j]
                
                w1 = np.array(sources[s1]['widths'], dtype=float)
                w2 = np.array(sources[s2]['widths'], dtype=float)
                
                if len(w1) < 5 or len(w2) < 5:
                    continue
                
                u_stat, p_val = self._mann_whitney(w1, w2)
                
                # Effect size (rank-biserial correlation)
                n1, n2 = len(w1), len(w2)
                r = 1 - (2 * u_stat) / (n1 * n2)
                
                pair = f"{s1} vs {s2}"
                results[pair] = {
                    'u_statistic': float(u_stat),
                    'p_value': float(p_val),
                    'effect_size': float(r),
                    'effect_magnitude': 'negligible' if abs(r) < 0.1 else (
                        'small' if abs(r) < 0.3 else (
                        'medium' if abs(r) < 0.5 else 'large'
                    )),
                    'n1': n1, 'n2': n2,
                }
        
        return results
    
    def _mann_whitney(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Compute Mann-Whitney U statistic and approximate p-value."""
        combined = np.concatenate([x, y])
        ranks = np.argsort(np.argsort(combined)) + 1  # 1-indexed ranks
        
        r1 = ranks[:len(x)]
        u1 = np.sum(r1) - len(x) * (len(x) + 1) / 2
        u2 = len(x) * len(y) - u1
        u = min(u1, u2)
        
        # Normal approximation for p-value
        mu = len(x) * len(y) / 2
        sigma = np.sqrt(len(x) * len(y) * (len(x) + len(y) + 1) / 12)
        
        if sigma == 0:
            return float(u), 1.0
        
        z = (u - mu) / sigma
        # Two-tailed p-value (normal CDF approximation)
        p = 2 * (1 - 0.5 * (1 + np.tanh(abs(z) * 0.7978845608)))
        
        return float(u), float(min(1.0, max(0.0, p)))
    
    # =========================================================================
    # Test 4: Bootstrap Confidence Intervals
    # =========================================================================
    
    def bootstrap_similarity(self, sources: Dict, n_bootstrap: int = 1000) -> Dict[str, Any]:
        """
        Bootstrap the cross-source similarity scores to get confidence intervals.
        """
        results = {}
        source_names = list(sources.keys())
        
        for i in range(len(source_names)):
            for j in range(i+1, len(source_names)):
                s1, s2 = source_names[i], source_names[j]
                
                w1 = np.array(sources[s1]['widths'], dtype=float)
                w2 = np.array(sources[s2]['widths'], dtype=float)
                
                if len(w1) < 10 or len(w2) < 10:
                    continue
                
                # Bootstrap dimension similarity
                boot_scores = []
                for _ in range(n_bootstrap):
                    sample1 = np.random.choice(w1, size=len(w1), replace=True)
                    sample2 = np.random.choice(w2, size=len(w2), replace=True)
                    
                    # Similarity = 1 - normalized mean difference
                    diff = abs(np.mean(sample1) - np.mean(sample2))
                    max_mean = max(np.mean(sample1), np.mean(sample2), 1)
                    sim = 1.0 - diff / max_mean
                    boot_scores.append(sim)
                
                boot_scores = np.array(boot_scores)
                
                pair = f"{s1} vs {s2}"
                results[pair] = {
                    'mean_similarity': float(np.mean(boot_scores)),
                    'ci_95_lower': float(np.percentile(boot_scores, 2.5)),
                    'ci_95_upper': float(np.percentile(boot_scores, 97.5)),
                    'ci_99_lower': float(np.percentile(boot_scores, 0.5)),
                    'ci_99_upper': float(np.percentile(boot_scores, 99.5)),
                    'std': float(np.std(boot_scores)),
                    'n_bootstrap': n_bootstrap,
                }
        
        return results
    
    # =========================================================================
    # Test 5: Type Measurement Analysis
    # =========================================================================
    
    def type_measurements(self, sources: Dict) -> Dict[str, Any]:
        """
        Compute typographic measurements per source:
        - Body height distribution
        - X-height estimation (from lowercase letters)
        - Set width patterns
        - Aspect ratio consistency
        """
        results = {}
        
        lowercase = set('abcdefghijklmnopqrstuvwxyz')
        uppercase = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
        for src_name, data in sources.items():
            instances = data['instances']
            
            # Filter by character category
            lc = [i for i in instances if i['character'] in lowercase and i['width'] > 3 and i['height'] > 3]
            uc = [i for i in instances if i['character'] in uppercase and i['width'] > 3 and i['height'] > 3]
            digits = [i for i in instances if i['character'].isdigit() and i['width'] > 3 and i['height'] > 3]
            all_valid = [i for i in instances if i['width'] > 3 and i['height'] > 3]
            
            if not all_valid:
                continue
            
            lc_heights = [i['height'] for i in lc] if lc else []
            uc_heights = [i['height'] for i in uc] if uc else []
            all_widths = [i['width'] for i in all_valid]
            all_heights = [i['height'] for i in all_valid]
            all_aspects = [i['width']/i['height'] for i in all_valid]
            
            measurements = {
                'total_chars': len(all_valid),
                'lowercase_count': len(lc),
                'uppercase_count': len(uc),
                'digit_count': len(digits),
                
                # Body height (all characters)
                'body_height_mean': float(np.mean(all_heights)),
                'body_height_median': float(np.median(all_heights)),
                'body_height_std': float(np.std(all_heights)),
                
                # X-height (lowercase only — approximation)
                'x_height_mean': float(np.mean(lc_heights)) if lc_heights else 0,
                'x_height_median': float(np.median(lc_heights)) if lc_heights else 0,
                'x_height_std': float(np.std(lc_heights)) if lc_heights else 0,
                
                # Cap height (uppercase)
                'cap_height_mean': float(np.mean(uc_heights)) if uc_heights else 0,
                'cap_height_std': float(np.std(uc_heights)) if uc_heights else 0,
                
                # Set widths
                'set_width_mean': float(np.mean(all_widths)),
                'set_width_median': float(np.median(all_widths)),
                'set_width_std': float(np.std(all_widths)),
                
                # Aspect ratio consistency
                'aspect_ratio_mean': float(np.mean(all_aspects)),
                'aspect_ratio_std': float(np.std(all_aspects)),
                
                # X-height / cap-height ratio (important type identity marker)
                'x_to_cap_ratio': float(np.mean(lc_heights) / np.mean(uc_heights)) if lc_heights and uc_heights else 0,
            }
            
            results[src_name] = measurements
        
        return results
    
    # =========================================================================
    # Run All Tests
    # =========================================================================
    
    def run_all(self, output_dir: Path) -> Dict[str, Any]:
        """Run all statistical tests and generate reports."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Gathering data from database...")
        sources = self.gather_data()
        
        for name, data in sources.items():
            logger.info(f"  {name}: {len(data['instances'])} characters")
        
        all_results = {}
        
        # Test 1: KS Test
        logger.info("Running Kolmogorov-Smirnov tests...")
        all_results['ks_test'] = self.ks_test_dimensions(sources)
        
        # Test 2: Chi-Squared
        logger.info("Running Chi-Squared frequency tests...")
        all_results['chi_squared'] = self.chi_squared_frequency(sources)
        
        # Test 3: Mann-Whitney
        logger.info("Running Mann-Whitney U tests...")
        all_results['mann_whitney'] = self.mann_whitney_dimensions(sources)
        
        # Test 4: Bootstrap
        logger.info("Running Bootstrap confidence intervals (1000 iterations)...")
        all_results['bootstrap'] = self.bootstrap_similarity(sources, n_bootstrap=1000)
        
        # Test 5: Type Measurements
        logger.info("Computing type measurements...")
        all_results['type_measurements'] = self.type_measurements(sources)
        
        # Save JSON
        json_path = output_dir / "formal_stats.json"
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Generate report
        self.generate_report(all_results, output_dir)
        
        # Print summary
        self.print_summary(all_results)
        
        return all_results
    
    def print_summary(self, results: Dict):
        """Print concise summary of all statistical tests."""
        print(f"\n{'='*75}")
        print(f"FORMAL STATISTICAL ANALYSIS — SUMMARY")
        print(f"{'='*75}")
        
        # KS Tests
        print(f"\n📊 Kolmogorov-Smirnov Test (same dimension distribution?)")
        for pair, data in results.get('ks_test', {}).items():
            short = pair.replace('_10057380', '').replace('_ppn777246686', '').replace('_178_1_theol_1s', '')
            print(f"  {short}:")
            print(f"    Width:  KS={data['ks_width']:.3f}, p={data['p_width']:.4f} → {data['verdict_width']}")
            print(f"    Height: KS={data['ks_height']:.3f}, p={data['p_height']:.4f} → {data['verdict_height']}")
        
        # Chi-Squared
        print(f"\n📊 Chi-Squared Test (same character frequency distribution?)")
        for pair, data in results.get('chi_squared', {}).items():
            short = pair.replace('_10057380', '').replace('_ppn777246686', '').replace('_178_1_theol_1s', '')
            print(f"  {short}: χ²={data['chi2']:.1f}, df={data['degrees_freedom']}, "
                  f"p={data['p_value']:.4f} → {data['verdict']}")
        
        # Mann-Whitney
        print(f"\n📊 Mann-Whitney U Test (non-parametric dimension comparison)")
        for pair, data in results.get('mann_whitney', {}).items():
            short = pair.replace('_10057380', '').replace('_ppn777246686', '').replace('_178_1_theol_1s', '')
            print(f"  {short}: U={data['u_statistic']:.0f}, p={data['p_value']:.4f}, "
                  f"effect={data['effect_magnitude']}")
        
        # Bootstrap
        print(f"\n📊 Bootstrap 95% Confidence Intervals")
        for pair, data in results.get('bootstrap', {}).items():
            short = pair.replace('_10057380', '').replace('_ppn777246686', '').replace('_178_1_theol_1s', '')
            print(f"  {short}: {data['mean_similarity']:.3f} "
                  f"[{data['ci_95_lower']:.3f}, {data['ci_95_upper']:.3f}]")
        
        # Type Measurements
        print(f"\n📊 Type Measurements")
        for src, data in results.get('type_measurements', {}).items():
            short = src.replace('_10057380', '').replace('_ppn777246686', '').replace('_178_1_theol_1s', '')
            print(f"  {short}:")
            print(f"    Body height: {data['body_height_mean']:.1f} ± {data['body_height_std']:.1f}px")
            print(f"    X-height:    {data['x_height_mean']:.1f} ± {data['x_height_std']:.1f}px")
            print(f"    Set width:   {data['set_width_mean']:.1f} ± {data['set_width_std']:.1f}px")
            if data['x_to_cap_ratio'] > 0:
                print(f"    x/cap ratio: {data['x_to_cap_ratio']:.3f}")
    
    def generate_report(self, results: Dict, output_dir: Path):
        """Generate HTML statistical report."""
        html = """<!DOCTYPE html>
<html>
<head>
<title>Formal Statistical Analysis — Print Attribution</title>
<style>
    body { font-family: 'Georgia', serif; background: #1a1a2e; color: #e0e0e0;
           margin: 2em; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 2em; }
    h1 { color: #c9a94e; border-bottom: 2px solid #c9a94e; padding-bottom: 0.5em; }
    h2 { color: #8b7355; margin-top: 2em; }
    h3 { color: #a0a0a0; }
    table { border-collapse: collapse; width: 100%; margin: 1em 0; }
    th, td { border: 1px solid #444; padding: 8px 12px; text-align: left; }
    th { background: #0f3460; }
    tr:nth-child(even) { background: #16213e; }
    .pass { color: #4caf50; font-weight: bold; }
    .fail { color: #f44336; font-weight: bold; }
    .warn { color: #ff9800; font-weight: bold; }
    .ci-bar { height: 20px; background: #2196f3; border-radius: 3px; display: inline-block; }
    .verdict-box { padding: 1.5em; margin: 1em 0; border: 2px solid; border-radius: 8px;
                   text-align: center; font-size: 1.2em; }
    .verdict-same { border-color: #4caf50; color: #4caf50; background: rgba(76,175,80,0.1); }
    .verdict-diff { border-color: #f44336; color: #f44336; background: rgba(244,67,54,0.1); }
    .verdict-unclear { border-color: #ff9800; color: #ff9800; background: rgba(255,152,0,0.1); }
    .method-note { font-size: 0.85em; color: #888; font-style: italic; margin: 0.5em 0; }
</style>
</head>
<body>
<h1>📐 Formal Statistical Analysis — Print Attribution Evidence</h1>
<p>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M") + """</p>
<p class="method-note">All tests performed at α = 0.05 significance level.
p > 0.05 means we cannot reject the null hypothesis that the samples come from the same distribution.</p>
"""
        # KS Test Section
        html += "<h2>1. Kolmogorov-Smirnov Test</h2>"
        html += '<p class="method-note">Tests whether character dimension distributions differ between sources. '
        html += 'If same type → same distributions → p > 0.05.</p>'
        html += "<table><tr><th>Pair</th><th>KS (width)</th><th>p (width)</th><th>KS (height)</th><th>p (height)</th><th>Width</th><th>Height</th></tr>"
        for pair, data in results.get('ks_test', {}).items():
            w_cls = 'pass' if data['verdict_width'] == 'SAME' else 'fail'
            h_cls = 'pass' if data['verdict_height'] == 'SAME' else 'fail'
            html += f'<tr><td>{pair}</td>'
            html += f'<td>{data["ks_width"]:.4f}</td><td>{data["p_width"]:.4f}</td>'
            html += f'<td>{data["ks_height"]:.4f}</td><td>{data["p_height"]:.4f}</td>'
            html += f'<td class="{w_cls}">{data["verdict_width"]}</td>'
            html += f'<td class="{h_cls}">{data["verdict_height"]}</td></tr>'
        html += "</table>"
        
        # Chi-squared
        html += "<h2>2. Chi-Squared Test (Character Frequencies)</h2>"
        html += '<p class="method-note">Tests independence of character frequency distributions. '
        html += 'Same compositor/text → similar frequencies → p > 0.05.</p>'
        html += "<table><tr><th>Pair</th><th>χ²</th><th>df</th><th>p-value</th><th>Common Chars</th><th>Verdict</th></tr>"
        for pair, data in results.get('chi_squared', {}).items():
            cls = 'pass' if data['verdict'] == 'SIMILAR' else 'fail'
            html += f'<tr><td>{pair}</td><td>{data["chi2"]:.2f}</td><td>{data["degrees_freedom"]}</td>'
            html += f'<td>{data["p_value"]:.4f}</td><td>{data["common_chars"]}</td>'
            html += f'<td class="{cls}">{data["verdict"]}</td></tr>'
        html += "</table>"
        
        # Mann-Whitney
        html += "<h2>3. Mann-Whitney U Test</h2>"
        html += '<p class="method-note">Non-parametric test for dimension differences. Effect size indicates practical significance.</p>'
        html += "<table><tr><th>Pair</th><th>U</th><th>p-value</th><th>Effect Size</th><th>Magnitude</th></tr>"
        for pair, data in results.get('mann_whitney', {}).items():
            html += f'<tr><td>{pair}</td><td>{data["u_statistic"]:.0f}</td>'
            html += f'<td>{data["p_value"]:.4f}</td><td>{data["effect_size"]:.3f}</td>'
            html += f'<td>{data["effect_magnitude"]}</td></tr>'
        html += "</table>"
        
        # Bootstrap
        html += "<h2>4. Bootstrap Confidence Intervals</h2>"
        html += '<p class="method-note">1000 bootstrap resamples of dimension similarity. 95% CI shown.</p>'
        html += "<table><tr><th>Pair</th><th>Mean</th><th>95% CI Lower</th><th>95% CI Upper</th><th>Std</th></tr>"
        for pair, data in results.get('bootstrap', {}).items():
            html += f'<tr><td>{pair}</td><td>{data["mean_similarity"]:.4f}</td>'
            html += f'<td>{data["ci_95_lower"]:.4f}</td><td>{data["ci_95_upper"]:.4f}</td>'
            html += f'<td>{data["std"]:.4f}</td></tr>'
        html += "</table>"
        
        # Type Measurements
        html += "<h2>5. Type Measurements</h2>"
        html += '<p class="method-note">Typographic dimensions in pixels. Note: different scan resolutions affect absolute values. '
        html += 'Ratios (x/cap) are more reliable for cross-source comparison.</p>'
        html += "<table><tr><th>Source</th><th>Chars</th><th>Body Height</th><th>X-Height</th>"
        html += "<th>Cap Height</th><th>Set Width</th><th>x/cap Ratio</th></tr>"
        for src, data in results.get('type_measurements', {}).items():
            html += f'<tr><td>{src}</td><td>{data["total_chars"]}</td>'
            html += f'<td>{data["body_height_mean"]:.1f} ± {data["body_height_std"]:.1f}</td>'
            html += f'<td>{data["x_height_mean"]:.1f} ± {data["x_height_std"]:.1f}</td>'
            html += f'<td>{data["cap_height_mean"]:.1f} ± {data["cap_height_std"]:.1f}</td>'
            html += f'<td>{data["set_width_mean"]:.1f} ± {data["set_width_std"]:.1f}</td>'
            ratio = data.get('x_to_cap_ratio', 0)
            html += f'<td>{ratio:.3f}</td></tr>' if ratio else '<td>—</td></tr>'
        html += "</table>"
        
        # Overall verdict
        html += "<h2>Overall Statistical Verdict</h2>"
        
        # Count passing tests
        pass_count = 0
        total_count = 0
        for test_name in ['ks_test', 'chi_squared', 'mann_whitney']:
            for pair, data in results.get(test_name, {}).items():
                total_count += 1
                if test_name == 'ks_test':
                    if data.get('verdict_width') == 'SAME' or data.get('verdict_height') == 'SAME':
                        pass_count += 1
                elif test_name == 'chi_squared':
                    if data.get('verdict') == 'SIMILAR':
                        pass_count += 1
                elif test_name == 'mann_whitney':
                    if data.get('effect_magnitude') in ('negligible', 'small'):
                        pass_count += 1
        
        ratio = pass_count / total_count if total_count > 0 else 0
        if ratio > 0.6:
            html += '<div class="verdict-box verdict-same">✅ Statistical evidence SUPPORTS common printer attribution<br>'
            html += f'{pass_count}/{total_count} tests consistent with same type</div>'
        elif ratio > 0.3:
            html += '<div class="verdict-box verdict-unclear">⚠️ Statistical evidence is MIXED<br>'
            html += f'{pass_count}/{total_count} tests consistent with same type</div>'
        else:
            html += '<div class="verdict-box verdict-diff">❌ Statistical evidence AGAINST common printer attribution<br>'
            html += f'{pass_count}/{total_count} tests consistent with same type</div>'
        
        html += "</body></html>"
        
        report_path = output_dir / "stats_report.html"
        with open(report_path, 'w') as f:
            f.write(html)
        logger.info(f"HTML report: {report_path}")


def main():
    output_dir = Path("reports/statistical_analysis")
    stats = ForensicStatistics()
    stats.run_all(output_dir)


if __name__ == "__main__":
    main()
