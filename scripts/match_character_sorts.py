#!/usr/bin/env python3
"""
Cross-Source Character Sort Matcher
====================================

Compares character sorts (individual letterforms) across multiple sources
to determine if they were printed using the same set of type.

Method:
1. For each character (e.g. 'A', 'e', 'S', etc.), gather all instances from all sources
2. Normalize crops to standard size
3. Compute damage fingerprints using BlockFingerprinter
4. Cluster instances within a character class
5. Compare cluster centroids across sources
6. If the same cluster appears in multiple sources → same type sort → same printer

Usage:
    python scripts/match_character_sorts.py

Output:
    reports/character_sort_match/
        sort_comparison.json
        sort_report.html
"""

import sys
import json
import cv2
import numpy as np
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from app.services.block_fingerprinter import BlockFingerprinter, BlockFingerprint

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("sort_matcher")


class CharacterSortMatcher:
    """Match character sorts across sources using fingerprint comparison."""
    
    # Characters of interest for early modern German/Latin type
    TARGET_CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    # Also check special chars common in these publications
    TARGET_CHARS += ['ſ', 'ß', 'ä', 'ö', 'ü', 'Ä', 'Ö', 'Ü']
    
    MIN_INSTANCES_PER_SOURCE = 3  # Need at least this many to compare
    MATCH_THRESHOLD = 0.55        # Fingerprint similarity threshold
    
    def __init__(self, db_path: str = "data/forensic.db"):
        self.db_path = db_path
        self.fingerprinter = BlockFingerprinter()
        
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def gather_character_data(self) -> Dict[str, Dict[str, List[Dict]]]:
        """
        Gather all character instances grouped by character then by source.
        Returns: {character: {source_name: [instances...]}}
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT ci.*, p.page_number, p.image_path as page_image, 
                   s.name as source_name,
                   si.image_path as crop_path
            FROM character_instances ci
            JOIN pages p ON ci.page_id = p.id
            JOIN sources s ON p.source_id = s.id
            LEFT JOIN sort_images si ON si.character_instance_id = ci.id
            ORDER BY ci.character, s.name
        """)
        
        data = defaultdict(lambda: defaultdict(list))
        for row in cursor.fetchall():
            char = row['character']
            source = row['source_name']
            data[char][source].append(dict(row))
        
        conn.close()
        
        # Filter to characters with data from multiple sources
        multi_source = {}
        for char, sources in data.items():
            if len(sources) >= 2:
                # Check minimum instances
                valid_sources = {s: insts for s, insts in sources.items() 
                               if len(insts) >= self.MIN_INSTANCES_PER_SOURCE}
                if len(valid_sources) >= 2:
                    multi_source[char] = valid_sources
        
        logger.info(f"Found {len(multi_source)} characters with instances in 2+ sources")
        return multi_source
    
    def fingerprint_character_instances(self, instances: List[Dict]) -> List[BlockFingerprint]:
        """Fingerprint a list of character instance crops."""
        fingerprints = []
        
        for inst in instances:
            crop_path = inst.get('crop_path')
            if not crop_path:
                continue
            
            full_path = Path("reports/crops") / crop_path
            if not full_path.exists():
                continue
            
            fp = self.fingerprinter.fingerprint(str(full_path), block_type="character")
            fingerprints.append(fp)
        
        return fingerprints
    
    def compute_source_centroid(self, fingerprints: List[BlockFingerprint]) -> np.ndarray:
        """Compute the average feature vector for a source's character instances."""
        if not fingerprints:
            return np.zeros(58)
        
        vectors = [np.array(fp.feature_vector) for fp in fingerprints if fp.feature_vector]
        if not vectors:
            return np.zeros(58)
        
        # Pad/truncate to same length
        max_len = max(len(v) for v in vectors)
        padded = []
        for v in vectors:
            if len(v) < max_len:
                v = np.concatenate([v, np.zeros(max_len - len(v))])
            padded.append(v[:max_len])
        
        return np.mean(padded, axis=0)
    
    def compare_sources_for_character(self, char: str, 
                                       source_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Compare character instances across sources.
        Returns comparison result dict.
        """
        source_fps = {}
        source_centroids = {}
        source_metrics = {}
        
        for source_name, instances in source_data.items():
            fps = self.fingerprint_character_instances(instances)
            source_fps[source_name] = fps
            source_centroids[source_name] = self.compute_source_centroid(fps)
            
            # Compute per-source metrics
            if fps:
                dims = [(inst['width'], inst['height']) for inst in instances 
                       if inst.get('width') and inst.get('height')]
                if dims:
                    widths = [d[0] for d in dims]
                    heights = [d[1] for d in dims]
                    source_metrics[source_name] = {
                        'count': len(instances),
                        'fingerprinted': len(fps),
                        'avg_width': np.mean(widths),
                        'std_width': np.std(widths),
                        'avg_height': np.mean(heights),
                        'std_height': np.std(heights),
                        'avg_aspect': np.mean([w/h for w,h in dims if h > 0]),
                        'avg_ink_density': np.mean([fp.ink_density for fp in fps]) if fps else 0,
                        'avg_edge_density': np.mean([fp.edge_density for fp in fps]) if fps else 0,
                        'avg_damage_count': np.mean([fp.damage_point_count for fp in fps]) if fps else 0,
                    }
        
        # Pairwise comparison between sources
        source_names = list(source_fps.keys())
        pairwise_scores = {}
        
        for i in range(len(source_names)):
            for j in range(i+1, len(source_names)):
                s1, s2 = source_names[i], source_names[j]
                
                # Compare centroids
                c1, c2 = source_centroids[s1], source_centroids[s2]
                if np.any(c1) and np.any(c2):
                    min_len = min(len(c1), len(c2))
                    cosine_sim = np.dot(c1[:min_len], c2[:min_len]) / (
                        np.linalg.norm(c1[:min_len]) * np.linalg.norm(c2[:min_len]) + 1e-10
                    )
                else:
                    cosine_sim = 0.0
                
                # Also do pairwise fingerprint comparison (sample)
                fp_scores = []
                fps1 = source_fps[s1][:10]  # Sample up to 10
                fps2 = source_fps[s2][:10]
                
                for fp1 in fps1:
                    for fp2 in fps2:
                        scores = self.fingerprinter.compare(fp1, fp2)
                        fp_scores.append(scores.get('aggregate', 0))
                
                avg_fp_score = np.mean(fp_scores) if fp_scores else 0
                max_fp_score = max(fp_scores) if fp_scores else 0
                
                # Dimension comparison
                m1 = source_metrics.get(s1, {})
                m2 = source_metrics.get(s2, {})
                
                dim_similarity = 0
                if m1 and m2:
                    # How similar are the average dimensions?
                    w_diff = abs(m1.get('avg_width',0) - m2.get('avg_width',0))
                    h_diff = abs(m1.get('avg_height',0) - m2.get('avg_height',0))
                    max_w = max(m1.get('avg_width',1), m2.get('avg_width',1))
                    max_h = max(m1.get('avg_height',1), m2.get('avg_height',1))
                    dim_similarity = 1.0 - (w_diff/max_w + h_diff/max_h) / 2
                
                pair_key = f"{s1} vs {s2}"
                pairwise_scores[pair_key] = {
                    'cosine_similarity': float(cosine_sim),
                    'avg_fingerprint_score': float(avg_fp_score),
                    'max_fingerprint_score': float(max_fp_score),
                    'dimension_similarity': float(dim_similarity),
                    'combined_score': float(
                        0.3 * cosine_sim + 0.4 * avg_fp_score + 0.3 * dim_similarity
                    ),
                }
        
        return {
            'character': char,
            'sources': {s: source_metrics.get(s, {}) for s in source_names},
            'pairwise': pairwise_scores,
        }
    
    def run_full_comparison(self, output_dir: Path):
        """Run character sort comparison across all sources."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Gather data
        char_data = self.gather_character_data()
        
        if not char_data:
            logger.error("No characters with multi-source data found!")
            logger.info("Make sure extract_characters.py has been run on multiple sources.")
            return
        
        results = []
        
        for char, source_data in sorted(char_data.items()):
            logger.info(f"Comparing '{char}' across {list(source_data.keys())}")
            result = self.compare_sources_for_character(char, source_data)
            results.append(result)
        
        # Save JSON
        json_path = output_dir / "sort_comparison.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved: {json_path}")
        
        # Generate HTML report
        self.generate_html_report(results, output_dir)
        
        # Summary
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results: List[Dict]):
        """Print summary of character sort comparison."""
        print(f"\n{'='*70}")
        print(f"CHARACTER SORT MATCHING — CROSS-SOURCE RESULTS")
        print(f"{'='*70}")
        print(f"Characters compared: {len(results)}")
        
        # Aggregate scores per pair
        pair_scores = defaultdict(list)
        for r in results:
            for pair, scores in r.get('pairwise', {}).items():
                pair_scores[pair].append(scores['combined_score'])
        
        print(f"\nPairwise similarity (averaged across all characters):")
        for pair, scores in sorted(pair_scores.items()):
            avg = np.mean(scores)
            std = np.std(scores)
            verdict = "✅ SAME TYPE" if avg > 0.6 else ("⚠️ UNCLEAR" if avg > 0.4 else "❌ DIFFERENT TYPE")
            print(f"  {pair}: {avg:.3f} ± {std:.3f}  {verdict}")
        
        # Top matching characters
        print(f"\nTop 10 most similar characters:")
        char_best = []
        for r in results:
            best_pair_score = max(
                (s['combined_score'] for s in r.get('pairwise', {}).values()), 
                default=0
            )
            char_best.append((r['character'], best_pair_score))
        
        for char, score in sorted(char_best, key=lambda x: x[1], reverse=True)[:10]:
            print(f"  '{char}': {score:.3f}")
    
    def generate_html_report(self, results: List[Dict], output_dir: Path):
        """Generate HTML report for character sort comparison."""
        html = """<!DOCTYPE html>
<html>
<head>
<title>Character Sort Matching — Cross-Source Report</title>
<style>
    body { font-family: 'Georgia', serif; background: #1a1a2e; color: #e0e0e0; 
           margin: 2em; line-height: 1.6; }
    h1 { color: #c9a94e; border-bottom: 2px solid #c9a94e; padding-bottom: 0.5em; }
    h2 { color: #8b7355; margin-top: 2em; }
    table { border-collapse: collapse; width: 100%; margin: 1em 0; }
    th, td { border: 1px solid #444; padding: 8px 12px; text-align: left; }
    th { background: #0f3460; color: #e0e0e0; }
    tr:nth-child(even) { background: #16213e; }
    .score-high { color: #4caf50; font-weight: bold; }
    .score-med { color: #ff9800; }
    .score-low { color: #f44336; }
    .verdict { font-size: 1.2em; padding: 1em; margin: 1em 0;
               border: 2px solid; border-radius: 8px; text-align: center; }
    .verdict.same { border-color: #4caf50; color: #4caf50; }
    .verdict.unclear { border-color: #ff9800; color: #ff9800; }
    .verdict.different { border-color: #f44336; color: #f44336; }
    .char-block { background: #16213e; padding: 0.5em 1em; border-radius: 4px;
                  display: inline-block; margin: 2px; font-size: 1.3em; }
</style>
</head>
<body>
<h1>🔤 Character Sort Matching — Cross-Source Forensic Report</h1>
<p>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M") + """</p>
"""
        
        # Aggregate pair scores
        pair_scores = defaultdict(list)
        for r in results:
            for pair, scores in r.get('pairwise', {}).items():
                pair_scores[pair].append(scores['combined_score'])
        
        # Verdict
        for pair, scores in pair_scores.items():
            avg = np.mean(scores)
            if avg > 0.6:
                html += f'<div class="verdict same">✅ {pair}: Average similarity {avg:.3f} — Evidence suggests SAME TYPE SET</div>'
            elif avg > 0.4:
                html += f'<div class="verdict unclear">⚠️ {pair}: Average similarity {avg:.3f} — INCONCLUSIVE</div>'
            else:
                html += f'<div class="verdict different">❌ {pair}: Average similarity {avg:.3f} — Evidence suggests DIFFERENT TYPE</div>'
        
        # Summary table
        html += "<h2>Pairwise Summary</h2>"
        html += "<table><tr><th>Pair</th><th>Avg Score</th><th>Std Dev</th><th>Characters Compared</th></tr>"
        for pair, scores in sorted(pair_scores.items()):
            avg = np.mean(scores)
            std = np.std(scores)
            sc = 'score-high' if avg > 0.6 else ('score-med' if avg > 0.4 else 'score-low')
            html += f'<tr><td>{pair}</td><td class="{sc}">{avg:.3f}</td><td>{std:.3f}</td><td>{len(scores)}</td></tr>'
        html += "</table>"
        
        # Per-character detail
        html += "<h2>Per-Character Analysis</h2>"
        html += "<table><tr><th>Char</th>"
        all_pairs = sorted(pair_scores.keys())
        for pair in all_pairs:
            html += f"<th>{pair}</th>"
        html += "<th>Dimensions</th></tr>"
        
        for r in sorted(results, key=lambda x: max(
            (s['combined_score'] for s in x.get('pairwise', {}).values()), default=0
        ), reverse=True):
            char = r['character']
            html += f'<tr><td class="char-block">{char}</td>'
            for pair in all_pairs:
                score = r.get('pairwise', {}).get(pair, {}).get('combined_score', 0)
                sc = 'score-high' if score > 0.6 else ('score-med' if score > 0.4 else 'score-low')
                html += f'<td class="{sc}">{score:.3f}</td>'
            
            # Dimensions
            dims = []
            for src, metrics in r.get('sources', {}).items():
                if metrics:
                    short = src.split('_')[0]
                    dims.append(f"{short}: {metrics.get('avg_width',0):.0f}×{metrics.get('avg_height',0):.0f}")
            html += f'<td>{", ".join(dims)}</td>'
            html += "</tr>"
        
        html += "</table>"
        html += "</body></html>"
        
        report_path = output_dir / "sort_report.html"
        with open(report_path, 'w') as f:
            f.write(html)
        logger.info(f"HTML report: {report_path}")


def main():
    output_dir = Path("reports/character_sort_match")
    matcher = CharacterSortMatcher()
    matcher.run_full_comparison(output_dir)


if __name__ == "__main__":
    main()
