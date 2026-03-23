#!/usr/bin/env python3
"""
Final Forensic Report (Task 7)

Compiles all Phase 6 findings into a comprehensive HTML report.
"""

from pathlib import Path

REPORT_DIR = Path("reports")
OUTPUT_PATH = REPORT_DIR / "forensic_report_phase6.html"

def generate_report():
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>CODEFINDER Phase 6: Deep Forensic Analysis Report</title>
    <style>
        body { font-family: 'Georgia', serif; background: #fafafa; color: #333; max-width: 900px; margin: 0 auto; padding: 30px; }
        h1 { color: #1a1a1a; border-bottom: 3px solid #333; padding-bottom: 15px; }
        h2 { color: #444; margin-top: 40px; border-bottom: 1px solid #ccc; padding-bottom: 8px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background: #f0f0f0; }
        .pass { color: #2e7d32; font-weight: bold; }
        .fail { color: #c62828; font-weight: bold; }
        .highlight { background: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 20px 0; }
        .conclusion { background: #e8f5e9; padding: 20px; border-radius: 8px; margin: 30px 0; }
        .conclusion.negative { background: #ffebee; }
        code { background: #f5f5f5; padding: 2px 6px; border-radius: 3px; }
    </style>
</head>
<body>
    <h1>📊 CODEFINDER Phase 6: Deep Forensic Analysis</h1>
    <p><strong>Subject:</strong> 1609 Shakespeare Sonnets (Wright vs Aspley Editions)</p>
    <p><strong>Date:</strong> February 2026</p>
    <p><strong>Objective:</strong> Determine if the two editions are typographically identical (same print run) or distinct.</p>

    <h2>Executive Summary</h2>
    <div class="conclusion negative">
        <strong>VERDICT: The editions are NOT typographically identical.</strong><br><br>
        Multiple independent analyses confirm systematic size differences between the Wright and Aspley editions.
        The Aspley edition shows consistently larger character dimensions (47-69% larger depending on character class),
        while ink density remains comparable. This indicates the scans are from sources with different
        resolution/DPI, or the physical copies themselves differ in enlargement.
    </div>

    <h2>Hypothesis Test Results</h2>
    <table>
        <tr>
            <th>ID</th>
            <th>Hypothesis</th>
            <th>Threshold</th>
            <th>Observed</th>
            <th>Result</th>
        </tr>
        <tr>
            <td>H1</td>
            <td>Period sizes consistent WITHIN each edition</td>
            <td>σ < 2px</td>
            <td>Aspley σ=4.1px, Wright σ=3.9px</td>
            <td class="fail">FAIL ✗</td>
        </tr>
        <tr>
            <td>H2</td>
            <td>Period sizes identical ACROSS editions</td>
            <td>Mean diff < 1px</td>
            <td>7.55px (47% larger in Aspley)</td>
            <td class="fail">FAIL ✗</td>
        </tr>
        <tr>
            <td>H3</td>
            <td>95% of aligned chars within ±3px width</td>
            <td>≥95%</td>
            <td>8.6% (1,104 / 12,799)</td>
            <td class="fail">FAIL ✗</td>
        </tr>
        <tr>
            <td>H4</td>
            <td>Ink density difference < 5%</td>
            <td>< 5%</td>
            <td>1.3% (Aspley 26.6%, Wright 27.9%)</td>
            <td class="pass">PASS ✓</td>
        </tr>
    </table>

    <h2>Key Findings</h2>
    
    <h3>1. Period Analysis (Control Experiment)</h3>
    <p>Periods are small, round, and should be perfectly consistent.</p>
    <ul>
        <li><strong>Aspley:</strong> Mean diameter 23.5px (σ = 4.1px)</li>
        <li><strong>Wright:</strong> Mean diameter 15.95px (σ = 3.9px)</li>
        <li><strong>Ratio:</strong> Aspley periods are <strong>47% larger</strong></li>
    </ul>
    <p>After filtering OCR noise (5-30px range), we analyzed 218 (Aspley) and 238 (Wright) periods.</p>

    <h3>2. Character Geometry Census</h3>
    <p>Analyzed 71 characters present in both editions with ≥10 instances each.</p>
    <div class="highlight">
        <strong>Width Ratio (Aspley / Wright):</strong><br>
        Mean: 1.69 | Median: 1.67 | Range: 1.11 - 2.19
    </div>
    <p>The ratio is NOT uniform across characters:</p>
    <ul>
        <li>Capital letters (G, E, R): up to <strong>2.19x larger</strong></li>
        <li>Lowercase/punctuation: ~1.5x larger</li>
        <li>Pipe character (|): only 1.11x larger</li>
    </ul>
    <p><strong>Implication:</strong> This non-uniformity suggests different type sizes or fonts, not just scan DPI.</p>

    <h3>3. Ink Density Measurement</h3>
    <p>Sampled 500 characters per edition and measured "percent black" within bounding boxes.</p>
    <ul>
        <li><strong>Aspley:</strong> 26.6% mean ink density</li>
        <li><strong>Wright:</strong> 27.9% mean ink density</li>
        <li><strong>Difference:</strong> 1.3% (within 5% threshold)</li>
    </ul>
    <p><strong>Implication:</strong> Despite different sizes, ink coverage is essentially identical. Both were printed with similar technique.</p>

    <h3>4. Aligned Character Comparison</h3>
    <p>Using text-anchored alignment with spatial veto, we matched <strong>12,799 character pairs</strong> across all pages.</p>
    <ul>
        <li><strong>Mean Width Difference:</strong> +4.57px (Aspley larger)</li>
        <li><strong>Mean Height Difference:</strong> +9.40px (Aspley taller)</li>
        <li><strong>Within ±3px:</strong> Only 8.6% of pairs</li>
    </ul>
    <p>This confirms the size difference is systematic across the entire corpus.</p>

    <h2>Forensic Conclusion</h2>
    <div class="conclusion negative">
        <strong>The Wright and Aspley editions cannot be confirmed as typographically identical.</strong><br><br>
        <strong>Evidence of Difference:</strong>
        <ul>
            <li>Systematic size differences (47-69% depending on character class)</li>
            <li>Non-uniform scaling ratios (capitals vs lowercase)</li>
            <li>Only 8.6% of aligned characters within ±3px width tolerance</li>
        </ul>
        <strong>Evidence of Similarity:</strong>
        <ul>
            <li>Ink density is nearly identical (1.3% difference)</li>
            <li>Text content alignment is achievable (12,799 pairs matched)</li>
        </ul>
        <strong>Possible Explanations:</strong>
        <ol>
            <li>Scans are from copies with different DPI/enlargement settings</li>
            <li>Physical copies themselves differ (e.g., one is a facsimile)</li>
            <li>Genuine typographical differences in the original printing</li>
        </ol>
        <strong>Recommendation:</strong> Verify scan metadata and source provenance before drawing historical conclusions.
    </div>

    <h2>Data Files</h2>
    <ul>
        <li><code>reports/period_analysis/period_instances.csv</code> - All period measurements</li>
        <li><code>reports/geometry_census/character_geometry.csv</code> - Character size ratios</li>
        <li><code>reports/ink_density/ink_density_summary.txt</code> - Ink analysis summary</li>
        <li><code>reports/aligned_comparison/aligned_pairs.csv</code> - 12,799 matched pairs</li>
    </ul>

    <p style="margin-top: 50px; color: #888; font-size: 0.9em;">
        Report generated by CODEFINDER Forensic Analysis Pipeline<br>
        Phase 6: Deep Forensic Analysis
    </p>
</body>
</html>
"""
    
    with open(OUTPUT_PATH, 'w') as f:
        f.write(html)
    
    print(f"Report generated: {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_report()
