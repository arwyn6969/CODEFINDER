import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from app.services.forensic_report_visuals import (
    CHART_FILES,
    build_bootstrap_rows,
    build_formal_test_matrix_rows,
    build_manual_review_rows,
    build_pairwise_similarity_rows,
    generate_report_charts,
)
from scripts.generate_pdf_report import build_pdf


BASE_DIR = Path(__file__).resolve().parents[1]
SORT_RESULTS = json.loads((BASE_DIR / "reports" / "character_sort_match" / "sort_comparison.json").read_text())
STATS_RESULTS = json.loads((BASE_DIR / "reports" / "statistical_analysis" / "formal_stats.json").read_text())
MANUAL_REVIEW_LEDGER = json.loads((BASE_DIR / "reports" / "manual_review" / "manual_review_ledger.json").read_text())


class TestForensicReportVisuals(unittest.TestCase):
    def test_pairwise_similarity_rows_match_current_scores(self):
        rows = build_pairwise_similarity_rows(SORT_RESULTS)

        self.assertEqual(len(rows), 6)
        score_map = {row["label"]: (round(row["avg_similarity"], 3), row["characters_compared"]) for row in rows}
        self.assertEqual(score_map["BSB-GDZ"], (0.688, 25))
        self.assertEqual(score_map["BSB-Google"], (0.663, 13))
        self.assertEqual(score_map["BSB-HAB"], (0.709, 33))
        self.assertEqual(score_map["GDZ-Google"], (0.621, 12))
        self.assertEqual(score_map["GDZ-HAB"], (0.669, 18))
        self.assertEqual(score_map["Google-HAB"], (0.660, 13))

    def test_manual_review_rows_match_current_totals(self):
        rows = build_manual_review_rows(MANUAL_REVIEW_LEDGER)

        self.assertEqual(len(rows), 6)
        self.assertEqual(sum(row["same_design"] for row in rows), 29)
        self.assertEqual(sum(row["possible_same_sort_block"] for row in rows), 2)
        self.assertEqual(sum(row["inconclusive"] for row in rows), 29)

        bsb_gdz = next(row for row in rows if row["label"] == "BSB-GDZ")
        self.assertEqual(bsb_gdz["same_design"], 7)
        self.assertEqual(bsb_gdz["possible_same_sort_block"], 2)
        self.assertEqual(bsb_gdz["inconclusive"], 1)

    def test_bootstrap_rows_cover_all_pairs_and_intervals(self):
        rows = build_bootstrap_rows(STATS_RESULTS)

        self.assertEqual(len(rows), 6)
        row_map = {row["label"]: row for row in rows}
        self.assertEqual(round(row_map["BSB-GDZ"]["mean_similarity"], 4), 0.9490)
        self.assertEqual(round(row_map["BSB-GDZ"]["ci_low"], 4), 0.9079)
        self.assertEqual(round(row_map["BSB-GDZ"]["ci_high"], 4), 0.9912)
        self.assertEqual(round(row_map["GDZ-HAB"]["mean_similarity"], 4), 0.7978)
        self.assertEqual(round(row_map["GDZ-HAB"]["ci_low"], 4), 0.7652)
        self.assertEqual(round(row_map["GDZ-HAB"]["ci_high"], 4), 0.8321)

    def test_formal_test_matrix_rows_match_current_verdicts(self):
        rows = build_formal_test_matrix_rows(STATS_RESULTS)

        self.assertEqual(len(rows), 6)
        for row in rows:
            self.assertEqual(row["ks_width_verdict"], "DIFFERENT")
            self.assertEqual(row["ks_height_verdict"], "DIFFERENT")
            self.assertEqual(row["chi_verdict"], "DIFFERENT")
            self.assertEqual(row["mw_effect_magnitude"], "negligible")

        bsb_gdz = next(row for row in rows if row["label"] == "BSB-GDZ")
        self.assertAlmostEqual(bsb_gdz["mw_effect_size"], 0.0457, places=4)

    def test_generate_report_charts_writes_all_expected_pngs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            outputs = generate_report_charts(
                SORT_RESULTS,
                STATS_RESULTS,
                MANUAL_REVIEW_LEDGER,
                output_dir,
            )

            self.assertEqual(set(outputs), set(CHART_FILES))
            for key, filename in CHART_FILES.items():
                self.assertEqual(outputs[key], output_dir / filename)
                self.assertTrue(outputs[key].exists())
                self.assertGreater(outputs[key].stat().st_size, 0)

    def test_forensic_pdf_contains_new_chart_suite_and_standalone_wording(self):
        pdftotext = shutil.which("pdftotext")
        if pdftotext is None:
            self.fail("pdftotext is required for this test")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "forensic_report.pdf"
            build_pdf(output_path=output_path)

            text_path = Path(tmpdir) / "forensic_report.txt"
            subprocess.run(
                [pdftotext, "-layout", str(output_path), str(text_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            normalized = " ".join(text_path.read_text().split())

        self.assertIn("Figure 5: Pairwise average character-form similarity by source pair", normalized)
        self.assertIn("Figure 6: Manual-review outcomes by source pair", normalized)
        self.assertIn("Figure 7: Bootstrap mean similarity with 95% confidence intervals", normalized)
        self.assertIn("Figure 8: Summary matrix of KS width, KS height, chi-squared, and Mann-Whitney readings by source pair.", normalized)
        self.assertIn("Table A3: Kolmogorov-Smirnov width comparison.", normalized)
        self.assertNotIn("Table 9: KS test on character dimensions.", normalized)

        lowered = normalized.lower()
        for phrase in ("shakespeare", "wright", "aspley"):
            self.assertNotIn(phrase, lowered)


if __name__ == "__main__":
    unittest.main()
