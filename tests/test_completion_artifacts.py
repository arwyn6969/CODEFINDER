import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
VALID_MANUAL_ASSESSMENTS = {
    "same design",
    "possible same sort/block",
    "inconclusive",
}


class TestCompletionArtifacts(unittest.TestCase):
    def test_manual_review_ledger_covers_all_german_pairs(self):
        ledger_path = ROOT / "reports" / "manual_review" / "manual_review_ledger.json"
        self.assertTrue(ledger_path.exists(), ledger_path)

        rows = json.loads(ledger_path.read_text())
        self.assertEqual(len(rows), 60)
        self.assertEqual(sum(bool(row["manual_assessment"]) for row in rows), 60)

        unique_pairs = {row["source_pair"] for row in rows}
        self.assertEqual(len(unique_pairs), 6)

        for row in rows:
            self.assertIn(row["manual_assessment"], VALID_MANUAL_ASSESSMENTS)
            if row["manual_assessment"] == "possible same sort/block":
                self.assertTrue(row["manual_notes"].strip())

    def test_manual_review_summary_matches_ledger_counts(self):
        ledger_path = ROOT / "reports" / "manual_review" / "manual_review_ledger.json"
        summary_path = ROOT / "reports" / "manual_review" / "manual_review_summary.md"

        self.assertTrue(summary_path.exists(), summary_path)
        rows = json.loads(ledger_path.read_text())
        summary = summary_path.read_text()

        self.assertIn("Reviewed rows: `60` / `60`", summary)
        for assessment in sorted(VALID_MANUAL_ASSESSMENTS):
            count = sum(row["manual_assessment"] == assessment for row in rows)
            self.assertIn(f"`{assessment}`: `{count}`", summary)

    def test_damage_report_stays_diagnostic_after_chronology_fix(self):
        damage_path = ROOT / "reports" / "damage_evolution" / "damage_evolution.json"
        self.assertTrue(damage_path.exists(), damage_path)

        payload = json.loads(damage_path.read_text())
        verdict = payload["verdict"]

        self.assertEqual(verdict["evidence_strength"], "INSUFFICIENT")
        self.assertEqual(verdict["increasing_damage"], 1)
        self.assertEqual(verdict["total_metrics"], 5)

    def test_negative_control_memo_records_internal_stress_test(self):
        memo_path = ROOT / "reports" / "negative_control" / "negative_control_memo.md"
        self.assertTrue(memo_path.exists(), memo_path)

        memo = memo_path.read_text()
        self.assertIn("Control source: `folger_iiif_aspley`", memo)
        self.assertIn("Greenman matches in control scan: `0`", memo)
        self.assertIn("internal stress test", memo)
        self.assertIn("publication-grade negative control", memo)

    def test_negative_control_search_log_records_rejected_candidates(self):
        search_log = ROOT / "reports" / "negative_control" / "negative_control_search_log.md"
        self.assertTrue(search_log.exists(), search_log)
        text = search_log.read_text()
        self.assertIn("negative_control_bsb10222478", text)
        self.assertIn("negative_control_bsb10326315", text)
        self.assertIn("negative_control_bsb00090503", text)
        self.assertIn("no publication-grade negative control has been accepted yet", text)
        self.assertIn("sort_metric_calibration.md", text)
        self.assertIn("metric or threshold insufficiency remains", text)

    def test_negative_control_candidates_are_namespaced_and_rejected(self):
        summaries = sorted(
            (ROOT / "reports" / "negative_control").glob("negative_control_*/negative_control_summary.json")
        )
        self.assertGreaterEqual(len(summaries), 3)

        rejected = []
        for summary_path in summaries:
            payload = json.loads(summary_path.read_text())
            self.assertIn("negative_control_", payload["control_source"])
            self.assertEqual(payload["metric_version"], "sort_metric_v1_0_30_40_30")
            self.assertEqual(payload["greenman_matches"], 0)
            self.assertTrue(payload["all_distributional_verdicts_different"])
            memo_path = summary_path.with_name("negative_control_memo.md")
            self.assertTrue(memo_path.exists(), memo_path)
            if not payload.get("publication_grade_accepted"):
                rejected.append((summary_path, payload))
                self.assertGreaterEqual(payload["median_pair_average"], 0.60)
                self.assertGreater(payload["max_pair_average"], 0.65)
                self.assertIn("Publication-grade status: `REJECTED`", memo_path.read_text())

        self.assertGreaterEqual(len(rejected), 3)

    def test_greenman_review_sheet_exists(self):
        sheet_path = ROOT / "reports" / "manual_review" / "greenman_review_sheet.png"
        self.assertTrue(sheet_path.exists(), sheet_path)

    def test_sort_metric_calibration_artifact_records_failure(self):
        calibration_path = ROOT / "reports" / "negative_control" / "sort_metric_calibration.json"
        fallback_path = ROOT / "reports" / "negative_control" / "fallback_source_search.md"
        self.assertTrue(calibration_path.exists(), calibration_path)
        self.assertTrue(fallback_path.exists(), fallback_path)

        payload = json.loads(calibration_path.read_text())
        self.assertEqual(payload["grid_size"], 231)
        self.assertEqual(payload["best_external_pass_count"], 0)
        self.assertIsNone(payload["chosen_formula"])
        self.assertEqual(payload["baseline"]["metric_version"], "sort_metric_v1_0_30_40_30")
        self.assertIn("PPN832871516", fallback_path.read_text())

    def test_final_summary_uses_similar_forms_language(self):
        summary_path = ROOT / "reports" / "final_report" / "summary.txt"
        self.assertTrue(summary_path.exists(), summary_path)

        text = summary_path.read_text()
        self.assertIn("SIMILAR FORMS", text)
        self.assertIn("FOLIATE-HEAD ORNAMENT ANALYSIS", text)
        self.assertNotIn("GREENMAN WOODBLOCK ANALYSIS", text)
        self.assertNotIn("  SAME\n", text)

    def test_final_report_artifacts_keep_cautious_claim_language(self):
        summary_text = (ROOT / "reports" / "final_report" / "summary.txt").read_text().lower()
        html_text = (ROOT / "reports" / "final_report" / "final_report.html").read_text().lower()
        corpus = "\n".join([summary_text, html_text])

        self.assertIn("publication-grade negative control", corpus)
        self.assertIn("source-specific", corpus)
        self.assertIn("not a settled attribution claim", summary_text)

        for phrase in (
            "demonstrate physical identity",
            "identical type produces identical character shapes",
            "same woodblock across all witnesses",
        ):
            self.assertNotIn(phrase, corpus)


if __name__ == "__main__":
    unittest.main()
