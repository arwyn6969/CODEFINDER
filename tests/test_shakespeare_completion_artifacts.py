import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REBUILD_PYTHON = (ROOT / ".venv" / "bin" / "python")
ALLOWED_STATUSES = {"included", "excluded_paratext", "unmatched"}
ALLOWED_PAIR_STATUSES = {"included", "excluded_paratext"}
ALLOWED_MANUAL_ASSESSMENTS = {
    "same design",
    "possible same sort/block",
    "inconclusive",
}
BANNED_PHRASES = (
    "identical type-blocks",
    "conclusive proof",
    "null hypothesis accepted",
    "h₀ (accepted)",
    "accepted): both editions were printed from the same type setting",
    "ocr quality alone explains all variance",
)


class TestShakespeareCompletionArtifacts(unittest.TestCase):
    def test_page_equivalence_manifest_classifies_each_witness_page_once(self):
        manifest_path = ROOT / "reports" / "shakespeare" / "page_equivalence_manifest.json"
        self.assertTrue(manifest_path.exists(), manifest_path)

        payload = json.loads(manifest_path.read_text())
        summary = payload["summary"]

        for witness in ("wright", "aspley"):
            entries = payload["witness_pages"][witness]
            self.assertEqual(len(entries), summary[witness]["total_images"])
            self.assertEqual(len({entry["sequence_number"] for entry in entries}), len(entries))

            counts = {"included": 0, "excluded_paratext": 0, "unmatched": 0}
            for entry in entries:
                self.assertIn(entry["status"], ALLOWED_STATUSES)
                counts[entry["status"]] += 1

            self.assertEqual(counts["included"], summary[witness]["included"])
            self.assertEqual(counts["excluded_paratext"], summary[witness]["excluded_paratext"])
            self.assertEqual(counts["unmatched"], summary[witness]["unmatched"])

        self.assertEqual(len(payload["pairs"]), summary["shared_sequence"]["paired_images"])
        self.assertEqual(
            sum(pair["pair_status"] == "included" for pair in payload["pairs"]),
            summary["shared_sequence"]["included_pairs"],
        )
        self.assertEqual(
            sum(pair["pair_status"] == "excluded_paratext" for pair in payload["pairs"]),
            summary["shared_sequence"]["paratext_pairs"],
        )
        for pair in payload["pairs"]:
            self.assertIn(pair["pair_status"], ALLOWED_PAIR_STATUSES)

    def test_docs_and_archive_summary_agree_on_key_counts(self):
        manifest_path = ROOT / "reports" / "shakespeare" / "page_equivalence_manifest.json"
        payload = json.loads(manifest_path.read_text())

        summary_text = (ROOT / "reports" / "shakespeare" / "summary.txt").read_text()
        manuscript = (ROOT / "docs" / "SHAKESPEARE_MANUSCRIPT.md").read_text()
        appendix = (ROOT / "docs" / "SHAKESPEARE_METHODS_APPENDIX.md").read_text()

        expected_snippets = (
            f"Wright local images: {payload['summary']['wright']['total_images']}",
            f"Aspley local images: {payload['summary']['aspley']['total_images']}",
            f"Shared folio-sequence images: {payload['summary']['shared_sequence']['paired_images']}",
            f"Included shared pairs: {payload['summary']['shared_sequence']['included_pairs']}",
            f"Unmatched Aspley extras: {payload['summary']['aspley']['unmatched']}",
        )
        for snippet in expected_snippets:
            self.assertIn(snippet, summary_text)

        for text in (manuscript, appendix):
            self.assertIn("53", text)
            self.assertIn("67", text)
            self.assertIn("41", text)
            self.assertIn("14", text)
            self.assertIn("38,909", text)
            self.assertIn("48,142", text)
            self.assertIn("external-safe", text.lower())

    def test_canonical_shakespeare_outputs_avoid_retired_claim_language(self):
        files = [
            ROOT / "docs" / "SHAKESPEARE_MANUSCRIPT.md",
            ROOT / "docs" / "SHAKESPEARE_METHODS_APPENDIX.md",
            ROOT / "docs" / "SHAKESPEARE_INTERNAL_SUMMARY.md",
            ROOT / "reports" / "shakespeare" / "summary.txt",
            ROOT / "reports" / "shakespeare" / "manual_review" / "manual_review_summary.md",
        ]

        corpus = "\n".join(path.read_text().lower() for path in files)
        for phrase in BANNED_PHRASES:
            self.assertNotIn(phrase, corpus)

    def test_manual_review_summary_matches_ledger(self):
        ledger_path = ROOT / "reports" / "shakespeare" / "manual_review" / "manual_review_ledger.json"
        summary_path = ROOT / "reports" / "shakespeare" / "manual_review" / "manual_review_summary.md"

        rows = json.loads(ledger_path.read_text())
        summary = summary_path.read_text()

        self.assertEqual(len(rows), 6)
        self.assertIn("Reviewed rows: `6` / `6`", summary)

        counts = {label: 0 for label in ALLOWED_MANUAL_ASSESSMENTS}
        for row in rows:
            self.assertIn(row["manual_assessment"], ALLOWED_MANUAL_ASSESSMENTS)
            counts[row["manual_assessment"]] += 1
            if row["manual_assessment"] == "possible same sort/block":
                self.assertTrue(row["manual_notes"].strip())

        for label, count in counts.items():
            self.assertIn(f"`{label}`: `{count}`", summary)

    def test_normalized_comparison_jsons_are_populated_and_explicit(self):
        whole_book = json.loads((ROOT / "reports" / "shakespeare" / "comparison" / "whole_book_comparison.json").read_text())
        high_conf = json.loads((ROOT / "reports" / "shakespeare" / "comparison" / "high_confidence_comparison.json").read_text())
        page_variance = json.loads((ROOT / "reports" / "shakespeare" / "comparison" / "page_variance_diagnostic.json").read_text())

        self.assertEqual(whole_book["scope"], "whole_witness")
        self.assertEqual(high_conf["scope"], "matched_high_confidence")
        self.assertEqual(page_variance["scope"], "page_matched_diagnostic")

        for payload in (whole_book, high_conf):
            self.assertIn("STC 22353a", payload["source1"]["name"])
            self.assertIn("STC 22353", payload["source2"]["name"])
            self.assertGreater(len(payload["top_character_deltas"]), 5)
            self.assertTrue(all(row["character"] for row in payload["top_character_deltas"]))
            self.assertNotEqual(payload["deltas"]["character_delta"], 0)

        self.assertEqual(page_variance["significant_pairs"], 40)
        self.assertGreaterEqual(len(page_variance["top_outliers"]), 5)
        self.assertIn("diagnostic", " ".join(page_variance["notes"]).lower())

    def test_rebuild_shakespeare_lane_reproduces_canonical_metrics(self):
        tracked_manifest = json.loads((ROOT / "reports" / "shakespeare" / "page_equivalence_manifest.json").read_text())
        tracked_package = json.loads((ROOT / "reports" / "shakespeare" / "package_manifest.json").read_text())

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            output_root = tmp_root / "reports" / "shakespeare"
            shareable_root = tmp_root / "reports" / "shareable"

            subprocess.run(
                [
                    str(REBUILD_PYTHON if REBUILD_PYTHON.exists() else Path(sys.executable)),
                    "scripts/maintenance/rebuild_shakespeare_lane.py",
                    "--output-root",
                    str(output_root),
                    "--shareable-root",
                    str(shareable_root),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            rebuilt_manifest = json.loads((output_root / "page_equivalence_manifest.json").read_text())
            rebuilt_package = json.loads((output_root / "package_manifest.json").read_text())

            self.assertEqual(rebuilt_manifest["summary"], tracked_manifest["summary"])
            self.assertEqual(rebuilt_package["primary_corpus"], tracked_package["primary_corpus"])
            self.assertEqual(rebuilt_package["key_metrics"], tracked_package["key_metrics"])
            self.assertTrue((shareable_root / "CODEFINDER_Shakespeare_Summary.pdf").exists())

    def test_shareable_pdf_text_stays_cautious_when_pdftotext_is_available(self):
        pdftotext = shutil.which("pdftotext")
        if pdftotext is None:
            self.skipTest("pdftotext is required for PDF text extraction")

        pdf_path = ROOT / "reports" / "shareable" / "CODEFINDER_Shakespeare_Summary.pdf"
        with tempfile.TemporaryDirectory() as tmpdir:
            text_path = Path(tmpdir) / "shakespeare_summary.txt"
            subprocess.run(
                [pdftotext, "-layout", str(pdf_path), str(text_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            text = " ".join(text_path.read_text().split()).lower()

        self.assertIn("standalone shakespeare packet", text)
        self.assertIn("provisional computational evidence", text)
        self.assertIn("source-equivalence constraints remain unresolved", text)
        for phrase in ("identical type", "conclusive proof", "null hypothesis accepted"):
            self.assertNotIn(phrase, text)


if __name__ == "__main__":
    unittest.main()
