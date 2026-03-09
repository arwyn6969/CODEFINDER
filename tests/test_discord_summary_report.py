import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from app.services.shareable_report_content import SOURCE_METADATA
from scripts.generate_discord_summary import build_report_data, generate_discord_summary


class TestDiscordSummaryReport(unittest.TestCase):
    def test_build_report_data_uses_corrected_dates_and_counts(self):
        report_data = build_report_data()

        self.assertEqual(report_data.total_pages, 1237)
        self.assertEqual(report_data.total_chars, 27713)
        self.assertEqual(len(report_data.greenman_rows), 1)
        self.assertEqual(report_data.greenman_rows[0].source_name, "gdz_goettingen_ppn777246686")
        self.assertEqual(report_data.greenman_rows[0].page, 9)

        self.assertEqual(SOURCE_METADATA["gdz_goettingen_ppn777246686"].date, 1609)
        self.assertEqual(SOURCE_METADATA["google_books_tractatus_brevis"].date, 1613)
        self.assertEqual(SOURCE_METADATA["bsb_munich_10057380"].date, 1616)
        self.assertEqual(SOURCE_METADATA["hab_wolfenbuettel_178_1_theol_1s"].date, 1616)

        pair_scores = {row.pair_key: round(row.avg_similarity, 3) for row in report_data.pairwise_rows}
        self.assertEqual(pair_scores["bsb_munich_10057380 vs gdz_goettingen_ppn777246686"], 0.688)
        self.assertEqual(pair_scores["bsb_munich_10057380 vs google_books_tractatus_brevis"], 0.663)
        self.assertEqual(pair_scores["bsb_munich_10057380 vs hab_wolfenbuettel_178_1_theol_1s"], 0.709)
        self.assertEqual(pair_scores["gdz_goettingen_ppn777246686 vs google_books_tractatus_brevis"], 0.621)
        self.assertEqual(pair_scores["gdz_goettingen_ppn777246686 vs hab_wolfenbuettel_178_1_theol_1s"], 0.669)
        self.assertEqual(pair_scores["google_books_tractatus_brevis vs hab_wolfenbuettel_178_1_theol_1s"], 0.660)

        self.assertTrue(report_data.all_ks_different)
        self.assertTrue(report_data.all_chi_different)

    def test_generated_pdf_text_stays_cautious_and_standalone(self):
        pdftotext = shutil.which("pdftotext")
        if pdftotext is None:
            self.fail("pdftotext is required for this test")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "discord_summary.pdf"
            generate_discord_summary(output_path=output_path)

            text_path = Path(tmpdir) / "discord_summary.txt"
            subprocess.run(
                [pdftotext, "-layout", str(output_path), str(text_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            text = text_path.read_text()
            normalized = " ".join(text.split())

        self.assertIn("Standalone Kempten packet", normalized)
        self.assertIn("standalone overview of the German/Kempten corpus", normalized)
        self.assertIn("needs manual validation", normalized)
        self.assertIn("Damage chronology is excluded from the verdict", normalized)

        self.assertIn("1,237", normalized)
        self.assertIn("27,713", normalized)
        self.assertIn("13,215", normalized)

        self.assertIn("1609", normalized)
        self.assertIn("1613", normalized)
        self.assertIn("1616", normalized)
        self.assertNotIn("~1609", normalized)
        self.assertNotIn("~1610", normalized)
        self.assertNotIn("1614", normalized)

        lowered = normalized.lower()
        for phrase in (
            "proved",
            "same physical sorts",
            "same woodblock across all sources",
            "three independent lines converge",
            "chronological wear established",
            "shakespeare",
            "wright",
            "aspley",
        ):
            self.assertNotIn(phrase, lowered)


if __name__ == "__main__":
    unittest.main()
