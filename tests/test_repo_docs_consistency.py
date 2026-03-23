import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text()


class TestRepoDocsConsistency(unittest.TestCase):
    def test_runtime_contract_is_consistent_in_core_docs(self):
        readme = read_text("README.md")
        contract = read_text("docs/REPO_CONTRACT.md")
        architecture = read_text("docs/architecture.md")
        retention_lock = read_text("docs/CONSOLIDATION_RETENTION_LOCK.md")

        for text in (readme, contract, architecture, retention_lock):
            self.assertIn("app.api.main:app", text)
        for text in (readme, contract, architecture):
            self.assertIn("/api/*", text)

    def test_german_lane_remains_the_active_priority(self):
        readme = read_text("README.md")
        handoff = read_text("docs/DEVELOPER_HANDOFF.md")
        architecture = read_text("docs/architecture.md")

        self.assertIn("| German/Kempten research | Active priority |", readme)
        self.assertIn("German/Kempten lane: still the primary research lane", handoff)
        self.assertIn("| German/Kempten research | Active priority |", architecture)

    def test_shakespeare_lane_docs_agree_on_canonical_package_status(self):
        readme = read_text("README.md")
        contract = read_text("docs/REPO_CONTRACT.md")
        architecture = read_text("docs/architecture.md")
        handoff = read_text("docs/DEVELOPER_HANDOFF.md")
        internal_summary = read_text("docs/SHAKESPEARE_INTERNAL_SUMMARY.md")

        self.assertIn("| Shakespeare research | Secondary canonical |", readme)
        self.assertIn("Shakespeare canonical research", contract)
        self.assertIn("| Shakespeare research | Secondary canonical |", architecture)
        self.assertIn("secondary canonical research lane", handoff.lower())
        self.assertIn("canonical package", architecture.lower())
        self.assertIn("canonical package", internal_summary.lower())

    def test_internal_legacy_lane_is_documented_consistently(self):
        readme = read_text("README.md")
        repo_index = read_text("docs/REPO_INDEX.md")
        contract = read_text("docs/REPO_CONTRACT.md")
        architecture = read_text("docs/architecture.md")
        roadmap = read_text("docs/CODEFINDER_ROADMAP.md")

        self.assertIn("| Legacy exploratory tools | Internal legacy |", readme)
        self.assertIn("Internal legacy exploratory lane", repo_index)
        self.assertIn("Internal legacy exploratory tools", contract)
        self.assertIn("| Legacy exploratory tools | Internal legacy |", architecture)
        self.assertIn("internal legacy lane", roadmap.lower())


if __name__ == "__main__":
    unittest.main()
