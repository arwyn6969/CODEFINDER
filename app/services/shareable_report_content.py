"""
Canonical wording and corrected source metadata for the Discord-safe
German/Kempten summary PDF.

The wording in this module is aligned to:
- docs/BOOK_HISTORY_MANUSCRIPT.md
- docs/BOOK_HISTORY_METHODS_APPENDIX.md
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ShareableSourceMetadata:
    display_name: str
    short_title: str
    institution: str
    record_id: str
    date: int


SOURCE_ORDER = (
    "gdz_goettingen_ppn777246686",
    "google_books_tractatus_brevis",
    "bsb_munich_10057380",
    "hab_wolfenbuettel_178_1_theol_1s",
)

SOURCE_METADATA = {
    "gdz_goettingen_ppn777246686": ShareableSourceMetadata(
        display_name="GDZ Goettingen",
        short_title="Jacobus Ruelichius, Vita, Non Vita",
        institution="Goettinger Digitalisierungszentrum",
        record_id="PPN777246686",
        date=1609,
    ),
    "google_books_tractatus_brevis": ShareableSourceMetadata(
        display_name="Google Books (Tractatus)",
        short_title="Antonio degli Albizzi, Tractatus brevis",
        institution="Google Books",
        record_id="uThoAAAAcAAJ",
        date=1613,
    ),
    "bsb_munich_10057380": ShareableSourceMetadata(
        display_name="BSB Munich",
        short_title="Hans Sachs, Sehr Herrliche, Schoene und Warhaffte Gedicht. 4",
        institution="Bayerische Staatsbibliothek",
        record_id="bsb10057380 / 4 P.o.germ. 176 i-4",
        date=1616,
    ),
    "hab_wolfenbuettel_178_1_theol_1s": ShareableSourceMetadata(
        display_name="HAB Wolfenbuettel",
        short_title="Antonio Albizzi, Exercitationum theologicarum pars prima",
        institution="Herzog August Bibliothek",
        record_id="178-1-theol-1s",
        date=1616,
    ),
}

COVER_LABELS = {
    "Audience": "informal research sharing",
    "Status": "cautious summary, not final attribution",
    "Corpus": "Standalone Kempten packet",
}

SUMMARY_SCOPE = (
    "This summary is a standalone overview of the German/Kempten corpus and is "
    "intentionally narrower than the full forensic report."
)

HEADLINE_VERDICT = (
    "Provisional computational evidence consistent with shared or transferred "
    "printing materials."
)

VERDICT_CONTEXT = (
    "Character-form evidence is strongest in the current rerun. Ornament "
    "evidence is weaker and remains source-specific. The summary is designed "
    "to be cautious and still needs manual validation by specialist readers."
)

WHAT_WE_CAN_SAY = (
    "The cleaned rerun covers four German/Kempten-related sources and 1,237 pages / 27,713 characters.",
    "Character-form similarity is high across all six source pairs in the current comparison set.",
    "One foliate-head candidate is verified in GDZ only under the current thresholds.",
    "A completed 60-row manual-review ledger now supports a restrained reading of the sort evidence.",
    "The current evidence is consistent with shared or transferred printing materials.",
)

WHAT_WE_CANNOT_SAY = (
    "The current evidence does not establish reuse of identical physical objects.",
    "The current evidence does not establish cross-source woodblock reuse.",
    "No publication-grade negative control has been accepted yet; three BSB candidates were rejected, and a 0.05-step metric calibration sweep did not resolve the blocker.",
    "Damage chronology is excluded from the verdict because the corrected rerun remains diagnostic only.",
)

NEXT_STEPS = (
    "Have a specialist bibliographer review the completed 60-row manual ledger before any stronger claim is circulated.",
    "Keep the verified GDZ ornament paired with the below-threshold BSB control in the fixed foliate-head review sheet.",
    "Tighten or redesign the sort discriminator before widening source search further, because the three tested BSB controls and the calibration sweep still leave unrelated witnesses too close.",
    "Keep damage chronology diagnostic-only unless a later rerun changes that result materially.",
)

FORBIDDEN_PHRASES = (
    "proved",
    "same physical sorts",
    "same woodblock across all sources",
    "three independent lines converge",
    "chronological wear established",
)

REQUIRED_PHRASES = (
    "standalone overview",
    "German/Kempten corpus",
    "needs manual validation",
    "Damage chronology is excluded from the verdict",
)


def validate_shareable_text(text: str) -> None:
    """Reject forbidden phrasing and ensure the key guardrails stay visible."""
    lowered = text.lower()
    banned_hits = [phrase for phrase in FORBIDDEN_PHRASES if phrase.lower() in lowered]
    if banned_hits:
        raise ValueError(f"Forbidden shareable phrasing detected: {banned_hits}")

    missing = [phrase for phrase in REQUIRED_PHRASES if phrase not in text]
    if missing:
        raise ValueError(f"Required shareable phrasing missing: {missing}")
