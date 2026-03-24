"""
Shared corpus-registry enums and normalization helpers.
"""

from __future__ import annotations

from enum import Enum
import re
import unicodedata


class CorpusLane(str, Enum):
    PRINTED_EUROPEAN = "printed_european"
    MANUSCRIPT_ANCIENT = "manuscript_ancient"


class PrintMaterialType(str, Enum):
    PRINTED = "printed"
    MANUSCRIPT = "manuscript"


class OCRProfile(str, Enum):
    FRAKTUR_BLACKLETTER = "fraktur_blackletter"
    ROMAN_ITALIC = "roman_italic"
    MIXED_LATIN_VERNACULAR = "mixed_latin_vernacular"
    MANUSCRIPT_LAYOUT_REQUIRED = "manuscript_layout_required"


class CandidateType(str, Enum):
    CLUSTER_CANDIDATE = "cluster_candidate"
    FORENSIC_CANDIDATE = "forensic_candidate"


class ReviewVocabulary(str, Enum):
    SAME_DESIGN = "same design"
    POSSIBLE_SAME_SORT_BLOCK = "possible same sort/block"
    INCONCLUSIVE = "inconclusive"


def normalize_token(value: str | None) -> str:
    """Normalize text for stable matching and deduplication."""
    if not value:
        return ""

    normalized = unicodedata.normalize("NFKD", value)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def normalize_printer_name(value: str | None) -> str:
    """
    Normalize common early-modern printer spelling variants conservatively.
    """
    token = normalize_token(value)
    if not token:
        return ""

    replacements = {
        "christoff": "christoph",
        "christophoro": "christoph",
        "christophorus": "christoph",
        "krause": "kraus",
        "krausen": "kraus",
        "krauss": "kraus",
        "johann": "john",
        "joannes": "john",
        "georg": "george",
    }

    words = []
    for part in token.split():
        words.append(replacements.get(part, part))
    return " ".join(words)


def normalize_place(value: str | None) -> str:
    token = normalize_token(value)
    if not token:
        return ""

    replacements = {
        "campidoni": "kempten",
        "augspurg": "augsburg",
        "wolfenbuttel": "wolfenbuettel",
    }
    return replacements.get(token, token)
