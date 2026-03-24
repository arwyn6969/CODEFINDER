#!/usr/bin/env python3
"""
Bootstrap the shared corpus registry from the local source configuration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from app.core.database import SessionLocal, init_db  # noqa: E402
from app.services.corpus_registry_service import CorpusRegistryService  # noqa: E402
from app.services.corpus_registry_types import CorpusLane  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap the corpus registry from local sources")
    parser.add_argument("--config", default="data/sources/config.yaml", help="Path to the source config YAML")
    parser.add_argument("--source", action="append", dest="source_keys", help="Specific source key(s) to ingest")
    parser.add_argument("--skip-candidates", action="store_true", help="Skip candidate generation after ingestion")
    args = parser.parse_args()

    init_db()
    db = SessionLocal()
    try:
        service = CorpusRegistryService(db)
        result = {
            "bootstrap": service.bootstrap_from_sources_config(
                config_path=args.config,
                source_keys=args.source_keys,
            )
        }
        if not args.skip_candidates:
            result["candidates"] = service.build_similarity_candidates(lane=CorpusLane.PRINTED_EUROPEAN)
        result["summary"] = service.summarize_registry()
        print(json.dumps(result, indent=2))
    finally:
        db.close()


if __name__ == "__main__":
    main()
