# Data Store

`data/` is the managed local store for corpora, source configuration, and forensic databases.

## Current layout

- `data/sources/` - local source PDFs, extracted source directories, and source configuration
- `data/forensic.db` - active German/Kempten forensic database
- `data/codefinder.db` - legacy or secondary database context; do not use this as the German/Kempten source of truth

## Rules

- Keep large source PDFs local unless there is a deliberate reason to track them.
- Treat source metadata and database selection as part of the research contract.
- Add future negative-control corpora under `data/sources/` with clear labels so they are distinguishable from active corpora.
- Do not use `data/` as a general scratch area for temporary experiments.
