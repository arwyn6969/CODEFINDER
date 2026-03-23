# Root Legacy Scripts

This folder preserves one-off scripts that historically lived at repo root before the consolidation pass.

Rules:

- Treat these as historical snapshots, not active workflow entrypoints.
- Prefer promoting useful logic into `app/`, `scripts/research/`, or `scripts/maintenance/` instead of editing these directly.
- If a legacy script still needs to be runnable, update its imports explicitly rather than moving it back to repo root.
