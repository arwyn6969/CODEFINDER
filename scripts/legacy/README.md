# Legacy Scripts

Use this lane for retired or superseded entrypoints that must be preserved for auditability.

Anything moved here should be treated as historical context, not as an active workflow. If a legacy script still contains useful logic, promote that logic into an active lane rather than extending the retired entrypoint.

Conventions:

- Use `scripts/legacy/root_legacy/` for historical one-off scripts migrated out of repo root during consolidation.
- Keep path updates explicit in docs/tests when a legacy script used to live at repo root.
