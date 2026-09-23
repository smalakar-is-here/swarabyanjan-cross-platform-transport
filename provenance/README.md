# Provenance

- `canonical/` — frozen scientific-stage manifests, checksums, locks, status records, audit records, environment records, and other provenance.
- `feasibility_audit/` — News-decontamination feasibility audit.
- `rowlevel_audit/` — decontamination input/audit records not duplicated elsewhere.
- `release/` — Stage10 release/package assembly manifests and validation records.
- `sensitivity/` — Rule A / Rule B execution provenance.
- `archive/` — historical/repair notebooks retained for scientific history.

Release/anonymized notebook copies are not treated as extra canonical notebooks.
- `external_restricted/` — exact paths, sizes, SHA-256 values, access routes and verification instructions for bytes intentionally excluded from the public package.
