# Reproducibility

## Scientific execution boundary

The canonical scientific workflow is Stage01–Stage09. Stage10 is release/package engineering only. The implementation is notebook-embedded; there is no fabricated standalone `src/` implementation.

## What the public repository preserves

The public package preserves:

1. all nine canonical scientific notebooks unchanged;
2. Stage10 release engineering unchanged;
3. frozen configuration/schema locks;
4. aggregate H1/H2/H3/H3-P/H4/bridge results;
5. publication figure/table source data and rendered figures available in the core;
6. Rule A / Rule B aggregate sensitivity outputs and comparison evidence;
7. stage manifests/checksums/status records and non-text-bearing provenance;
8. exact hashes and access/reproduction metadata for intentionally excluded restricted artifacts.

## What is intentionally external

A fresh clone does not contain every byte needed to rerun every stage. Public-redistribution-uncertain artifacts containing literal third-party text/row-level source bytes, plus the 10 unresolved-license Stage03 `.joblib` binaries, are excluded.

See `../provenance/external_restricted/RESTRICTED_ARTIFACT_MANIFEST.json` for the exact dependency list. Restore an authorised copy at the recorded expected path and require SHA-256 equality before executing a dependent notebook.

## Retained non-text-bearing derived interfaces

The repository retains metadata/numerical derived interfaces such as the Stage03 clean-row hash manifest, train/dev assignment manifest, and Stage04 DEV prediction table because the public-safety inspection verified that these files do not carry the excluded literal title fields. Their original scientific bytes are unchanged.

## From-scratch boundary

Full from-scratch reconstruction depends on obtaining the external datasets under their original source/access conditions, satisfying the Stage01/Stage02 frozen identity contracts, and reproducing any excluded derived/model artifacts. External URLs may change over time. This repository therefore does **not** claim one-command, fresh-clone reproducibility.

## Primary execution sequence

1. Stage01 acquisition/freeze
2. Stage02 integrity/provenance/leakage audit
3. Stage03 primary model development
4. Stage04 calibration
5. Stage05 external validation/H3
6. Stage06 fixed-budget reliability
7. Stage07 exploratory H3→H4 bridge
8. Stage08 deterministic manuscript figures/tables
9. Stage09 source-site clustering sensitivity

Stage10 assembles/verifies release artifacts only.

## Rule A / Rule B

Rule A and Rule B remain separate post-hoc News-decontamination sensitivity branches. Their aggregate results are public; their text-bearing branch caches are externally restricted and hash-documented rather than redistributed.
