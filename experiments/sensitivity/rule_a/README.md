# Rule A News-Decontamination Sensitivity

This directory contains the retained Rule A branch notebooks.

- Source branch notebooks: **3**
- Executed notebook captures available in the isolated package: **3**

Rule A is a **post-hoc sensitivity analysis only**. It does not replace, revise, or rank above the frozen Stage01–Stage09 primary analysis.

Scientific outputs are under `../../../results/sensitivity/rule_a/`.
Branch-specific derived caches are under `../../../data/derived/sensitivity/rule_a/`.
Execution provenance is under `../../../provenance/sensitivity/rule_a/`.

## Exact-primary duplicates intentionally not recopied

- `BAITBUSTER_HUMAN_PREDICTIONS.parquet` — exact SHA-256 match to frozen primary artifact; not recopied.
- `LOCKED_SOURCE_TEST_PREDICTIONS.parquet` — exact SHA-256 match to frozen primary artifact; not recopied.
- `H3_PERTURBATION_EXCLUSIONS.csv` — exact SHA-256 match to frozen primary artifact; not recopied.
- `H3_BAITBUSTER_REPLICATION_CORRECTED.csv` — exact SHA-256 match to frozen primary artifact; not recopied.

Those artifacts remain available from the frozen primary locations and are linked through SHA-256 identity.

Public-safe note: branch-specific text-bearing prediction and H3-P pair Parquets are excluded from this public package and hash-documented under `../../../provenance/external_restricted/`. Aggregate branch outputs remain unchanged.
