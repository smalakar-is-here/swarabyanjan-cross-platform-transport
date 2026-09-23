# Notebook Audit

Every notebook retained in the repository was inspected individually for Markdown structure, title/intro content, heading hierarchy, execution-state wording indicators, archive/path references, and environment-specific paths.

**Retained notebooks audited: 23**

Authoritative notebooks were preserved byte-for-byte. No code, output, execution count, scientific claim, or provenance field was changed. Where a notebook has limited Markdown or environment-specific paths, the issue is documented rather than silently altering the frozen artifact.

| Notebook | Markdown cells | H1/title | Execution state | Heading hierarchy | Package refs | Absolute paths | Action | Audit note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| experiments/sensitivity/rule_a/notebooks/executed/05_EXTERNAL_VALIDATION_H3_vFINAL_executed.ipynb | 10 | Stage 05 — External Validation and H3 (Canonical Final) | EXECUTED | PASS | 0 | 2 | Preserved unchanged | Environment-specific absolute path(s) present in code |
| experiments/sensitivity/rule_a/notebooks/executed/06_FIXED_BUDGET_RELIABILITY_EVALUATION_vFINAL_executed.ipynb | 7 | Stage 06 — Fixed-Budget Review and Reliability Evaluation | EXECUTED | PASS | 0 | 2 | Preserved unchanged | Environment-specific absolute path(s) present in code |
| experiments/sensitivity/rule_a/notebooks/executed/07_H3_H4_BRIDGE_EXPLORATORY_vFINAL_executed.ipynb | 2 | Stage 07 — H3→H4 Bridge (Exploratory / Secondary) — vFINAL | EXECUTED | PASS | 0 | 2 | Preserved unchanged | Environment-specific absolute path(s) present in code |
| experiments/sensitivity/rule_a/notebooks/source/05_EXTERNAL_VALIDATION_H3_vFINAL.ipynb | 10 | Stage 05 — External Validation and H3 (Canonical Final) | EXECUTED | PASS | 0 | 0 | Preserved unchanged | No human-facing issue requiring notebook modification |
| experiments/sensitivity/rule_a/notebooks/source/06_FIXED_BUDGET_RELIABILITY_EVALUATION_vFINAL.ipynb | 7 | Stage 06 — Fixed-Budget Review and Reliability Evaluation | EXECUTED | PASS | 0 | 2 | Preserved unchanged | Environment-specific absolute path(s) present in code |
| experiments/sensitivity/rule_a/notebooks/source/07_H3_H4_BRIDGE_EXPLORATORY_vFINAL.ipynb | 2 | Stage 07 — H3→H4 Bridge (Exploratory / Secondary) — vFINAL | EXECUTED | PASS | 0 | 2 | Preserved unchanged | Environment-specific absolute path(s) present in code |
| experiments/sensitivity/rule_b/notebooks/executed/07_H3_H4_BRIDGE_EXPLORATORY_vFINAL_executed.ipynb | 2 | Stage 07 — H3→H4 Bridge (Exploratory / Secondary) — vFINAL | EXECUTED | PASS | 0 | 2 | Preserved unchanged | Environment-specific absolute path(s) present in code |
| experiments/sensitivity/rule_b/notebooks/source/05_EXTERNAL_VALIDATION_H3_vFINAL.ipynb | 10 | Stage 05 — External Validation and H3 (Canonical Final) | EXECUTED | PASS | 0 | 2 | Preserved unchanged | Environment-specific absolute path(s) present in code |
| experiments/sensitivity/rule_b/notebooks/source/06_FIXED_BUDGET_RELIABILITY_EVALUATION_vFINAL.ipynb | 7 | Stage 06 — Fixed-Budget Review and Reliability Evaluation | EXECUTED | PASS | 0 | 0 | Preserved unchanged | No human-facing issue requiring notebook modification |
| experiments/sensitivity/rule_b/notebooks/source/07_H3_H4_BRIDGE_EXPLORATORY_vFINAL.ipynb | 2 | Stage 07 — H3→H4 Bridge (Exploratory / Secondary) — vFINAL | EXECUTED | PASS | 0 | 2 | Preserved unchanged | Environment-specific absolute path(s) present in code |
| notebooks/01-dataset-acquisition-and-freeze.ipynb | 10 | External Dataset Acquisition and Immutable Byte Freeze | EXECUTED | PASS | 1 | 1 | Preserved unchanged | Archive/package filename reference(s) present; Environment-specific absolute path(s) present in code |
| notebooks/02-p0-integrity-provenance-leakage-audit.ipynb | 12 | P0 Integrity, Provenance, and Leakage Audit | EXECUTED | PASS | 1 | 3 | Preserved unchanged | Archive/package filename reference(s) present; Environment-specific absolute path(s) present in code |
| notebooks/03-primary-model-development.ipynb | 8 | Stage 03 — Primary Model Development | EXECUTED | PASS | 1 | 2 | Preserved unchanged | Archive/package filename reference(s) present; Environment-specific absolute path(s) present in code |
| notebooks/04-calibration.ipynb | 7 | Stage 04 — Scalar Temperature Calibration | EXECUTED | PASS | 0 | 2 | Preserved unchanged | Environment-specific absolute path(s) present in code |
| notebooks/05-external-validation-and-h3-vfinal.ipynb | 10 | Stage 05 — External Validation and H3 (Canonical Final) | EXECUTED | PASS | 0 | 2 | Preserved unchanged | Environment-specific absolute path(s) present in code |
| notebooks/06-fixed-budget-reliability-evaluation-vfinal.ipynb | 7 | Stage 06 — Fixed-Budget Review and Reliability Evaluation | EXECUTED | PASS | 0 | 2 | Preserved unchanged | Environment-specific absolute path(s) present in code |
| notebooks/07-h3-h4-bridge-exploratory-vfinal.ipynb | 2 | Stage 07 — H3→H4 Bridge (Exploratory / Secondary) — vFINAL | EXECUTED | PASS | 0 | 2 | Preserved unchanged | Environment-specific absolute path(s) present in code |
| notebooks/08-manuscript-figures-and-tables-vfinal.ipynb | 0 | (none) | EXECUTED | PASS | 0 | 5 | Preserved unchanged | No Markdown cells; Environment-specific absolute path(s) present in code |
| notebooks/09-source-publisher-cluster-sensitivity-vfinal.ipynb | 1 | Stage 09 — Source-Site Host Cluster Sensitivity | EXECUTED | PASS | 0 | 3 | Preserved unchanged | Environment-specific absolute path(s) present in code |
| provenance/archive/notebooks/05-external-validation.ipynb | 11 | Stage 05 — Locked Source Test and External Validation | EXECUTED | PASS | 0 | 2 | Preserved unchanged | Environment-specific absolute path(s) present in code |
| provenance/archive/notebooks/05-h3-statistical-repair-only.ipynb | 1 | Stage 05 — H3 Statistical Repair Only | EXECUTED | PASS | 0 | 2 | Preserved unchanged | Environment-specific absolute path(s) present in code |
| provenance/archive/notebooks/06-fixed-budget-reliability-evaluation.ipynb | 7 | Stage 06 — Fixed-Budget Review and Reliability Evaluation | EXECUTED | PASS | 0 | 2 | Preserved unchanged | Environment-specific absolute path(s) present in code |
| scripts/release/10-release-package-assembly-vfinal.ipynb | 15 | Stage 10 — Release Package Assembly vFINAL | EXECUTED | WARNING | 0 | 4 | Preserved unchanged | Heading-level jump detected; Environment-specific absolute path(s) present in code |

## Notable documentation finding

`notebooks/08-manuscript-figures-and-tables-vfinal.ipynb` contains no Markdown cells. Because it is an authoritative canonical source artifact, it remains unchanged. Its scientific role and input/output position are documented in `CANONICAL_NOTEBOOK_MAP.md`, `EVIDENCE_TRACEABILITY.md`, and `REPRODUCIBILITY.md`.

No cleaned scientific copy was created because doing so would introduce a second source identity.
