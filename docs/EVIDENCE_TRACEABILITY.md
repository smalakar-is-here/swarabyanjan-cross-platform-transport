# Evidence Traceability — Public-Safe Core

| Evidence / claim family | Public aggregate evidence | Scientific execution / provenance | External restricted dependency where applicable |
|---|---|---|---|
| Stage01 source identity | `provenance/canonical/stage01/` | `notebooks/01-dataset-acquisition-and-freeze.ipynb` | raw third-party datasets are external |
| Stage02 integrity/leakage | Stage02 CSV/hash reports | `notebooks/02-p0-integrity-provenance-leakage-audit.ipynb`; checksum manifests | text-bearing P0 JSON reports excluded and hash-recorded |
| Stage03 clean/model development | metadata-only clean-row/split manifests; config locks | `notebooks/03-primary-model-development.ipynb`; Stage03 manifests | clean text Parquets + 10 `.joblib` binaries excluded |
| Stage04 calibration | `results/primary/stage04/`; `data/derived/primary/stage04/DEV_PREDICTIONS.parquet` | `notebooks/04-calibration.ipynb`; Stage04 provenance | upstream clean text/model bytes external |
| H1/H2 | `results/primary/stage05/H1_RESULTS.json`; `H2_RESULTS.json`; Stage08 tables/figures | Stage05 manifests/checksums | Stage05 text-bearing prediction caches excluded |
| H3 | `results/primary/stage05/H3_RESULTS.json`; H3 CSVs; Stage08 Table/Figure 3 | Stage05 manifests/checksums | Stage05 text-bearing prediction caches excluded |
| H3-P | `H3_PERTURBATION_RESULTS.json`; exclusions CSV; Stage08 supplementary artifacts | Stage05 manifests/checksums | `H3_PERTURBATION_PAIRS.parquet` excluded |
| H4 | `results/primary/stage06/` and Stage08 Table/Figure 4 | Stage06 reproducibility manifest/checksums | Stage05 prediction caches excluded |
| H3→H4 bridge | `results/primary/stage07/` and Stage08 Figure 5 | Stage07 reproducibility manifest/checksums | Stage05 prediction caches excluded |
| Source-site clustering sensitivity | `results/primary/stage09/H1_H2_SOURCE_CLUSTER_SENSITIVITY.json`, `H3_SOURCE_CLUSTER_SENSITIVITY.json`, `H3P_SOURCE_CLUSTER_SENSITIVITY.json`, `H4_SOURCE_CLUSTER_SENSITIVITY.json`, `CLAIM_CONCORDANCE_TABLE.csv` | Stage09 manifest/checksums | row-level source URL mapping/diagnostic artifacts excluded |
| Rule A | `results/sensitivity/rule_a/` | `experiments/sensitivity/rule_a/`; `provenance/sensitivity/rule_a/` | text-bearing branch caches excluded |
| Rule B | `results/sensitivity/rule_b/` | `experiments/sensitivity/rule_b/`; `provenance/sensitivity/rule_b/` | text-bearing branch caches excluded |
| Table S10 / S11 evidence | `results/sensitivity/comparison/`; `TABLE_S10_S11_TRACEABILITY.md` | Rule A/B Stage05–07 branch provenance | no missing aggregate evidence; current final LaTeX source remains outside core |
| Release engineering | `provenance/release/` | `scripts/release/10-release-package-assembly-vfinal.ipynb` | Stage10 full-release records may name external restricted artifacts by hash/path |

## Verification rule

When a manuscript-critical computation depends on an excluded artifact, the public repository provides the exact expected path and SHA-256 through `provenance/external_restricted/`. A researcher should restore only an authorised copy and verify its hash before executing the dependent stage.

No excluded artifact is treated as locally available merely because its identity is recorded.
