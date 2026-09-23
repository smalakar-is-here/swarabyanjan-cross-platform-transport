# COMPLETE NOTEBOOK INVENTORY

## Counts

- Physical notebook occurrences in byte-inspected archives: **160**
- Unique whole-notebook SHA-256 contents: **38**
- Unique normalized cell-source contents (code + Markdown; outputs/metadata excluded): **21**
- Provenance-established canonical scientific notebooks: **9** (Stage01–Stage09)
- Release-engineering notebook: **1** (Stage10)
- Known canonical-history legacy/repair notebook contents: **3**
- Rule A/B sensitivity-only physical notebook occurrences: **28**
- Rule A/B sensitivity-only unique whole-notebook contents: **9**
- Stage10 release/anonymized/release-naming copy occurrences: **80**

The 160 physical occurrences are **not** a canonical count.

## Canonical scientific Stage01–Stage09 map from RELEASE_MANIFEST.json

| Stage | Canonical source name | Release name | Canonical source SHA-256 | Release whole-notebook SHA-256 | Execution |
| --- | --- | --- | --- | --- | --- |
| 01 | 01-dataset-acquisition-and-freeze.ipynb | 01-dataset-acquisition-and-freeze.ipynb | 009f92a04bfd38d5f4b521fbbec3306845bafa3281206736b16b49e8f1514196 | 0223500a7d1a58c1044d664e01c20e7e776fd4ceb58e8e8575181ac1944c0481 | EXECUTED |
| 02 | 02-p0-integrity-provenance-leakage-audit.ipynb | 02-p0-integrity-provenance-leakage-audit.ipynb | 2f248e31bdf6421a2c4fba6ac78d2300f1f479bd09da317d833f2f916179d68b | 77f5ad63c6c60dc637d15e2f7eec544e23138fb8f7d3b9e1c25e083f7f601d22 | EXECUTED |
| 03 | 03-primary-model-development.ipynb | 03-primary-model-development.ipynb | 260246bd57233e6a663ea5258a69e8a500b5b1b4a059381df51125b148a0271f | 9847f4b0d6f31afca35a698557f9bf3cf339f37b9c14386829960474f59645da | EXECUTED |
| 04 | 04-calibration.ipynb | 04-calibration.ipynb | d5f0c82ca353780e15cef5f27b8ae5fbcb3e840a18624ef1c7196b8e821e9908 | 64d23570c8df9c0559fce2b4194f40f3f298b9e9b2ca844644382f5e0bf5f4ef | EXECUTED |
| 05 | 05-external-validation-and-h3-vfinal.ipynb | 05_EXTERNAL_VALIDATION_H3_vFINAL.ipynb | 28f131f3004bd38a057a0f17f0a89e14fd6fbd13c5f5d17f1a540898736b869b | 9155f804884958c43b119d68b43e1b2d4763b5daa661e1e009039782ab9d4ad5 | EXECUTED |
| 06 | 06-fixed-budget-reliability-evaluation-vfinal.ipynb | 06_FIXED_BUDGET_RELIABILITY_EVALUATION_vFINAL.ipynb | 3edac9c63c3cb5711ee5a4e01afff89727929fd1629afab1256136335b3bb809 | 7d73b4a6a58ca0746deb90e6acee271cc1e261364a7192001eceafd89f416702 | EXECUTED |
| 07 | 07-h3-h4-bridge-exploratory-vfinal.ipynb | 07_H3_H4_BRIDGE_EXPLORATORY_vFINAL.ipynb | c5930a6a11ffdb49f169e75b8dff73f7fbdceed5eb5da160646200c5164efa87 | 967a7af8f994ab8273d58fd49aca561c2f4b3db232739e3501db9ab8036e0fcb | EXECUTED |
| 08 | 08-manuscript-figures-and-tables-vfinal.ipynb | 08_MANUSCRIPT_FIGURES_AND_TABLES_vFINAL.ipynb | 038699569e4562388ebd77df3a8cb6069d0d0a37040b839222a21cde8b795042 | 8e779f1388d143c26a004e5cdd4985e9462241909e33f8723dfcfa1e82fd8095 | EXECUTED |
| 09 | 09-source-publisher-cluster-sensitivity-vfinal.ipynb | 09_SOURCE_SITE_CLUSTER_SENSITIVITY_vFINAL.ipynb | e214251831c1dde32d60ffe0e06196e331e05ce9d331747861fc058751b39a65 | 346ce289c38410c964dd8779661eea2eba82785c4c7b591fc91a52b6da2ea09b | EXECUTED |

The manifest states `canonical_scientific_stage_count = 9`. Stage10 is release engineering and is not a scientific inference stage.

## Historical notebooks explicitly identified by RELEASE_MANIFEST.json

| Notebook | Manifest label | Expected whole-file SHA-256 |
| --- | --- | --- |
| 05-external-validation.ipynb | HISTORICAL / NON-CANONICAL — DO NOT EXECUTE FOR REPRODUCTION | 685b2be79ec58696f0d9f0e022dd7b7ef32a56c2ba2807a463d598ce3d804bdc |
| 05-h3-statistical-repair-only.ipynb | HISTORICAL / NON-CANONICAL — DO NOT EXECUTE FOR REPRODUCTION | 5891a31b23564e1b502638422d837149c915c68fbd4345f10a86fe8ce0e8f177 |
| 06-fixed-budget-reliability-evaluation.ipynb | HISTORICAL / NON-CANONICAL — DO NOT EXECUTE FOR REPRODUCTION | 31a18d08ff612cb49f37822bd1705445743a8134fa58e8e957f5d7f21038df69 |

## Sensitivity separation

Rule A and Rule B notebooks remain separate post-hoc sensitivity branches. They are not counted as canonical primary scientific notebooks.

- Rule A physical occurrences in byte-inspected archives: **24**
- Rule A unique whole-notebook contents: **6**
- Rule B physical occurrences in byte-inspected archives: **4**
- Rule B unique whole-notebook contents: **4**

## Complete physical matrix

The full 160-row physical matrix, including archive, exact internal path, whole-file hash, normalized cell-source hash, execution state, imports, hard-coded absolute-path indicators, provenance classification, related notebook, and disposition, is in:

`COMPLETE_NOTEBOOK_INVENTORY.csv`

No notebook was edited during this discovery.
