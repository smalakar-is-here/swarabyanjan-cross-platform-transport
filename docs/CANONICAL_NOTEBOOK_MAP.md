# Canonical Notebook Map

## Scientific canonical workflow: Stage01–Stage09

Exactly nine notebooks in `notebooks/` are the canonical scientific source notebooks.

| Stage | Canonical source notebook | Source whole-file SHA-256 | Manifest canonical-source SHA-256 | Stage10 release name | Stage10 release whole-file SHA-256 |
| --- | --- | --- | --- | --- | --- |
| 01 | 01-dataset-acquisition-and-freeze.ipynb | 627e38c7fd99c53b483bc109404b03e781b060c58215e639121838cb63abe8cf | 009f92a04bfd38d5f4b521fbbec3306845bafa3281206736b16b49e8f1514196 | 01-dataset-acquisition-and-freeze.ipynb | 0223500a7d1a58c1044d664e01c20e7e776fd4ceb58e8e8575181ac1944c0481 |
| 02 | 02-p0-integrity-provenance-leakage-audit.ipynb | 425357156376ea9b5724fae120ca969f8f2c93c94d0f92cba252461b2835a85f | 2f248e31bdf6421a2c4fba6ac78d2300f1f479bd09da317d833f2f916179d68b | 02-p0-integrity-provenance-leakage-audit.ipynb | 77f5ad63c6c60dc637d15e2f7eec544e23138fb8f7d3b9e1c25e083f7f601d22 |
| 03 | 03-primary-model-development.ipynb | db17c0ecbc55eb15a9c5e7518616df6494400db093c82a66576cd4911b221f1c | 260246bd57233e6a663ea5258a69e8a500b5b1b4a059381df51125b148a0271f | 03-primary-model-development.ipynb | 9847f4b0d6f31afca35a698557f9bf3cf339f37b9c14386829960474f59645da |
| 04 | 04-calibration.ipynb | 9c80cfe9a16cba71813f7cb539368e674694d820228b66b8392ad83341cd81ea | d5f0c82ca353780e15cef5f27b8ae5fbcb3e840a18624ef1c7196b8e821e9908 | 04-calibration.ipynb | 64d23570c8df9c0559fce2b4194f40f3f298b9e9b2ca844644382f5e0bf5f4ef |
| 05 | 05-external-validation-and-h3-vfinal.ipynb | ee258026c5d6634c6b0c0bb66553971d06a9148e36ff13a4ea7f4886905a81cf | 28f131f3004bd38a057a0f17f0a89e14fd6fbd13c5f5d17f1a540898736b869b | 05_EXTERNAL_VALIDATION_H3_vFINAL.ipynb | 9155f804884958c43b119d68b43e1b2d4763b5daa661e1e009039782ab9d4ad5 |
| 06 | 06-fixed-budget-reliability-evaluation-vfinal.ipynb | 4fa809490ffd9a08791fe2da1fab5b1056b51c68443bd97060577d05b1012c43 | 3edac9c63c3cb5711ee5a4e01afff89727929fd1629afab1256136335b3bb809 | 06_FIXED_BUDGET_RELIABILITY_EVALUATION_vFINAL.ipynb | 7d73b4a6a58ca0746deb90e6acee271cc1e261364a7192001eceafd89f416702 |
| 07 | 07-h3-h4-bridge-exploratory-vfinal.ipynb | e3c961eff6a380eb86f919dc598b5298cadb6d749548e3f33fb97c5b7c8f4967 | c5930a6a11ffdb49f169e75b8dff73f7fbdceed5eb5da160646200c5164efa87 | 07_H3_H4_BRIDGE_EXPLORATORY_vFINAL.ipynb | 967a7af8f994ab8273d58fd49aca561c2f4b3db232739e3501db9ab8036e0fcb |
| 08 | 08-manuscript-figures-and-tables-vfinal.ipynb | 8766604fe554acb362811d35314982d62f413a7489068aeca8b619915b479c62 | 038699569e4562388ebd77df3a8cb6069d0d0a37040b839222a21cde8b795042 | 08_MANUSCRIPT_FIGURES_AND_TABLES_vFINAL.ipynb | 8e779f1388d143c26a004e5cdd4985e9462241909e33f8723dfcfa1e82fd8095 |
| 09 | 09-source-publisher-cluster-sensitivity-vfinal.ipynb | d4b267d17debe23f470343dcb0d949c274a31510d3acc5178ee6a18f2adcb64c | e214251831c1dde32d60ffe0e06196e331e05ce9d331747861fc058751b39a65 | 09_SOURCE_SITE_CLUSTER_SENSITIVITY_vFINAL.ipynb | 346ce289c38410c964dd8779661eea2eba82785c4c7b591fc91a52b6da2ea09b |

The manifest-level canonical-source SHA-256 is the Stage10 source-identity field and is not assumed to be identical to the notebook container's whole-file hash.

## Stage10

`../scripts/release/10-release-package-assembly-vfinal.ipynb` is **release engineering only**. It assembles and validates an already frozen release and is not a tenth scientific inference stage.

## Legacy / repair notebooks

The three known historical variants are preserved under `../provenance/archive/notebooks/`:

- `05-external-validation.ipynb`
- `05-h3-statistical-repair-only.ipynb`
- `06-fixed-budget-reliability-evaluation.ipynb`

They are not canonical scientific notebooks.

## Rule A / Rule B

Post-hoc News-decontamination sensitivity notebooks are under:

- `../experiments/sensitivity/rule_a/`
- `../experiments/sensitivity/rule_b/`

They are not primary analyses, replacements, or preferred alternatives.
