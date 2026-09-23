# PROJECT ARTIFACT LINEAGE

## Frozen scientific chain

| Stage | Canonical notebook | Scientific role | Manifest SHA-256 | Checksum SHA-256 | Status SHA-256 |
| --- | --- | --- | --- | --- | --- |
| 01 | 01-dataset-acquisition-and-freeze.ipynb | frozen external dataset acquisition and byte identity | e09b083a38698648f0fe435bb8ff83f5c89896e37538de767549d9f05f72e455 | 3630605b75b593e3f6a5927a34e168de71ad6b6f5091a59b02f71e28b3a05ffa | 96a79231563344688eb35f1bf3a4e55b17a0d6dff6f9f9fa6f9784417cefac18 |
| 02 | 02-p0-integrity-provenance-leakage-audit.ipynb | P0 integrity/provenance/leakage audit | 59a4b6be556af89de26768202dc252cb92e2394dea20e26a1024c58548c233cd | d3111248dae423fdac5d2361ddb154b1d0928374151aee13f99ab2f8fdd822f2 |  |
| 03 | 03-primary-model-development.ipynb | primary model development on clean source splits | 184f3eabd32ab6f0d56da72cdb2b6a908b32b4d662c47eb40a6597acf9acad52 | e40f3c9ac69cb9f23e080e6070a1cd3b28969f43e3592d7b52a04f02a8dbafb2 | 3595786df6ea2218662e019371c19cfa853f032aa2e99fbe7c44dd2298234ae6 |
| 04 | 04-calibration.ipynb | source DEV scalar temperature calibration | 26b9c68be2312a38ac65eda8f7462e32cbe1d7d288693d58e929cfe0c5f059bd | 75811e378257d905152d25b584bbbfc707a9cc36abe890eb5c12eb0f9ec9ab10 | cb3a7c9e3c0a744224a709102773828afa4f0e4eabfbc3fd4bca6694165dce60 |
| 05 | 05-external-validation-and-h3-vfinal.ipynb | canonical external validation and H1/H2/H3/H3-P inference | 7246bc3a49a5f8d8472005cea9f6062d118431cd67d5dd2c4fd8f2ee31f52c27 | 7171a0e581d8fb4bde2a630f4de631d74962954ee3890b8123d78a720823450a | 005399d829bb66870810faa3f140757a7ee3d82eac4bae394fbbe79e701db8e6 |
| 06 | 06-fixed-budget-reliability-evaluation-vfinal.ipynb | canonical fixed-budget reliability evaluation | 5a2962014445c723379b07fcf02e601644818c36e6c15fb7ee9478ca34711285 | 5f8d0f7fc9cde5e0f159bf4c5f4ca8387441675407b4c43ab87a8063afac3beb | 769f33b4771f3ec55700384ae6db7e9e2feb115d23717b27cbd9c17e4983538d |
| 07 | 07-h3-h4-bridge-exploratory-vfinal.ipynb | secondary/exploratory H3→H4 bridge | fa8651ff00a3306dfd48ed55d21df3ab2eef2fbd365db33657f5c2413822e714 | 86c8aeb1c4c0062303b44d397fece5908c315ff8860fca59afc624bfdde39f8d | 566f063f1bef265c6d377565ee6a22ccc1cbbbaa083f7bad7a89423ec6222280 |
| 08 | 08-manuscript-figures-and-tables-vfinal.ipynb | audited FINAL2 manuscript figure/table reproduction | b0ac3b392b2e3d1b5136fda5dbdbcf498e1b89c44cc1d920ff597d6c2967523f | ccd850d3dbe144651bb52e8277308e6438e3e3d869b2cfc232b39db96fed3af6 | d64ee2649a348bad4fd36ee3c4df79fd071b60807b248f0a7a575f5795bf9f33 |
| 09 | 09-source-publisher-cluster-sensitivity-vfinal.ipynb | sensitivity only; never replaces canonical Stages 05/06/07/08 | 445b9bd3bfccbf55af63c92c3e6e966eaf38959641b6507493cb6c249a7035db | c2c2e1c576a94b1efb8aa0072078dca7795375de386f0082666a2966eeb7f285 | c57fb1178ecdcb9e4525a9abaf8b55e62b7e6eabbc0050817aa1df76970937a3 |

### Stage01
`01-dataset-acquisition-and-freeze.ipynb`
→ `01_frozen_banglabait/ACQUISITION_MANIFEST.json`
→ `01_frozen_banglabait/SOURCES_LOCK.json`
→ `01_frozen_banglabait/SHA256SUMS.txt`
→ recorded external-data byte identity.

### Stage02
`02-p0-integrity-provenance-leakage-audit.ipynb`
→ `02_cleaning/P0_AUDIT_REPORT.json`
→ overlap/schema/duplicate/label reports
→ `02_cleaning/SPLIT_FIREWALL.json`
→ Stage03 decontamination response.

### Stage03
`03-primary-model-development.ipynb`
→ `03_modeling/CLEAN_DATA_MANIFEST.json`
→ clean split Parquets and row manifest
→ model/config locks and model artifacts
→ downstream frozen prediction interface.

### Stage04
`04-calibration.ipynb`
→ `04_calibration/DEV_PREDICTIONS.parquet`
→ `04_calibration/TEMPERATURES.json`
→ `04_calibration/CALIBRATION_ARTIFACT_MANIFEST.json`.

### Stage05
`05-external-validation-and-h3-vfinal.ipynb`
→ locked source/BanglaClick/BaitBuster prediction Parquets
→ `H1_RESULTS.json`
→ `H2_RESULTS.json`
→ `H3_RESULTS.json`
→ `H3_PERTURBATION_RESULTS.json`
→ corrected H3 tables
→ `STATISTICAL_SUMMARY.json`
→ `STAGE05_ARTIFACT_MANIFEST.json`.

### Stage06
`06-fixed-budget-reliability-evaluation-vfinal.ipynb`
→ `H4_RESULTS.json`
→ `FIXED_BUDGET_TABLES.csv`
→ `RISK_COVERAGE_DATA.csv`
→ 10% primary-budget tables
→ reproducibility/provenance manifest.

### Stage07
`07-h3-h4-bridge-exploratory-vfinal.ipynb`
→ `BRIDGE_32_CELL_POINT_TABLE.csv`
→ `H3_H4_BRIDGE_RESULTS.json`
→ leave-one-platform-out point table
→ reproducibility/provenance manifest.

### Stage08
`08-manuscript-figures-and-tables-vfinal.ipynb`
→ audited figure-source data
→ canonical main figures/tables
→ supplementary figures/tables
→ reproduction audits/manifests.

### Stage09
`09-source-publisher-cluster-sensitivity-vfinal.ipynb`
→ source-publisher mapping
→ H1/H2/H3/H3-P/H4 source-cluster sensitivity JSONs
→ `CLAIM_CONCORDANCE_TABLE.csv`
→ `SENSITIVITY_SUMMARY.md`.

Stage09 is explicitly **sensitivity only** and does not replace canonical Stages05/06/07/08.

### Stage10
`10-release-package-assembly-vfinal.ipynb`
→ package assembly / validation / release engineering only.
Stage10 is not scientific inference.

## News decontamination sensitivity lineage

`NEWS_DECONTAMINATION_INPUTS.zip`
→ `NEWS_DECONTAMINATION_FEASIBILITY_AUDIT.zip`
→ Rule A / Rule B isolated execution package
→ Rule-specific Stage05/06/07 outputs
→ Rule A/B comparison package
→ supplementary sensitivity material including S10/S11.

Rule A and Rule B remain post-hoc sensitivity analyses. No preferred rule is selected.

## Release lineage

Canonical source notebook
→ executed/captured release notebook
→ Stage10 release copy
→ optional anonymized release copy.

Release and anonymized copies are not additional scientific stages.
