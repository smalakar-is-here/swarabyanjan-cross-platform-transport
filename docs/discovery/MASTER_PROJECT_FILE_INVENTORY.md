# MASTER PROJECT FILE INVENTORY

## Scope and counting rules

This inventory covers the **byte-accessible Swarabyanjan artifact universe in the active runtime** and separately records the accessible Library/project metadata layer.

- Byte-inspected artifact occurrences: **3,791**
- Unique SHA-256 contents among byte-inspected occurrences: **574**
- Byte-inspected top-level ZIP archives: **13**
- Nested archives discovered and recursively inventoried: **1**
- Additional Swarabyanjan Library/project metadata records identified: **411**
- Confirmed critical Library artifacts whose raw-byte materialization failed: **3**

`3,791` is an **occurrence count**, not a canonical-file count. Repeated release copies, anonymized copies, generated repository copies, and duplicated archive members are intentionally retained in the inventory.

The Library/project metadata count is reported separately because those records do not all expose retrievable bytes in the active runtime and therefore are not included in SHA-256 unique-content arithmetic.

## File-type counts — byte-inspected occurrence universe

| Extension / group | Count |
| --- | --- |
| .ipynb | 160 |
| .py | 0 |
| .sh | 0 |
| .bat | 0 |
| .ps1 | 0 |
| .yaml | 0 |
| .yml | 0 |
| .toml | 0 |
| .json | 1267 |
| .csv | 1023 |
| .tsv | 0 |
| .xlsx | 12 |
| .parquet | 130 |
| .md | 272 |
| .tex | 62 |
| .bib | 2 |
| .pdf | 170 |
| .png | 224 |
| .jpeg | 2 |
| .zip | 1 |
| .tar | 0 |
| .gz | 0 |
| other extensions | 466 |

## Primary role classification

| Inventory role | Physical occurrences | Unique SHA-256 contents |
| --- | --- | --- |
| CONFIG_OR_METADATA | 35 | 20 |
| DATA_OR_CACHE_ARTIFACT | 16 | 6 |
| DATA_OR_EVIDENCE_TABLE | 75 | 37 |
| DOCUMENTATION | 58 | 26 |
| MANUSCRIPT_ARTIFACT | 482 | 85 |
| MODEL_ARTIFACT | 20 | 10 |
| NOTEBOOK | 160 | 38 |
| OTHER | 9 | 3 |
| PROVENANCE_ARTIFACT | 1048 | 216 |
| PUBLICATION_OR_DOCUMENT_ARTIFACT | 20 | 19 |
| RESULT_ARTIFACT | 934 | 185 |
| SUPPLEMENTARY_ARTIFACT | 909 | 80 |
| TEXT_OR_LOG | 25 | 15 |

## Source-code finding

No standalone `.py`, `.sh`, `.bat`, or `.ps1` implementation files were found in the byte-inspected direct/archive universe.

The implemented scientific pipeline source is therefore primarily **notebook-embedded code** in the discovered `.ipynb` files. This is an evidence finding, not a claim that no historical standalone source ever existed elsewhere.

## Library/project metadata layer

The Files/Library survey identified **411** Swarabyanjan-related metadata records. Their type distribution was:

| Type | Metadata records |
| --- | --- |
| .zip | 106 |
| .ipynb | 63 |
| .md | 60 |
| .json | 47 |
| application/json (no extension) | 1 |
| .txt | 42 |
| .docx | 29 |
| .csv | 24 |
| .pdf | 13 |
| .log | 13 |
| .xlsx | 9 |
| .png | 2 |
| .whl | 1 |
| .bib | 1 |

These Library records include historical repair notebooks, repair ZIPs, execution logs, manuscript packages, final PDFs, evidence workbooks, and documentation. They are not silently merged into byte-level counts because their bytes were not uniformly materializable.

## Machine-readable inventory

`MASTER_PROJECT_FILE_INVENTORY.csv` contains every byte-inspected occurrence with:

- exact path;
- source/archive;
- nesting level;
- extension/type;
- byte size;
- SHA-256;
- inferred stage;
- role classification;
- scientific class;
- public-repository candidacy.

No repository normalization, deletion, or movement was performed during discovery.
