# Final Validation Report — Public-Safe GitHub Core

- Restricted artifacts removed: **26**
- Restricted original paths remaining: **0**
- Exact restricted-byte duplicates remaining elsewhere: **0**
- `.joblib` files remaining: **0**
- Retained Parquets with literal `title` / `original_title` / `neutralized_title` schema: **0**
- Unsafe provenance/metadata literal-text findings after allowlist: **0**
- Notebook outputs containing Bengali literal text: **0**
- Potential secrets: **0**
- Canonical scientific notebooks: **9**
- Stage10 release notebooks: **1**
- Legacy/repair notebooks: **3**
- Rule A sensitivity notebooks: **6**
- Rule B sensitivity notebooks: **4**
- Retained scientific artifact hash changes: **0**
- Manuscript-critical aggregate cross-check: **PASS**
- Largest file: **9,841,445 bytes** (`provenance/rowlevel_audit/03_modeling/CLEAN_DATA_MANIFEST.json`)
- File >100 MB: **NO**

The remaining Bengali text detected in provenance is the author-defined frozen cue lexicon in `provenance/rowlevel_audit/03_modeling/CUE_FEATURE_SCHEMA.json`, not a sampled third-party headline/excerpt.

Validation errors: **None**

## Fresh extraction

- ZIP integrity: **PASS**
- Single repository root: **PASS**
- File-by-file SHA-256 equality after extraction: **PASS**
- Restricted-path absence: **PASS**
- Restricted text-bearing Parquet schema absence: **PASS**
- No `.joblib`: **PASS**
- Notebook structure counts: **PASS**
