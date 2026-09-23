# Public Redistribution Safety Audit

## Decision

Restricted bytes are excluded conservatively because public redistribution permission is unresolved. This is a repository packaging decision, not a legal conclusion about the underlying works.

## Removed artifacts

Exact count: **26**

- `data/derived/primary/stage03/clean_splits/banglabait_train_clean.parquet` — 2104360 bytes — `a00c02c92a51eef1e5fb10f358909e5feceaaa3413cae83f080a952c206a3c89` — Contains the literal third-party `title` field; public redistribution permission is unresolved.
- `data/derived/primary/stage03/clean_splits/banglabait_dev_clean.parquet` — 227702 bytes — `584533f179c3bef867ef93e9b431d24f063ee85ea010ed38ecc850dd041d4ef4` — Contains the literal third-party `title` field; public redistribution permission is unresolved.
- `data/derived/primary/stage03/clean_splits/banglabait_test_clean.parquet` — 571585 bytes — `3f033ca9479db97e1f11684b6ae223f095678e8fa6b1f2a7b9fa03ac51147e96` — Contains the literal third-party `title` field; public redistribution permission is unresolved.
- `data/derived/primary/stage05/LOCKED_SOURCE_TEST_PREDICTIONS.parquet` — 1716919 bytes — `df863d7a24f7d451451822890de58613c3295eb0bcc1aa882150137b0aa9880a` — Contains literal third-party `title` values. Public redistribution permission is unresolved.
- `data/derived/primary/stage05/BANGLACLICK_PLATFORM_PREDICTIONS.parquet` — 18359666 bytes — `5685fd4142a0e8e7f85cd6040cf48042da2368d5f9596a7740c4f9f09c9519d5` — Contains literal third-party `title` values. Public redistribution permission is unresolved.
- `data/derived/primary/stage05/BAITBUSTER_HUMAN_PREDICTIONS.parquet` — 22941908 bytes — `a3e95820c62de00984a189e2c250f7fffd223ad4b3954db04a5368c7a544b7b9` — Contains literal third-party `title` values. Public redistribution permission is unresolved.
- `data/derived/primary/stage05/H3_PERTURBATION_PAIRS.parquet` — 12545189 bytes — `aa43d73922dbd9ab7b7d753a6aef8e25ad7e6438836fd13394b7925514ed5bb1` — Contains literal `original_title` and `neutralized_title` values derived from third-party text. Public redistribution permission is unresolved.
- `data/derived/sensitivity/rule_a/05_external_validation_and_H3_vFINAL/BANGLACLICK_PLATFORM_PREDICTIONS.parquet` — 11083967 bytes — `1a2496cc810e35a0082c1781af766a8a253b3d63340cb67fd709d1025d55cc54` — Contains third-party text-bearing rows carried into the isolated sensitivity branch; public redistribution permission is unresolved.
- `data/derived/sensitivity/rule_a/05_external_validation_and_H3_vFINAL/H3_PERTURBATION_PAIRS.parquet` — 4262313 bytes — `7b8dd09ed22b177c461a6137c706ff64120d4f071601e0708cd8f6477229eb94` — Contains third-party text-bearing rows carried into the isolated sensitivity branch; public redistribution permission is unresolved.
- `data/derived/sensitivity/rule_b/05_external_validation_and_H3_vFINAL/BANGLACLICK_PLATFORM_PREDICTIONS.parquet` — 11036020 bytes — `a93e45a8fd37a8ee1938dc606c08fa450b2cb73d520a78e8e81595b1c26b587f` — Contains third-party text-bearing rows carried into the isolated sensitivity branch; public redistribution permission is unresolved.
- `data/derived/sensitivity/rule_b/05_external_validation_and_H3_vFINAL/H3_PERTURBATION_PAIRS.parquet` — 4255538 bytes — `87b551728737d6db302bba9d4b6df6144cdadb2eb7925aa82bcaca35130f7f15` — Contains third-party text-bearing rows carried into the isolated sensitivity branch; public redistribution permission is unresolved.
- `provenance/canonical/stage02/P0_AUDIT_REPORT.json` — 292961 bytes — `59a4b6be556af89de26768202dc252cb92e2394dea20e26a1024c58548c233cd` — Contains literal third-party title/excerpt examples; public redistribution permission is unresolved.
- `provenance/canonical/stage02/P0_BLOCKER_FORENSICS.json` — 128706 bytes — `d24c412ae58143685dfa6978b7a5849f35796141ea770805f781b111c2506ded` — Contains literal third-party title/excerpt examples; public redistribution permission is unresolved.
- `provenance/canonical/stage03/DECONTAMINATION_FORENSICS.json` — 34391 bytes — `46e439c949d74940382aaa6de35424733a5b6c9a0fbaf4866db69d5d3aef8f7f` — Contains literal third-party title/excerpt examples; public redistribution permission is unresolved.
- `data/derived/primary/stage09/SOURCE_PUBLISHER_MAPPING.csv` — 687813 bytes — `dcb97621c92b32c422cfdaded4f3c1b43c6885281af88399a8e3cd2efb4b1e58` — Contains row-level raw source URLs from the third-party BanglaBait data interface; public redistribution permission is unresolved.
- `results/primary/stage09/SOURCE_CLUSTER_DIAGNOSTICS.json` — 9152 bytes — `385f2d80680f75cfe835f5cf004a356b27f8243164de3e7860c823e287783735` — Contains representative row-level raw source URLs from the third-party BanglaBait data interface; aggregate Stage09 results remain public.
- `results/primary/stage03/models/ebm_seed137.joblib` — 21426 bytes — `0e420c3e7a29be6e2fe4195685ab36984701dedf9ac723b3f7f05f4785590577` — Model artifact redistribution permission is explicitly unresolved in data/access/MODEL_LICENSES.md; binary excluded from the public core.
- `results/primary/stage03/models/ebm_seed2718.joblib` — 21385 bytes — `49028e420b290d2a8d5eeafe882d6b4ef2209b4bd276514ad9148931010f4419` — Model artifact redistribution permission is explicitly unresolved in data/access/MODEL_LICENSES.md; binary excluded from the public core.
- `results/primary/stage03/models/ebm_seed31415.joblib` — 21405 bytes — `f46e18631ec4bd1c781986920f1774d649c419e703c0589c445e9764d8788768` — Model artifact redistribution permission is explicitly unresolved in data/access/MODEL_LICENSES.md; binary excluded from the public core.
- `results/primary/stage03/models/ebm_seed42.joblib` — 21394 bytes — `7b3bd07b386e8ab32a91b253cb043fb2f2901be1b13a800edc4eb7be8638e2b1` — Model artifact redistribution permission is explicitly unresolved in data/access/MODEL_LICENSES.md; binary excluded from the public core.
- `results/primary/stage03/models/ebm_seed65537.joblib` — 21414 bytes — `2a530d2bbcdb70334967a90d7dc8de7d1552d1ebe7b7c1b08e8fed58eb58a15a` — Model artifact redistribution permission is explicitly unresolved in data/access/MODEL_LICENSES.md; binary excluded from the public core.
- `results/primary/stage03/models/tfidf_lr_seed137.joblib` — 1806907 bytes — `33682958995021ed62cd7fe00561c69c63fcf7678ef4b251f984fe626d691444` — Model artifact redistribution permission is explicitly unresolved in data/access/MODEL_LICENSES.md; binary excluded from the public core.
- `results/primary/stage03/models/tfidf_lr_seed2718.joblib` — 1806909 bytes — `26ea76e64c8a995809b2990320152eeab9169319557a1970a7715850947c2766` — Model artifact redistribution permission is explicitly unresolved in data/access/MODEL_LICENSES.md; binary excluded from the public core.
- `results/primary/stage03/models/tfidf_lr_seed31415.joblib` — 1806911 bytes — `0ce702f21924bd6840953d6b5351d245638ded1ec71513d2cf36040f31868ff8` — Model artifact redistribution permission is explicitly unresolved in data/access/MODEL_LICENSES.md; binary excluded from the public core.
- `results/primary/stage03/models/tfidf_lr_seed42.joblib` — 1806906 bytes — `a3a302540bcc4c404c30a3ca865a9cd51d72daf2f51952f8faef363948f153fc` — Model artifact redistribution permission is explicitly unresolved in data/access/MODEL_LICENSES.md; binary excluded from the public core.
- `results/primary/stage03/models/tfidf_lr_seed65537.joblib` — 1806907 bytes — `05a7da63ee6caa1d2e871d39d809ad1aa95b15585fe53e93165406331268c974` — Model artifact redistribution permission is explicitly unresolved in data/access/MODEL_LICENSES.md; binary excluded from the public core.

## Content inspection findings

- Clean BanglaBait split Parquets: actual Parquet footer metadata contains a literal `title` column.
- Stage05 prediction Parquets: actual footer metadata contains a literal `title` column.
- H3-P pair Parquets: actual footer metadata contains `original_title` and `neutralized_title`; binary string inspection also exposed literal text fragments.
- Stage02/Stage03 forensic JSON files: actual JSON parsing confirmed literal `excerpt`/`title` examples.
- Stage09 source/publisher mapping: actual CSV contains row-level `raw_domain` URLs; the diagnostics JSON contains representative raw URL rows.
- Ten Stage03 `.joblib` model binaries: removed because their public redistribution status remained unresolved, independent of whether individual binaries embed source text.

## Retained derived Parquets inspected

The following retained Parquets were checked against their actual footer schema and notebook write contract:

- `data/derived/primary/stage03/CLEAN_ROW_MANIFEST.parquet` — hashes/IDs/decisions; no literal title field.
- `data/derived/primary/stage03/TRAIN_DEV_SPLIT_ROWS.parquet` — row ID/hash/label/assignment; no literal title field.
- `data/derived/primary/stage04/DEV_PREDICTIONS.parquet` — row ID/label/model/seed/logit/probability fields; no literal title field.

## Safety principle

A hash, manifest, filename, access URL, or verification instruction is retained as provenance; the excluded third-party text/model bytes themselves are not.
