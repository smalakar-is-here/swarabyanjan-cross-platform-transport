# NEWS DECONTAMINATION FEASIBILITY AUDIT

## Execution result

**Central answer: YES WITH CAVEATS**

This audit was performed directly from the uploaded `NEWS_DECONTAMINATION_INPUTS.zip`, including direct row-level decoding of the required Parquet artifacts. Candidate Rule A and Rule B views were evaluated in memory only.

### Integrity

| Check | Result |
|---|---|
| Input integrity | PASS |
| Parquet access | PASS |
| Hash integrity | PASS |
| Overlap reconstruction | PASS |
| Canonical artifacts modified | NO |
| Data deleted | NO |
| Model retrained | NO |
| Inference rerun | NO |

Input bundle SHA-256: `15bcae4daef5dd50ae79edde68686359c2111a0399dfe71b2e059d94e14fadae`

Upstream Stage10 ZIP SHA-256 recorded by the release audit: `e229e6a6d88312cf1eb9067664ce422f63edca4bc89860313061aad71360caaa`

Reader: custom read-only Python Parquet decoder (Compact Thrift metadata + Snappy + PLAIN/RLE_DICTIONARY page decoding). This was deterministic read/diagnostic processing, not model or inferential rerun.

## Direct row-level validation

### Dataset sizes

- CLEAN TRAIN: **10,544**
- CLEAN DEV: **1,100**
- CLEAN TEST: **2,784**
- BanglaClick cache: **99,220 physical rows = 4,961 unique rows × 20 model/seed combinations**
- BaitBuster cache: **200,000 physical rows = 10,000 unique rows × 20 model/seed combinations**

All repeated cache metadata checked for consistency across row IDs: **no inconsistencies**.

### Hash integrity

Stored source raw-title hashes and normalized hashes recomputed exactly from the actual titles:
- TRAIN raw: 10,544/10,544
- TRAIN normalized: 10,544/10,544
- DEV raw: 1,100/1,100
- DEV normalized: 1,100/1,100
- TEST raw: 2,784/2,784
- TEST normalized: 2,784/2,784
- BanglaClick normalized: 99,220/99,220
- BaitBuster normalized on unique rows: 10,000/10,000

Frozen normalization used exactly:
NFKC → remove U+200B/U+200C/U+200D/U+FEFF → casefold → trim → collapse whitespace; punctuation retained.

## Previous overlap findings — direct recheck

| Finding | Previously reported | Directly recomputed | Status |
|---|---:|---:|---|
| TRAIN→News raw shared hash classes | 1,025 | 1,025 | MATCH |
| TRAIN→News normalized shared hash classes | 1,028 | 1,028 | MATCH |
| TRAIN→News normalization-only classes | 3 | 3 | MATCH |
| TRAIN→News affected News rows | 1,044 | 1,044 | MATCH |
| DEV→News normalized affected News rows | 7 | 7 | MATCH |
| TEST→News normalized affected News rows | 23 | 23 | MATCH |
| YouTube↔BaitBuster shared normalized classes | 544 | 544 | MATCH |

### Critical target-ID union

TRAIN-affecting News IDs: **1,044**

DEV-affecting News IDs: **7**

TEST-affecting News IDs: **23**

Pairwise target-ID intersections:
- TRAIN∩DEV = 0
- TRAIN∩TEST = 0
- DEV∩TEST = 0
- TRAIN∩DEV∩TEST = 0

Therefore:

**1,044 + 7 + 23 = 1,074 unique affected News rows — TRUE.**

Rule B removes **1,074/1,500 = 71.6%** of News.

## Label conflict diagnostics

At normalized-title-hash level:

| Source split | Shared hash classes | Same-label classes | Conflicting classes |
|---|---:|---:|---:|
| TRAIN↔News | 1,028 | 496 | 532 |
| DEV↔News | 7 | 1 | 6 |
| TEST↔News | 23 | 5 | 18 |

These are observed label disagreements only. They do not establish which label is correct.

## Candidate Rule A

Definition: remove News rows whose normalized protected-title hash matches CLEAN TRAIN.

Original News:
- n = **1,500**
- label 1 = **1,175**
- label 0 = **325**
- prevalence = **78.33%**

Rule A retained:
- n = **456**
- removed = **1,044 (69.6%)**
- label 1 = **295**
- label 0 = **161**
- prevalence = **64.69%**
- prevalence shift = **−13.64 percentage points**

H4 slot counts under the exact frozen Stage06 ceiling rule:
- 5% = **23**
- 10% = **46**
- 20% = **92**

**Classification: FEASIBLE WITH SERIOUS SUPPORT/REPRESENTATIVENESS CAVEATS**

## Candidate Rule B

Definition: remove News rows whose normalized protected-title hash matches CLEAN TRAIN ∪ DEV ∪ TEST.

Original News:
- n = **1,500**
- label 1 = **1,175**
- label 0 = **325**
- prevalence = **78.33%**

Rule B retained:
- n = **426**
- removed = **1,074 (71.6%)**
- label 1 = **270**
- label 0 = **156**
- prevalence = **63.38%**
- prevalence shift = **−14.95 percentage points**

H4 slot counts:
- 5% = **22**
- 10% = **43**
- 20% = **86**

**Classification: FEASIBLE WITH SERIOUS SUPPORT/REPRESENTATIVENESS CAVEATS**

## Cue-family support

Support means rows with a stored cue-family value > 0.

| Cue family | Original n | Rule A n | Rule B n |
|---|---:|---:|---:|
| curiosity_information_withholding | 27 | 2 | 1 |
| direct_address_imperative | 48 | 7 | 4 |
| exaggeration_superlative | 29 | 5 | 4 |
| interrogative_structure | 111 | 23 | 18 |
| punctuation_intensity | 262 | 47 | 37 |
| sensational_emotional_loading | 40 | 2 | 2 |
| specificity_numerals | 360 | 112 | 109 |
| vague_deictic_reference | 85 | 11 | 9 |

No cue field had nulls in the audited News rows.

The retained samples therefore remain structurally defined for all eight frozen cue families, but several families have very small support counts. This creates a serious sparsity/estimability caveat rather than a reason to alter the cue definitions.

## Target cluster/source structure

The available target metadata expose `cluster_id`. URL and publication-date fields are not present in the audited prediction cache.

| Metric | Original | Rule A | Rule B |
|---|---:|---:|---:|
| News rows | 1,500 | 456 | 426 |
| unique clusters | 943 | 54 | 47 |
| singleton clusters | 918 | 31 | 28 |
| dominant cluster | `bc:news:jugantor.com` | same | same |
| dominant cluster n | 112 | 112 | 112 |
| dominant cluster share | 7.47% | 24.56% | 26.29% |
| top-5 cluster share | 21.60% | 58.77% | 62.91% |
| top-10 cluster share | 30.60% | 77.41% | 82.39% |
| cluster HHI | 0.01364 | 0.10687 | 0.12108 |

These figures show a large observable composition change after removal, especially cluster concentration. `cluster_id` is treated only as an analytical clustering/structure field; no legal publisher identity is inferred.

## Within-News duplication

| Metric | Original | Rule A | Rule B |
|---|---:|---:|---:|
| duplicated normalized-title classes | 27 | 11 | 11 |
| duplicated rows | 54 | 22 | 22 |
| largest duplicate class | 2 | 2 | 2 |
| duplication rate | 3.60% | 4.82% | 5.16% |
| unique normalized titles | 1,473 | 445 | 415 |

This is target-internal duplication, separate from source→target textual contamination.

## H4 structural adequacy

The frozen Stage06 implementation computes review slots as:

`max(1, ceil(budget × n))`

so the candidate retained samples support all three fixed budgets without capacity failure. The analysis remains structurally executable; statistical estimability must be re-established by a scientific rerun.

## What becomes stale if either rule is adopted?

| Analysis | Affected? | Scientific rerun required? | Reason |
|---|---|---|---|
| H1 | YES | YES | News target rows change |
| H2 | YES | YES | News target rows change |
| H3 | YES | YES | News cue–outcome and proper-loss cells change |
| H3-P | YES | YES | News controlled-sensitivity cells change |
| H4 | YES | YES | News fixed-budget metrics/bootstrap inputs change |
| H3→H4 bridge | YES | YES | News H3/H4 quantities change |
| News prevalence/sparsity diagnostics | YES | YES | target composition changes |
| News supplementary tables/figures | YES | YES | News-dependent outputs change |
| Facebook/YouTube/Twitter row-level analyses | NO | NO from this rule alone | only News is decontaminated |
| BaitBuster row-level analyses | NO | NO from this rule alone | separate target remains unchanged |
| Multi-platform summary tables/figures | YES | YES | aggregate presentation contains News |
| Stage09 source-site sensitivity outputs for News | YES | YES | target-side inputs change |

No reruns were executed here.

## Interpretation

The audit establishes four separate phenomena:

1. **Textual contamination:** exact source→News title/hash overlap exists.
2. **Procedural holdout:** the prior source-side firewall can remain true while textual overlap also exists.
3. **Target-target dependence:** 544 normalized-title classes are shared by BanglaClick YouTube and BaitBuster.
4. **Label disagreement:** many shared normalized-title hashes have different observed binary labels.

These should not be conflated.

## Final conclusion

Both Rule A and Rule B can be implemented as **new versioned News analyses without redesigning the underlying estimand definitions**. Both retain nonzero support for all eight frozen cue families and enough rows for all three fixed review budgets.

However, both remove a very large fraction of the original News sample and materially change label prevalence and observable cluster composition. Several cue families become extremely sparse. Therefore the existing frozen scientific results cannot be carried forward unchanged; adopting either rule requires a new scientific version with a full rerun of every News-dependent analysis and its downstream figures/tables.

# CENTRAL ANSWER: YES WITH CAVEATS

## Required safety statement

NO DATA WERE DELETED.

NO FROZEN SCIENTIFIC RESULT WAS CHANGED.

NO MODEL WAS RETRAINED.

NO INFERENCE WAS RERUN.

Candidate rules were evaluated only in memory.

The canonical Stage10 Parquets were not modified.

The central answer is based on direct row-level inspection of the uploaded Parquet artifacts.
