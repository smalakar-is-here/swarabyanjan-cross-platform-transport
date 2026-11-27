# Swarabyanjan --- R6 Baseline and SCOPE Successor

Swarabyanjan contains two related but distinct research tracks. This
repository/branch preserves the R6 historical reproducibility core and
the SCOPE successor implementation. Do not conflate their protocols,
results or claims.

## Research tracks

### R6 --- historical baseline

R6 is the previously completed Bengali clickbait external-validity
study. Its original Stage01--Stage09 notebooks, frozen primary results,
sensitivity outputs, provenance and Stage10 release engineering remain
historical baseline artifacts.

The R6 analysis used BanglaBait as the source, BanglaClick as separate
target-platform datasets, and BaitBuster-Bangla for external
corroboration only. The historical hypotheses, estimands, bootstrap
rules and non-estimability statuses are documented in the R6 provenance
and `docs/SCIENTIFIC_LOCKS.md` in the parent workspace.

R6 results must not be silently replaced by successor results.

### SCOPE --- successor research

SCOPE investigates why Bengali clickbait detection performance and cue
relationships vary across datasets and platforms, including the roles of
prevalence, text/cue distributions and annotation/label definitions.

The successor programme is separately versioned and must be
reconstructed from the current source code, configurations,
machine-readable results and run manifests. The preliminary design is
not proof that all proposed components are identifiable or novel.

The current branch documentation identifies: - implementation under
`src/scope/`; - automated tests under `tests/`; - experiment runner
`python -m scope.run`; - GPU worker specification under
`experiments/kaggle/WORKER_UNITS.md`; - worker-package generator
`scripts/scope_make_worker_packages.py`; - experiment registry under
`docs/scope/EXPERIMENT_REGISTRY.csv`; - claim/evidence map under
`docs/scope/CLAIM_EVIDENCE_MAP.md`; - reproduction instructions in
`docs/scope/REPRODUCE.md`.

**Verify that each path exists in the current checkout before relying on
it.** If a documented path is absent, report the mismatch rather than
inventing a replacement.

## SCOPE research principles

1.  Separate platform effects from dataset and annotation effects where
    the design permits.
2.  Reconcile overlap/anchor denominators and label thresholds before
    making comparisons.
3.  Formalise estimands and identification assumptions.
4.  Compare against fair, strong baselines.
5.  Use cluster-aware uncertainty and appropriate multiple-testing
    procedures.
6.  Report negative, partial and non-estimable findings.
7.  Do not claim causality, universal decomposition or SCOPE superiority
    without evidence.
8.  Keep R6 and SCOPE results separately versioned.

## Data and privacy boundary

Restricted third-party text and private research records are not part of
the public repository. Keep private data in the Git-ignored workspace
`private_data/scope/` when permitted by the data terms. A private Kaggle
dataset is still an external transfer; verify the licence and transfer
permission before uploading. Never commit tokens, credentials,
restricted raw text or private records.

Public-safe aggregates and reproducibility metadata may be shared only
according to their provenance and applicable licences.

## Repository map

``` text
notebooks/                         R6 historical Stage01–Stage09 notebooks
scripts/release/                   R6 Stage10 release engineering
results/primary/                   R6 frozen primary results
results/sensitivity/               R6 Rule A / Rule B sensitivity evidence
provenance/                        R6 provenance and restricted-artifact metadata
src/scope/                         SCOPE successor implementation (verify current checkout)
tests/                             SCOPE tests (verify current checkout)
experiments/kaggle/                SCOPE independent GPU work units
scripts/scope_make_worker_packages.py  SCOPE worker package generator
docs/scope/                        SCOPE registry, evidence map and reproduction guide
results/scope_v1/                  versioned SCOPE results, if present
```

## Reproducibility

-   R6: follow the historical notebook and provenance instructions in
    `docs/REPRODUCIBILITY.md` and the R6 manifests.
-   SCOPE: follow `docs/scope/REPRODUCE.md`. Use the SCOPE environment
    and versioned outputs; do not run successor experiments inside the
    frozen R6 notebook sequence.

## Current evidence and limitations

`docs/scope/CLAIM_EVIDENCE_MAP.md` is a claim index. Verify every claim
against the underlying code, result files, logs and manifests before
using it in a manuscript. A claim map entry is not itself a
completed-run artifact.

The current evidence map records limitations, including partial loss
decomposition, point-estimate-only cue-spread evidence, YouTube-specific
regime/platform contrast, no PPI gain, unsupported anchor
representativeness for frozen models, and no causal or
platform-by-regime interaction claim. Preserve these qualifications
unless new verified evidence justifies a change.

## Citation and licence

Do not invent a repository-wide licence or `CITATION.cff`. Follow the
current repository licence-status documentation and obtain author
approval for any new licence or public data release.

## Integrity

Every reported number must map to a versioned machine-readable result.
Keep configuration hashes, code revision, data hashes, seeds, logs and
output manifests. Never hand-edit result values.
