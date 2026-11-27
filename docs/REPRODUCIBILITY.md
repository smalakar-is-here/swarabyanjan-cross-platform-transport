# Reproducibility --- R6 Historical Baseline and SCOPE Successor

This repository contains two distinct research tracks. Their code,
environments, outputs and claims must not be mixed.

## 1. R6 historical baseline

R6 uses the original Stage01--Stage09 notebook workflow. Stage10 is
release/package engineering only for the historical R6 study.

The R6 tree, frozen results and provenance must remain unchanged when
running SCOPE. For R6 reproduction, follow the canonical notebook map,
stage manifests, external-restricted artifact manifest and historical
scientific locks.

A fresh public clone may not contain restricted third-party text or
model binaries. Verify the exact hashes and access conditions before
restoring excluded inputs. Do not infer redistribution permission from
public accessibility.

## 2. SCOPE successor

SCOPE is a separately versioned successor study. Its implementation,
tests, configurations and results are distinct from the R6 notebook
baseline.

Use: - `src/scope/` for the implementation (verify path in the current
checkout); - `tests/` for validation; - `requirements-scope.txt` for the
SCOPE environment; - `python -m scope.run` for CPU-compatible experiment
groups; - `experiments/kaggle/WORKER_UNITS.md` for independent GPU
units; - `scripts/scope_make_worker_packages.py` for worker-package
generation; - `docs/scope/EXPERIMENT_REGISTRY.csv` for run status; -
`docs/scope/CLAIM_EVIDENCE_MAP.md` for claim-to-evidence mapping; -
`docs/scope/REPRODUCE.md` for commands and data paths.

Verify each path and command against the current checkout before
execution. Documentation is not proof that a run completed.

## 3. Data boundary

Keep restricted data in the Git-ignored workspace `private_data/scope/`
where the licence and access terms permit. The repository must not
contain restricted row-level third-party text, secrets or credentials.

A private Kaggle dataset is an external transfer. Confirm that the
dataset terms permit the intended processing and hosting before upload.
If permission is unclear, continue local analyses and experiments using
cleared data.

## 4. Run provenance

Every experiment must record: - unique experiment ID; - code revision
and configuration hash; - data version and SHA-256; - model and
hyperparameters; - split and seed; - execution environment and account
alias; - submission, running, completion and download
timestamps/status; - output file hashes; - validation and scoring
status.

A submitted kernel is not a completed experiment. A completed kernel is
not a scored result. A scored result is not validated until its outputs,
configuration and metric calculations are checked.

## 5. Statistical and scientific reproducibility

Use the same metric implementation across baselines. Respect cluster
dependence and the pre-specified inferential design. Report uncertainty,
negative findings, partial identification and non-estimability honestly.
Never tune on held-out target test labels.

## 6. Publication assets

Generate figures, tables and manuscript numbers from machine-readable
results. Store scripts and manifests. Verify every figure/table value
against its source. Preserve the R6 release assets separately from SCOPE
successor assets.
