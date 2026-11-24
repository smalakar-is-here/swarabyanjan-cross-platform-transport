# Swarabyanjan — Public-Safe Reproducibility Core

Swarabyanjan is a reproducibility and evidence repository for a Bengali clickbait external-validity study of cue–outcome transport, cue-conditioned proper-loss transport, probability reliability, and fixed-budget review behaviour under distribution shift.

## Study at a glance

| Resource | Role |
|---|---|
| BanglaBait (decontaminated) | Source. Models are developed only here: TRAIN 10,544, DEV 1,100, TEST 2,784 after quarantining 628 protected-title rows. Temperature scaling is fitted on DEV only. |
| BanglaClick | Four separately evaluated target platforms: News 1,500, Facebook 1,961, YouTube 1,000, Twitter 500. Never used for training, calibration or tuning. |
| BaitBuster-Bangla | 10,000 human-labelled rows of a related YouTube/multimodal construct. **External corroboration only**, not a construct-equivalent replication. |
| BanMANI, BanFakeNews-2.0 | Construct-boundary resources only. |

Four model families (EBM, TF-IDF + logistic regression, BanglaBERT, XLM-R) × five seeds (42, 137, 2718, 31415, 65537).

| Analysis | Evidential role |
|---|---|
| H1, H2 | Supporting: model-level discrimination/reliability decoupling and calibration-gain attenuation |
| H3 | **Primary**: source-to-platform transport of eight prespecified cue–label (ΔRD) and cue-conditioned proper-loss (ΔNLL-gap) relationships on 32 cue–platform cells; 24/32 estimable, 8/32 non-estimable; the global 32-dimensional Wald test and both eight-family Simes/BH procedures are NON_ESTIMABLE |
| H3-P | Controlled model-sensitivity corroboration (150/150 estimable); non-causal |
| H4 | Operational: fixed-budget review at a 10% primary budget (5% and 20% sensitivity) |
| H3→H4 bridge | Exploratory and **descriptive only**; all four joint-bootstrap analyses are NON_ESTIMABLE |
| Stage09; Rule A/B | Source-site sensitivity; post-hoc News-decontamination sensitivity |

All inference uses 5,000 locked bootstrap draws and a fail-closed all-valid-draw rule. Non-estimability is a boundary on inference, not evidence of stability, instability or a null effect.

## Scientific workflow

The canonical scientific workflow contains **exactly nine notebooks: Stage01–Stage09** in `notebooks/`. Stage10 is **release engineering only** and is kept at `scripts/release/10-release-package-assembly-vfinal.ipynb`.

The scientific implementation is notebook-embedded. The completed forensic inventory found no standalone implementation tree, so this repository does not fabricate a `src/` package. The only standalone Python files are the presentation-only figure generators in `scripts/publication_figures/` (added 2026-11-15); they re-render frozen Stage08 source data and perform no scientific computation.

## Public-safe data boundary

This public core intentionally excludes frozen artifacts whose public redistribution permission is unresolved when they contain third-party text/row-level source bytes, together with the 10 Stage03 `.joblib` model binaries whose redistribution permission remains unresolved.

The excluded bytes are **not needed to inspect the published aggregate H1/H2/H3/H3-P/H4/bridge results, figures, tables, or Rule A/B comparison evidence**. Exact expected paths, filenames, sizes, SHA-256 values, access/reproduction routes, dependency links, and verification instructions are preserved under `provenance/external_restricted/`.

A fresh clone therefore does **not** contain every private/external input needed to rerun all notebooks. An authorised researcher must first obtain/reproduce the restricted inputs and verify their hashes. Public reproducibility metadata does not imply public redistribution rights.

## Repository map

```text
notebooks/                         9 canonical scientific notebooks
scripts/release/                   Stage10 release-engineering notebook
configs/                           frozen configuration/schema locks
experiments/sensitivity/           Rule A / Rule B branch notebooks
results/primary/                   aggregate frozen primary results/figures/tables
results/sensitivity/               Rule A / Rule B aggregate sensitivity evidence
results/publication/               journal-presentation figure re-renders (presentation only)
scripts/publication_figures/       deterministic generators and QA for results/publication/
data/access/                       data/model access and licensing boundary
data/derived/                      only retained non-text-bearing derived interfaces
provenance/canonical/              Stage01–Stage09 manifests/checksums/status records
provenance/external_restricted/    hashes + access metadata for excluded bytes
provenance/feasibility_audit/      News-decontamination feasibility evidence
provenance/rowlevel_audit/         non-text-bearing identity/audit metadata
provenance/sensitivity/            Rule A / Rule B execution provenance
provenance/release/                Stage10/repackaging release provenance
provenance/archive/                historical/repair notebooks
docs/                              reproducibility, traceability and safety audits
```

## Primary versus sensitivity evidence

The Stage01–Stage09 analysis remains frozen and primary. Rule A and Rule B are post-hoc News-decontamination sensitivity analyses only; neither is presented as a replacement or preferred rule. Aggregate Rule A/B tables and Table S10/S11 evidence remain public under `results/sensitivity/`.

## Data and models

Raw external datasets are not redistributed here. Text-bearing derived caches/splits and model binaries with unresolved public redistribution permission are also excluded. See `data/access/` and `provenance/external_restricted/`.

## Reproducibility and evidence

See:

- `docs/REPRODUCIBILITY.md`
- `docs/EVIDENCE_TRACEABILITY.md`
- `docs/CANONICAL_NOTEBOOK_MAP.md`
- `docs/PUBLIC_REDISTRIBUTION_SAFETY_AUDIT.md`
- `docs/MANUSCRIPT_REPOSITORY_CROSSCHECK_PUBLIC_SAFE.md`

## Publication materials

The manuscript source, manuscript PDFs and editorial records are kept outside this public repository.

`results/publication/journal_presentation_2026-11-15/` holds the journal-presentation versions of the five main and twelve supplementary figures, re-rendered deterministically from the frozen Stage08 figure source data. The canonical Stage08 figures in `results/primary/stage08/` are unchanged and remain the reference artefacts. Wording differences between the two sets (for example `Youtube` → `YouTube`, and BaitBuster "replication" → "external corroboration") are listed in that folder's `README.md`.

## Installation and verification

There is no single environment for every stage. `requirements.txt` is a snapshot of the Stage10 assembly environment, not a lockfile for the historical Stage01–Stage09 runs (see its header).

- **Publication figures:** create a Python 3.13 environment from `scripts/publication_figures/requirements-publication-figures.txt` and run `python scripts/publication_figures/regenerate_and_verify.py`. It writes only to a temporary directory and checks source hashes, exact values and output hashes (see `scripts/publication_figures/README.md`).
- **File integrity:** from the repository root run `sha256sum -c provenance/PUBLIC_REPOSITORY_SHA256SUMS.txt` (the manifest does not list itself). Line endings are pinned in `.gitattributes`, so a fresh clone verifies on Windows, macOS and Linux.
- **Notebooks:** see `docs/REPRODUCIBILITY.md`; some stages need restricted inputs listed in `provenance/external_restricted/`.

## Citation and license

`CITATION.cff` is intentionally omitted pending author approval of the final repository identity/URL. No repository-wide `LICENSE` is fabricated; see `docs/LICENSE_STATUS.md`.

## Integrity

Public-safety repackaging removed restricted bytes only. The retained scientific notebooks, aggregate results, figures/tables, and scientific provenance were not recomputed or numerically altered.
