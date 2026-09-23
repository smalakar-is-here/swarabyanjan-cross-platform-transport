# Swarabyanjan — Public-Safe Reproducibility Core

Swarabyanjan is a reproducibility and evidence repository for a Bengali clickbait external-validity study of cue–outcome transport, cue-conditioned proper-loss transport, probability reliability, and fixed-budget review behaviour under distribution shift.

## Scientific workflow

The canonical scientific workflow contains **exactly nine notebooks: Stage01–Stage09** in `notebooks/`. Stage10 is **release engineering only** and is kept at `scripts/release/10-release-package-assembly-vfinal.ipynb`.

The implementation is notebook-embedded. The completed forensic inventory found no standalone `.py`, `.sh`, `.bat`, or `.ps1` implementation tree, so this repository does not fabricate a `src/` package.

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

Publication source files are intentionally outside the public research core. The current final manuscript and supplementary PDFs were used for the final cross-consistency audit, but are not bundled here. The current author-hardened LaTeX source package was metadata-visible but its raw source bytes were not available to the build runtime, so it was not reconstructed from older packages.

## Citation and license

`CITATION.cff` is intentionally omitted pending author approval of the final repository identity/URL. No repository-wide `LICENSE` is fabricated; see `docs/LICENSE_STATUS.md`.

## Integrity

Public-safety repackaging removed restricted bytes only. The retained scientific notebooks, aggregate results, figures/tables, and scientific provenance were not recomputed or numerically altered.
