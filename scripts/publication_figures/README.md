# Publication figure generators (journal presentation, 2026-11-15)

Deterministic generators and QA for `results/publication/journal_presentation_2026-11-15/`. They re-render
figures from the frozen Stage08 figure source data and canonical result records. They do **not** run any
notebook, model, inference, bootstrap or statistical computation, and they do not write to Stage08 or Stage10.

| File | Role |
|---|---|
| `swfig_common.py` | shared style (8 pt minimum at print width, colour-blind-safe palettes), hash-checked CSV loading, text-collision QA run on every export |
| `make_fig1_v3.py` | Figure 1 study-design schematic; writes `FIG01_v3_SOURCE_MANIFEST.{json,csv}` mapping every displayed element to its source |
| `make_main_figures.py` | Figures 2–5 |
| `make_supp_figures.py` | Supplementary Figures S1–S10, with built-in exact-value parity checks |
| `check_parity.py` | independent exact-value parity test for Figures 2–5 (33 checks) |
| `qa_marker_clearance.py` | Figure 1 marker-versus-text clearance check |
| `publication_figure_map.py` | figure list, output names, canonical Stage08 counterparts, frozen sources, documented wording differences |
| `regenerate_and_verify.py` | one-command regeneration into a temporary directory and full verification (`--write-manifest` rewrites the manifests) |
| `requirements-publication-figures.txt` | pinned environment (matplotlib 3.10.0, pandas 2.3.3, Python 3.13.3) |

```bash
python -m venv .venv && .venv/bin/pip install -r scripts/publication_figures/requirements-publication-figures.txt
python scripts/publication_figures/regenerate_and_verify.py
```

Generators write to `$SWFIG_OUT` (default `build/publication_figures/`, git-ignored). Byte-identical output
requires the pinned versions and the Arial font; without Arial, matplotlib falls back to DejaVu Sans, which
changes the file bytes but not the plotted data.

Provenance notes: `make_fig1_v3.py` lists two manuscript sections as descriptive sources for equation and role
wording. The manuscript source is private and not distributed here, so those entries carry no hash. The
corresponding facts are also asserted against public records: `P0_AUDIT_SUMMARY.md`,
`CONSTRUCT_ALIGNMENT_MATRIX.csv` and `SPLIT_FIREWALL.json`.
