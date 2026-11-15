# Journal-presentation figures (2026-11-15)

Publication-specific re-renders of the manuscript figures (5 main, 12 supplementary; 17 figures × PDF, SVG,
PNG = 51 files). They are **presentation assets**, not new scientific results.

- **Canonical scientific record unchanged.** The canonical Stage08 figures (`results/primary/stage08/figures/`),
  Stage08 figure source data, Stage08 provenance and the Stage10 release records are untouched and remain
  authoritative. This directory is additive.
- **No recomputation.** Every plotted value is read from the frozen Stage08 figure source CSVs
  (`results/primary/stage08/figure_source_data/`, hashes checked against `FIGURE_SOURCE_DATA_INDEX.json`).
  Every status and count is read from stored fields. Figure 1 is a schematic study-design figure whose
  displayed facts are traced to upstream Stage01–Stage10 records in `FIG01_v3_SOURCE_MANIFEST.{json,csv}`; its
  central contrast-of-contrasts panel carries no values.
- **Manifest.** `PUBLICATION_FIGURE_MANIFEST.json` lists each file's SHA-256, generator, frozen sources (with
  hashes) and canonical Stage08 counterpart.
- **Regeneration.** `python scripts/publication_figures/regenerate_and_verify.py` regenerates all 51 files into a
  temporary directory and verifies byte identity, value parity, text collisions, marker clearance and
  minimum text size (≥ 8 pt at print width). Byte identity requires the pinned environment in
  `scripts/publication_figures/requirements-publication-figures.txt` and the Arial font.

## Known wording differences from canonical Stage08 (documented, intentionally not changed in Stage08)

| Canonical Stage08 | Canonical wording | Publication figure | Publication wording |
|---|---|---|---|
| `FIG05_H3_H4_BRIDGE` | `Youtube` | `mainfig_05_h3_h4_bridge_v2` | `YouTube` |
| `SUPP_BAITBUSTER_H3` | "independent-collection H3 replication" | `suppfig_S4_baitbuster_h3_v2` | "external corroboration from a related construct, not a construct-equivalent replication" |

The scientific role of BaitBuster in the manuscript is external corroboration from a related construct.

## Mapping (publication file → canonical Stage08 counterpart)

| Publication | Canonical Stage08 |
|---|---|
| `figures/main/mainfig_01_study_design_v3` | `FIG01_STUDY_DESIGN` |
| `figures/main/mainfig_02_h1_h2_v2` | `FIG02_H1_H2` |
| `figures/main/mainfig_03_h3_v2` | `FIG03_H3_CUE_TRANSPORT` |
| `figures/main/mainfig_04_h4_v2` | `FIG04_H4_FIXED_BUDGET` |
| `figures/main/mainfig_05_h3_h4_bridge_v2` | `FIG05_H3_H4_BRIDGE` |
| `figures/supplementary/suppfig_S1_h1_h2_model_seed_v2` | `SUPP_H1_H2_MODEL_SEED` |
| `figures/supplementary/suppfig_S2_h3_sparsity_v2` | `SUPP_H3_SPARSITY` |
| `figures/supplementary/suppfig_S3a_h3p_curiosity_v2` | `SUPP_H3P_curiosity_information_withholding` |
| `figures/supplementary/suppfig_S3b_h3p_direct_address_v2` | `SUPP_H3P_direct_address_imperative` |
| `figures/supplementary/suppfig_S3c_h3p_punctuation_v2` | `SUPP_H3P_punctuation_intensity` |
| `figures/supplementary/suppfig_S4_baitbuster_h3_v2` | `SUPP_BAITBUSTER_H3` |
| `figures/supplementary/suppfig_S5_h4_05pct_v2` | `SUPP_H4_05PCT` |
| `figures/supplementary/suppfig_S6_h4_20pct_v2` | `SUPP_H4_20PCT` |
| `figures/supplementary/suppfig_S7_h4_model_seed_v2` | `SUPP_H4_MODEL_SEED` |
| `figures/supplementary/suppfig_S8_risk_coverage_v2` | `SUPP_RISK_COVERAGE` |
| `figures/supplementary/suppfig_S9_bridge_loo_v2` | `SUPP_BRIDGE_LOO` |
| `figures/supplementary/suppfig_S10_prevalence_structural_sensitivity_v2` | `SUPP_SENSITIVITY` |

The PDF metadata field `Creator` reads "SWARABYANJAN figure_workbench prototypes (presentation only)", the name
of the authoring workbench. It is retained so that these files stay byte-identical to the approved manuscript
figures.
