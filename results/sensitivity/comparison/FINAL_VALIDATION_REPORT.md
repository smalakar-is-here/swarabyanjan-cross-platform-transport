# SWARABYANJAN — Rule A / Rule B isolated rerun and comparison

## Execution status
Both branches completed Stage05 → Stage06 → Stage07 with PASS status. Canonical Stage05/06/07 manifest hashes remained unchanged at their frozen pins.

## Input and branch validation
| Check | Rule A | Rule B |
|---|---:|---:|
| Retained News rows | 456 | 426 |
| Retained News prevalence | 64.69298% | 63.38028% |
| Non-News prediction rows changed | 0 | 0 |
| Source-test cache changed | No | No |
| BaitBuster cache changed | No | No |
| News mask exactly matches locked hash rule | Yes | Yes |

## Global H3
| Metric | Frozen v3.1 | Rule A | Rule B |
|---|---:|---:|---:|
| Status | NON_ESTIMABLE | NON_ESTIMABLE | NON_ESTIMABLE |
| Joint valid replicates | 2682 | 1908 | 1378 |
| Joint invalid replicates | 2318 | 3092 | 3622 |
| Valid-replicate change vs frozen | — | -774 | -1304 |

H3 cell estimability: Frozen 24/32 estimable; Rule A 20/32; Rule B 18/32. Both branches therefore retain the frozen global H3 NON_ESTIMABLE status while losing additional estimable cells on News.

## H1/H2 — News
Rule A: all 20 News model×seed settings remained estimable. Mean absolute changes vs frozen: AUROC 0.029581; relative-NLL 0.097624; balanced-Brier 0.234163; H2 target-minus-source NLL gain 0.025147.
Rule B: all 20 News model×seed settings remained estimable. Mean absolute changes vs frozen: AUROC 0.026033; relative-NLL 0.075575; balanced-Brier 0.229551; H2 target-minus-source NLL gain 0.019625.

## H3 — all 32 BanglaClick cells
Rule A: News max |ΔRD change| 0.292783 (curiosity_information_withholding); max |ΔNLL-gap change| 0.433456 (exaggeration_superlative). Non-News BanglaClick H3 point estimates remain exact; some bootstrap uncertainty/p-values can differ because the canonical routines consume a shared RNG stream.
Rule B: News max |ΔRD change| 0.780500 (curiosity_information_withholding); max |ΔNLL-gap change| 0.505562 (exaggeration_superlative). Non-News BanglaClick H3 point estimates remain exact; some bootstrap uncertainty/p-values can differ because the canonical routines consume a shared RNG stream.

## H3 multiplicity
Rule A family_simes_outcome: 2 family-level Simes p-values are numerically available; remaining families are non-estimable/incomplete under the locked four-platform rule.
Rule B family_simes_outcome: 2 family-level Simes p-values are numerically available; remaining families are non-estimable/incomplete under the locked four-platform rule.
Rule A family_simes_reliability: 2 family-level Simes p-values are numerically available; remaining families are non-estimable/incomplete under the locked four-platform rule.
Rule B family_simes_reliability: 2 family-level Simes p-values are numerically available; remaining families are non-estimable/incomplete under the locked four-platform rule.
family_fdr_outcome_full: Rule A and Rule B both report 2/8 estimable families and status NON_ESTIMABLE_FULL_8_FAMILY_BH.
family_fdr_reliability_full: Rule A and Rule B both report 2/8 estimable families and status NON_ESTIMABLE_FULL_8_FAMILY_BH.

## H4 — primary 10% News
Rule A: all 20 News model×seed primary-10% rows remain estimable. Mean absolute changes vs frozen: lift 0.199126; error capture 0.013523; AURC 0.050924. Non-News BanglaClick point estimates remain unchanged.
Rule B: all 20 News model×seed primary-10% rows remain estimable. Mean absolute changes vs frozen: lift 0.264979; error capture 0.013610; AURC 0.062947. Non-News BanglaClick point estimates remain unchanged.

## H3→H4 bridge
Rule A: all 12 bridge statistic rows remain NON_ESTIMABLE under the locked joint bootstrap. Mean absolute point change vs frozen = 0.034171; invalid joint bootstrap replicates increase by 614 for every bridge statistic because the branch changes the News input universe.
Rule B: all 12 bridge statistic rows remain NON_ESTIMABLE under the locked joint bootstrap. Mean absolute point change vs frozen = 0.048917; invalid joint bootstrap replicates increase by 1258 for every bridge statistic because the branch changes the News input universe.

## Rule A vs Rule B
Direct numerical differences are reported without ranking either branch. Examples of the largest absolute A→B shifts across the News-focused outputs are:

| Quantity | Largest absolute B−A change |
|---|---:|
| H1 News AUROC | 0.011487 |
| H1 News relative-NLL change | 0.074309 |
| H2 News target-minus-source NLL gain | 0.013580 |
| H3 News ΔRD | 0.487717 |
| H3 News ΔNLL-gap | 0.258747 |
| H4 News 10% lift | 0.099222 |
| H4 News 10% error capture | 0.009582 |
| H4 News 10% AURC | 0.024936 |
| H3→H4 bridge point statistic | 0.079179 |

The full row-level A→B tables are included in `A_vs_B_*.csv`. No branch is selected or ranked here.

## Files
- `FINAL_VALIDATION_SUMMARY.json` — machine-readable validation summary.
- `Rule A_*.csv`, `Rule B_*.csv` — full frozen-vs-branch quantitative tables.
- `A_vs_B_*.csv` — direct Rule A vs Rule B comparisons.