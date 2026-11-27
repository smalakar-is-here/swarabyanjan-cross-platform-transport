# Claim-to-evidence map (successor manuscript; kept outside the public repository)

Every number in the manuscript prose comes from a macro in `numbers.tex` written by the asset script, and every table and figure comes from `scripts/scope_make_paper_assets.py`. Both are generated from the files below; no value is typed by hand.

Status meanings:
- **NEW**: newly executed in this work.
- **R6**: previously completed and reused unchanged.
- **REPRO**: R6 value independently reproduced.

| # | Claim | Evidence file(s) | Status | Support |
|---|---|---|---|---|
| C1 | R6 BanglaClick label equals 1[level ≥ 1] on all 4,961 items | `src/scope/data.py::load_predictions` assertion; `tests/test_scope.py` | NEW | yes |
| C2 | R6 H1 calibrated AUROC/NLL reproduced (120/120) | `tests/test_scope.py::test_r6_label_is_level_at_least_one_and_reproduces_h1_auroc` | REPRO | yes |
| C3 | 1,083 anchor classes; κ 0.13 at cut 1 vs 0.40 at cut 2; prevalence matched at cut 2 | `results/scope_v1/E1/threshold_agreement.csv`, `bootstrap_summary.json` | NEW | yes |
| C4 | Attribution failure does not predict BanglaBait label; trigger level does | `E1/rubric_logit_HC1.csv` | NEW | yes |
| C5 | Unlabelled PCC estimates match aligned, not published, prevalence | `E3/prevalence_estimates_summary.csv` | NEW | yes (PCC). BBSE/MLLS are less accurate on Facebook, Twitter and YouTube. The claim is restricted to PCC |
| C6 | Aligned label raises AUROC on News/Facebook/Twitter, lowers it on YouTube | `E5/calibration_adaptation_summary.csv` | NEW | yes |
| C7 | Construct + prior explain most of the NLL gap for text models on News/FB/TW; YouTube residual largest | `E4/decomposition_summary.csv`, `E4/decomposition_cluster_bootstrap_ci.csv` | NEW | partial: the EBM on Facebook keeps a large residual, and the text says so |
| C8 | Construct alignment lowers platform spread of cue log-ORs for 6/8 cues | `E2/crossed_platform_regime_contrasts.csv` | NEW | yes (point estimates; no CI on the SD) |
| C9 | On YouTube the regime contrast (C−B, mean abs 2.43) is about 3× the mean platform contrast within B (0.84) | same | NEW | yes for YouTube only. B−A on News (0.51) is smaller than the platform contrasts; the text was narrowed accordingly |
| C10 | Pairs cluster bootstrap over-rejects at 10–40 clusters; WCR near nominal | `E13/few_cluster_size.csv` | NEW (simulation) | yes, under the stated data-generating process |
| C11 | Cue extractor fix changes cue presence ≤ 4.6% of items | `E7/cue_presence_change_by_cell.csv` | NEW | yes |
| C12 | Audit: 25 labels decide correctly under the R6 label; aligned-label decisions are near-boundary | `E11/audit_results.csv` | NEW | yes. PPI gives no gain, and this is reported as a negative result |
| C13 | Anchor representativeness (Assumption 1) is violated, and not by memorisation | `E1/anchor_representativeness.csv`, `E10/anchor_representativeness_dedup_models.csv` | NEW | yes: LOEO models that never saw the anchors still score them 0.08–0.17 higher at levels 0, 1 and 3 (level 2: +0.02; TF-IDF, BanglaBERT and XLM-R). The assumption is reported as violated |
| C14 | The label definition moves LOEO AUROC more than any training or shift-correction protocol does (TF-IDF) | `E10/loeo_tfidf_summary.csv`, `E16/shift_baselines_summary.csv` | NEW | yes: the aligned-minus-published gain is 0.04–0.18 on news, Facebook and Twitter; the median spread across five protocols is 0.051 |
| C15 | No baseline beats source-only training consistently; pooled ERM and IW (pooled) are worse on Facebook | `E17/protocol_contrasts_cluster_bootstrap.csv` | NEW | yes, with the few-cluster caveat for Facebook (26), YouTube-B (10) and BaitBuster (13) |
| C16 | Transformer Group DRO vs pooled ERM | `E17/protocol_contrasts_cluster_bootstrap.csv`, `E10/loeo_transformer_summary.csv` | NEW | partial: BanglaBERT +0.017 AUROC on news (published label), no other significant AUROC change; XLM-R loses 0.16 AUROC on BaitBuster under the published label (13 clusters, indicative), no significant AUROC change under the aligned label; both families lower log loss on Facebook and raise it on Twitter (published label) and lower it on BanglaClick YouTube (aligned label). Source-only transformer units not run, so no comparison with source-only |
| C17 | MLLS prior correction increases log loss in most LOEO cells; importance weighting gives no consistent gain | `E16/shift_baselines_summary.csv`, `E17/...` | NEW | yes (9 of 10 MLLS cells worse; IW ESS 9–62% of nominal) |
| C18 | Soft-label risk equals source-construct risk iff Y_A ⟂ f given L (Proposition, soft linking) | proof in manuscript; `E15/synthetic_recovery_summary.csv`; `tests/test_scope.py` | NEW | yes (proof; synthetic recovery at ω = 1) |
| C19 | On real anchors the soft-linking condition fails and the hard aligned label is the least biased proxy | `E18/anchor_proxy_check_summary.csv`, `E18/score_beyond_level_test.csv` | NEW | yes for the models and anchors tested; all proxies understate AUROC against Y_A |

Claims not made:
- No claim of a platform × regime interaction (not identifiable).
- No claim that the aligned label is "correct".
- No claim separating covariate from concept shift inside the residual.
- No causal claims.
