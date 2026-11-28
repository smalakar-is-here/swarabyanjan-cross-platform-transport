# SCOPE execution log

Dates are UTC. Every experiment listed here has an entry in `EXPERIMENT_REGISTRY.csv` and a `MANIFEST.json` in `results/scope_v1/<experiment>/`.

## Phase 0. Inputs and integrity (2026-11-26)

- Restored the restricted artefacts to a private, git-ignored directory and checked each against its recorded SHA-256 (E0). 22 of 26 restricted artefacts and 23 of 25 raw files verified. The four missing restricted files are post-hoc Rule A/B sensitivity caches, and the two missing raw files (BanFakeNews-2.0 full set, BanglaBait unlabelled data) are not used by any analysis.
- Recomputed the R6 H1 calibrated AUROC and NLL for all 120 model × seed × domain cells from the restored predictions. All matched (AUROC to 1e-9, NLL to 1e-6); this is a test.

## Phase 1. Phenomenon (E1–E7, E13, E14; local CPU)

| Experiment | Question | Outcome |
|---|---|---|
| E1 | Do the two annotation regimes agree on shared titles? | 1,083 anchor classes. κ = 0.13 at the published cut, 0.40 at level 2. Attribution failure does not predict the source label; trigger level does |
| E2 | Do cue associations differ by platform or by regime? | On YouTube the regime contrast is about 3× the platform contrast. On news the regime contrast is smaller, so the claim is restricted to YouTube |
| E3 | What do unlabelled prevalence estimators recover? | PCC matches the aligned prevalence, not the published one. BBSE and MLLS are less accurate |
| E4 | Decomposition of the frozen-model risk gap | Construct and prior dominate for the text models on news, Facebook and Twitter. The residual is largest on YouTube, and the EBM keeps a large Facebook residual |
| E5 | Calibration and prior adaptation | Reported in the supplement |
| E7 | Cue extractor v1.0 vs v1.1 | Cue presence changes for at most 4.6% of items in any cell |
| E13 | Few-cluster test size (simulation) | Pairs bootstrap over-rejects (0.11 at 10 clusters); the wild cluster restricted bootstrap is near nominal |
| E14 | Are the labels cluster labels? | BaitBuster: 99% of label variance lies between channels |

## Phase 2. Training benchmark (E10, E10T, E16, E17)

- **E10, TF-IDF LOEO (CPU).** Source-only, pooled ERM, batch Group DRO, an in-domain reference and a majority baseline, under both label definitions, with 3 seeds. Training excludes every title that occurs in the held-out cell.
- **E10T, transformer LOEO (Kaggle GPU, Account01).** Pooled ERM for BanglaBERT (30 runs, 1.86 GPU-h of training time) and XLM-R (30 runs, 5.35 GPU-h). Group DRO for BanglaBERT and XLM-R under both labels (w01–w04; for BanglaBERT Group DRO weights concentrated on BanglaBait news and BanglaClick Facebook, about 0.48 and 0.51). Predictions are scored locally with the same metric code as every other method.
- **E16, shift-correction baselines (CPU).** Prior correction to the MLLS estimate, prior correction to the oracle prevalence (reference), and importance weighting by a cross-fitted domain-classifier density ratio, for source and pooled training sets.
- **E17.** Paired cluster-bootstrap contrasts between protocols (1,000 draws, Holm correction). First run gave undefined intervals for every BaitBuster contrast: its labels are channel labels, so some resamples of its 13 channels contain one class and AUROC is undefined, and one undefined draw made the percentile interval undefined. Such draws are now dropped and the number of valid draws is reported (990–994 of 1,000 for BaitBuster).
- **Transformer Group DRO and source-only units** (`experiments/kaggle/WORKER_UNITS.md`). These were run through a queue on Account01 as GPU slots and quota allowed, and are available to team members to launch from their own accounts. The registry records which units completed.

### Group DRO audit (before any GPU run)

1. The first worker build failed static checks. Placeholder substitution put JSON inside a double-quoted Python string. This was fixed and is covered by `test_kaggle_worker_units_render_and_parse`.
2. Minibatch–group alignment originally relied on shared state between two generators. It was replaced by passing each batch's item indices explicitly.
3. The objective was moved between markers in the kernel template and unit-tested against a hand computation. The tests cover group weights after the exponentiated-gradient step, an absent group, the loss value, and per-item gradients q_g / n_g (`test_group_dro_objective_matches_hand_computation`).
4. CPU probes on Kaggle (no GPU quota): Group DRO and environment-balanced ERM learned a ranking close to pooled ERM (test AUROC 0.705 vs 0.718; logit correlation 0.91). The logit scale was smaller, and temperature-only calibration flattened their probabilities. This is the R6 calibration lock applied to all protocols alike and is reported, not tuned away.

## Phase 3. Validation of the method (E15, E18)

- **E15, synthetic recovery (known ground truth).** This experiment found a defect in the code. Soft-label Brier omitted the label-variance term q(1−q), which biased soft-label Brier downward by up to 0.2. `metrics.brier` now computes the expected Brier score. Hard-label results are unchanged by construction, and E4/E5/E11 were rerun; NLL and AUROC were identical and only soft-label Brier rows changed. A second defect: in one simulation cell, the Nelder–Mead calibration failed to converge. It was replaced by an exact logistic fit and E15 was rerun.
- **E18, real-data proxy check on anchors.** This uses models that never saw the anchor titles. Every target-side label understates AUROC against the source label, and the hard aligned label is the least biased. The score predicts the source label beyond the rubric level, so the condition under which soft linking is exact fails on these data.
- **Anchor representativeness (Assumption 1).** Models that never saw the anchors (TF-IDF, BanglaBERT and XLM-R, retrained leave-one-environment-out) still score anchors 0.08–0.17 higher than non-anchors within rubric levels 0, 1 and 3 (level 2: +0.02). Memorisation therefore does not explain the gap, and the assumption is reported as violated.

## Compute

- **GPU.** The Kaggle login configured on the execution machine, recorded as Account01. Kaggle allows two concurrent batch GPU sessions per account; a third push was refused with "Maximum batch GPU session count of 2 reached". A queue (`scripts/scope_gpu_queue.py`) refilled the slots until weekly quota fell below 2 GPU-h. The team's other Kaggle accounts were not used from this machine. GPU work is packaged as independent units that each member can launch under their own account.
- **No H200** machine was available.
- **CPU.** Local, 8 cores. Long jobs ran in the background in parallel.

## Negative and null results (reported, not hidden)

- PPI and active PPI gave no variance reduction in the audit.
- MLLS prior correction increased log loss in most LOEO cells.
- Importance weighting gave no consistent AUROC gain; its effective sample size averaged 32% (pooled) and 40% (source) of nominal, with a range of 9–62% across cells.
- Pooled ERM and Group DRO did not beat source-only training on news or Facebook under the aligned label.
- In-domain cross-fitting is below chance on BaitBuster YouTube. Its labels are channel labels, and grouped folds hold out whole channels, so that result is not a ceiling.
- Soft linking, first planned as a sensitivity analysis, is the most biased proxy on real anchors.

## Not done

- Human re-annotation under one guideline. This needs annotators, ethics review and funding.
- A large-language-model zero-shot baseline.
- Transformer source-only units w05–w08 (BanglaBERT and XLM-R, both labels) did not fit in the weekly quota. The queue stopped at 1.23 GPU-h remaining on 2026-11-27 02:31 UTC; the units are packaged for collaborators and recorded as not run in the registry.

## Phase 4. Score-conditional linking and independent validation (2026-11-27)

- **Audit at start.** Branch `research/scope-upgrade-2027` at `e735fb9`, equal to the remote. Kaggle quota on Account01: 0.06 GPU-h left, reset 2026-11-28T00:00Z.
- **GPU plan corrected.** w06 and w08 are duplicates of w05 and w07 (source-only data carry no regime-B label; TF-IDF source-only predictions are bit-identical across labels). They are marked NOT_NEEDED, which halves the outstanding GPU work. The kernel template now also writes anchor-blind cross-cell scores (`xcell_*`) for the news-held-out source-only run. A waiter on the coordinator machine launches w05 and w07 through `scripts/scope_gpu_queue.py` once quota is back (log: `$SCOPE_PRIVATE_DATA/kaggle_out/queue_w05_w07.jsonl`).
- **E19, score-conditional linking (new method).** The link P(Y_A | L, f(x)) is fitted on anchors with a scorer that never saw them. It is exact by the tower rule and identified on news under conditional representativeness (missing at random given level and score). That condition tolerates the documented violation of anchor representativeness. Transport to other platforms is handled as a gamma-sensitivity range, not assumed. Validation: cross-fitting on anchors (V1), selection stress tests with known truth (V2) and the anchor budget (V3). Application: the anchor-blind TF-IDF source model on all regime-B cells, with a joint cluster bootstrap. Its news predictions reproduce E10 exactly (max abs diff 0).
- **Failure mode reported.** When anchor selection depends on Y_A itself, SCL keeps AUROC but its log-loss and prevalence estimates are biased (V2, delta_label = 1).
- **Independent check** (`scripts/scope_independent_check.py`, no `scope` import, raw files only, own normalisation and platform rule). It reproduces 1,083 anchor classes, kappa 0.1326 / 0.4034, prevalence 0.346 vs 0.354, and the per-platform counts and prevalences of Table 1. Conflicts at the published cut: 564 of 1,083 (52.1%). The earlier figure of 556 / 1,058 is not reproduced by any rule tested (strict, loose, train-split only: 540 / 1,053). It is superseded and not used.

## 2026-11-27 E10 w05/w07 re-dispatch

- 07:58Z w05, w07 pushed on Account08 (v1), dataset_sources `smalakarishere/scope-training-pool-v1`. Earlier Account02/Account04 attempts failed: dataset_sources named a path those accounts could not read.
- 08:06Z duplicate w05, w07 pushed on Account01 (v1) at the researcher's request, as redundancy. Only one run per unit will be scored; the other is recorded as a replicate and compared for consistency.
- w05 replicate (Account01) vs w05 (Account08): all 18 output files agree; max |Δp_calibrated| = 0 at 4 d.p., logit correlation 1.0000. The fine-tuning pipeline is reproducible across independent Kaggle GPU sessions. The replicate is stored under private `kaggle_out/replicates/Account01/` (SHA-256 manifest) and is not scored separately.
- w07 replicate (Account01) vs w07 (Account08): all 18 files agree (max |Δp_calibrated| = 0, logit correlation 1.0). XLM-R fine-tuning is also reproducible across sessions. Stored under private `kaggle_out/replicates/Account01/`, not scored separately.
- 12:40Z w11 (BanglaBERT source-only, news fold only) pushed on Account08 and w12 (XLM-R, same) on Account01, v1, git c4d3f61, dataset_sources `smalakarishere/scope-training-pool-v1`. Purpose: E20 source reference. The kernel now also writes `srcval_*`, the same model's scores on its cluster-grouped 10% BanglaBait validation split, so the transformer identified sets can be compared with that model's own held-out source AUROC. No new data was transferred. Check: `xcell_*` must equal w05/w07 (deterministic fine-tuning, C33).
- w11 and w12 COMPLETE and downloaded. All 6 `xcell_` files per unit equal the w05/w07 originals (max |Δlogit| = 0; `results/scope_v1/REPLICATION/replication_check.json`, written by `scripts/scope_replication_check.py`). The 3 `srcval_` files per unit were copied into the w05/w07 result directories and both manifests refreshed; the rerun is archived under private `kaggle_out/replicates/srcval_rerun/` with a SHA-256 manifest. Source held-out AUROC (validation split): BanglaBERT 0.851/0.861/0.860, XLM-R 0.857/0.862/0.850 (seeds 137/2718/42). E20 rerun with transformer source references.
- 2026-11-27 evening. Review round 1 (astar-review-loop) analyses. News clusters changed from article URL (943) to publisher domain (53) for all inference (`data.publisher_cluster`, `run.inference_pool`); training splits unchanged, so no model was refit. Reruns on publisher clusters: E0 E1 E2 E4 E14 E17 E19 E20 E23C. New: E23 (ablations), E24 (robustness), E4T (decomposition under trigger label), E21S (misspecified working link). A first E24/E21S batch ran out of memory with five concurrent CPU jobs and was rerun sequentially. E19 failed for transformer scorers (source labels of training-pool rows missing) and was fixed by extending `src_y` as in E20/E24.
- 2026-11-28. All reruns complete (E19 fixed and rerun; E23C, E24, E21S, E21 rerun sequentially). Tests: 36 passed, 2 skipped (new: fit_bias recovery). Paper assets regenerated into the private CAS manuscript folder.
- 2026-11-28. Review round 2 fixes. E25 (`scripts/scope_media_audit.py`): 1,808 of 2,461 BanglaClick Facebook/Twitter items carry an image, video or link card; the CONTENT field is empty for all 4,961 items. Holm-family wording unified with `run.py` (within each metric, all paired contrasts). Paper assets regenerated.
- 2026-11-28. Review round 3. E26 placebo decomposition (run.py E26, 457 s): beyond a prevalence-matched permuted label, the aligned label explains 0.42-0.51 of the news log-loss gap, 0.26-0.63 on Facebook and 0.16-0.28 on Twitter under L>=2 (text models); every observed joint component exceeds all placebo draws. Non-anchor news SCL reported over gamma (BanglaBERT 0.634 at gamma 0 to 0.874 at gamma 1). Skill-scale residual now computed from each cell's own aligned prevalence (fills non-anchor news). Paper assets regenerated.
