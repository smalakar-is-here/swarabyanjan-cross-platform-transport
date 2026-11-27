# SCOPE validation report

Date: 2026-11-27 (UTC). Scope: code, results and claims of the SCOPE successor study on branch `research/scope-upgrade-2027`.

This audit was carried out by the same agent that wrote the code. It is a structured self-check, not an independent review; an independent audit is still recommended before submission.

## Integrity audit

| Check | Result | Evidence |
|---|---|---|
| A. Ground truth | PASS | Labels come from the released BanglaBait, BanglaClick and BaitBuster files (SHA-256 verified, E0). The aligned label is a deterministic function of the BanglaClick rubric level; no model output is used as a label. |
| B. Same metric code | PASS | Every protocol, TF-IDF and transformer, is scored locally by `src/scope/metrics.py` from saved per-item predictions (`run.py::_gather_predictions`). Kaggle kernels write predictions only. |
| C. Files exist | PASS | Every manuscript number is a macro written by `scripts/scope_make_paper_assets.py` from `results/scope_v1/*`; no number is typed by hand. |
| D. Dead code | PASS | Group DRO weights move (`dro_q` in runlogs), its objective is unit-tested against a hand computation, and DRO predictions differ from pooled ERM (E17 contrasts are non-zero). The prior-correction AUROC contrasts are exactly zero, as they must be. |
| E. Scope | WARN | TF-IDF: all protocols, both labels, 3 seeds. BanglaBERT: pooled ERM and Group DRO (both labels). XLM-R: pooled ERM and Group DRO (both labels). Transformer source-only units (w05–w08) were not run within the weekly quota and are packaged for collaborators. Claims about transformers are limited to the completed runs. |
| F. Evaluation type | PASS | E13 and E15 are simulations and are labelled as such. All other experiments use real labels. E18 measures proxies against the real source label on anchors. |
| G. Leakage | PASS | Every training path removes titles whose normalised hash occurs in the held-out cell (`test_leakage_guards_present`). Temperatures are fitted on a cluster-grouped split of the training pool, and no model selection uses target labels. |

## Defects found and fixed during this work

1. Soft-label Brier omitted the label-variance term q(1−q). Fixed in `metrics.brier`; E4, E5 and E11 were rerun, and only the soft-label Brier rows changed.
2. The Nelder–Mead calibration in the E15 simulation failed to converge in one cell. It was replaced by an exact logistic fit.
3. The worker kernel template substituted JSON inside a double-quoted string. Fixed and covered by a test.
4. Minibatch–group alignment relied on state shared between generators. Replaced by explicit indices.
5. E17 intervals were undefined for BaitBuster, because one-class resamples make AUROC undefined. Such draws are now dropped and counted.
6. The E17 percentile p-values had a resolution of 0.002, too coarse for Holm over about 60 contrasts. Holm now uses a Wald p-value from the bootstrap standard error, and the reference protocols that use target labels (in-domain, oracle prior) are excluded from the family.
7. The registry builder counted every runlog of a model for each of its kernels. It now counts per kernel.

## Tests

`python -m pytest tests -q`: 18 passed.

## Claim gate

See `CLAIM_EVIDENCE_MAP.md`.

- **Supported:** C1–C15, C17–C19.
- **Partial:** C7 (the EBM residual on Facebook) and C16 (transformer Group DRO is compared with pooled ERM only).
- **Not supported, and reported as such:** Assumption 1, anchor representativeness (C13 documents the violation).

## Open items

- Transformer source-only units (w05–w08) depend on GPU quota (resets 2026-11-28 00:00 UTC) or collaborator accounts.
- Contrasts in cells with fewer than 40 clusters (Facebook, BanglaClick YouTube, BaitBuster) are indicative only (E13).
