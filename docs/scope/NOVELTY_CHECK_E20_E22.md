# Novelty check: E20 (bounds), E21 (design-based auditing), E22 (regime cut-points)

Date: 2026-11-27. Status: first-pass web search; full-text reading of the closest papers is still required
before any novelty sentence enters the manuscript. Entries marked *title only* were seen in search results and
have not been read.

## E20: sharp bounds on source-construct risk under level-link transport

Closest work found:
- Learning with selectively labeled data from multiple decision-makers, arXiv 2306.07566: sharp partial-identification
  bounds on classification risk under a bounded-label assumption; reduces to Manski bounds with constant bounds.
- Assessing disparate impacts of personalized interventions: identifiability and bounds, arXiv 1906.01552: sharp
  intervals for TPR/TNR under (relaxed) monotone treatment response.
- Background: Manski monotone response / MIV; Molinari, Microeconometrics with Partial Identification, arXiv 2004.11751.

Not found in the search: a sharp AUROC identified set under a known per-stratum label mass, or the reduction
of weighted AUROC to a function linear in the soft labels at fixed total mass. Candidate contribution, pending a
targeted search (keywords: "AUC bounds missing labels", "partial identification ROC").

## E21: model-assisted anchor sampling

Closest work found (the estimator itself is not new):
- Kossen et al., Active Testing: Sample-Efficient Model Evaluation, ICML 2021 (PMLR 139), arXiv 2103.05331.
- Prediction-Powered Active Testing, arXiv 2607.08347 (preprint): LURE + prediction-powered control variate.
- Fisch et al., Stratified prediction-powered inference for hybrid evaluation of language models, NeurIPS 2024 (*title only*).
- Fogliato et al., A framework for efficient model evaluation through stratification, sampling, and estimation, ECCV 2024 (*title only*).
- Classical: Särndal, Swensson & Wretman, Model Assisted Survey Sampling (1992), difference/GREG estimators.

Defensible contribution: the predictor is a *construct link* transported from another platform. The difference
between the design-based and model-only estimates is a test of link transport. AUROC is handled as a smooth
function of two label totals once all scores are known. Not claimed: a new estimator.

## E22: latent intensity with regime-specific cut-points

Closest work found:
- Inferring ground truth from multi-annotator ordinal data (latent continuous perception + annotator thresholds), arXiv 1305.0015.
- Raykar et al., Learning from crowds (JMLR 2010); Rodrigues et al., GP classification with multiple annotators (ICML 2014, PMLR 32).
- Zhou et al., Simple multi-dataset detection, arXiv 2102.13086: unified model with dataset-specific outputs.
- Samejima (1969) graded response model; Kolen & Brennan (2014) test equating (concurrent calibration, common items).

Defensible contribution, if the experiment supports it: cut-points per *annotation regime* (not per annotator),
linked through cross-regime anchor items, used to transport a text classifier across label definitions. It nests
pooled ERM, which is the equal-cut special case. Not claimed: a new model class.

All citations above need DOI / venue verification before use.

## Update 2026-11-27 (second pass, E20)

Targeted search ("partial identification AUC", "AUC bounds unknown labels known proportion") found two closer
papers. Both were read through their arXiv HTML text:
- Fogliato, G'Sell & Chouldechova, Fairness Evaluation in Presence of Biased Noisy Labels, AISTATS 2020
  (arXiv 2003.13808). They derive sharp bounds on AUC, FPR, FNR and PPV when a share alpha of observed negatives are
  hidden positives, attained by placing the hidden positives at the highest or lowest scores. There is no
  monotonicity restriction, the labels are binary, and log loss and Brier score are not covered.
- Claesen, Davis, De Smet & De Moor, Assessing Binary Classifiers Using Only Positive and Unlabeled Data, arXiv
  1504.06837 (2015). They bound ROC and PR curves given one global class fraction beta, with the same rank-extreme
  construction.

Consequence for the manuscript claim: the rank-extreme construction (Prop. 3a) is not new and is now cited as such.
The remaining contribution is narrower:
1. known positive mass per stratum, supplied by a cross-regime link;
2. the monotonicity endpoint (Prop. 3b), which shows that the level-only link is the sharp worst case;
3. soft labels, with log loss and Brier score covered by the same argument;
4. the use of all of this to bound a construct-transported risk.
The manuscript's related work and contribution bullet were rewritten to match.
