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
