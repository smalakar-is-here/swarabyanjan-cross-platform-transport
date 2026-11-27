# Transformer work units for team members

The outstanding transformer runs of E10 are split into eight independent units. Each unit fine-tunes one
model under one training protocol and one regime-B label definition, for 5 held-out target cells × 3
seeds (15 fine-tunings, about the size of the pooled-ERM kernels already run). Units share nothing at run
time, so any number can run at once, each on a different team member's own Kaggle account.

| Unit | Model | Protocol | Label |
|---|---|---|---|
| w01 | BanglaBERT | Group DRO | r6 |
| w02 | BanglaBERT | Group DRO | aligned |
| w03 | XLM-R | Group DRO | r6 |
| w04 | XLM-R | Group DRO | aligned |
| w05 | BanglaBERT | source only | r6 |
| w06 | BanglaBERT | source only | aligned |
| w07 | XLM-R | source only | r6 |
| w08 | XLM-R | source only | aligned |

Pooled ERM for both models is already covered by the `scope-e10-*-full` kernels.

**Source-only units: w05 and w07 only (2026-11-27).** Under `source_only` the training and temperature-validation
items are all BanglaBait news, where `y_r6 == y_al`, so w06 is the same model as w05 and w08 the same as w07
(verified bit for bit on the TF-IDF source-only predictions). Only w05 and w07 are run; E10T and E17 score each
source-only prediction under both labels. The news-held-out run of each unit also writes
`xcell_<model>_<label>_heldout-banglaclick-news_s<seed>_source_only.parquet`: scores of that anchor-blind model on
every other non-source cell, used by E19 (score-conditional linking). Return these files with the `pred_*` files.

## Prerequisites (once per team member)

1. Your own Kaggle account with an API token. The token is delegated
   to the coordinator via the out-of-tree token file (never stored in
   this repository). Team members are listed in the coordinator's
   scheduling registry.
2. Read access to the private dataset `scope-training-pool-v1`, shared
   by the data holder. It holds restricted third-party text, so it
   stays private and is shared only with team members covered by the
   dataset licences.

## Launch a unit

Units are dispatched by the coordinator via `scripts/scope_multi_queue.py`.
From the repository root:

    python scripts/scope_multi_queue.py discover
    python scripts/scope_multi_queue.py check          # dataset access per account; no quota
    python scripts/scope_multi_queue.py queue w05 w07

Run `check` before spending GPU quota. A share made in the Kaggle UI
must name the account's Kaggle username. An account showing `NO_ACCESS`
gets a kernel with an empty `/kaggle/input` even though the push
succeeds. `push()` skips such accounts, and the kernel itself now stops
with `POOL_NOT_FOUND` and a listing of what was mounted.

The coordinator builds each unit's kernel folder under the target
account's credentials, sets the correct `dataset_sources` entry
(`<data-holder>/scope-training-pool-v1`), pushes the kernel, polls
`kaggle kernels status` on each account, downloads completed outputs
to `$SCOPE_PRIVATE_DATA/kaggle_out/full/<model>/<slug>/`, and appends
one line per event to `$SCOPE_PRIVATE_DATA/kaggle_out/multi_queue.jsonl`.

The units running today are w05 and w07 only. Every completed run is
scored by `python -m scope.run E10T` with the same metric code as
every other method; the registry entry records the account alias
(never username, never token).

## Group DRO implementation (audited)

Online Group DRO over training environments (Sagawa et al., 2020), step size 0.01:

- Each minibatch carries the item indices it was built from. The environment of each item is read
  through those indices, so labels, texts and groups are aligned by construction.
- Per-environment loss is the mean per-item cross-entropy within the batch (`index_add_` of losses over
  counts).
- Environments absent from a batch get loss 0. Their weight is unchanged before renormalisation, as in
  the reference implementation.
- The update is q_g ← q_g · exp(η ℓ_g) on detached losses, followed by renormalisation. The training
  loss is Σ_g q_g ℓ_g.
- Hyperparameters are otherwise the R6 lock (lr 2e-5, batch 16, weight decay 0.01, warm-up 10%,
  max length 128). Sagawa et al. pair Group DRO with strong ℓ2 regularisation. The R6 lock is kept so
  that protocols differ only in their objective, and this is a stated limitation.
- The final probe output records the environment weights `dro_q` per run, so the weights can be inspected.

## E22 units: regime cut-point model (`latent_cuts`)

| Unit | Model | Protocol | Labels written |
|---|---|---|---|
| w09 | BanglaBERT | latent_cuts | r6 and aligned (one model per target and seed) |
| w10 | XLM-R | latent_cuts | r6 and aligned (one model per target and seed) |

The objective (`latent_loss`, between the OBJECTIVE markers of `run_template.py`) is the E22 likelihood with
BanglaClick levels 0, 1, 2+ (= y_r6 + y_al; level 3 is merged because the uploaded pool carries only the two
binarisations). It is tested against `scope.latent` (`tests/test_scope.py::test_template_latent_loss_matches_cpu_latent_model`).
Cut-points have their own Adam optimiser (lr 1e-2) and are unscaled from the AMP loss scale before each step.
The fitted cuts are recorded per run in `runlog_<model>_latent_cuts.json`. Run a CPU probe first:
`SCOPE_DATASET_OWNER=smalakarishere python scripts/scope_make_worker_packages.py probe w09 <user>`.
