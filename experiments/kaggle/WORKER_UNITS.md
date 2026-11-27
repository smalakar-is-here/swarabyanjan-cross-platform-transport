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

## Prerequisites (once per team member)

1. Your own Kaggle account with the API set up on your own machine (`kaggle config view` shows your username).
2. Read access to the private dataset `scope-training-pool-v1`. It holds restricted third-party text, so the
   data holder shares it only with team members covered by the dataset licences. It must stay private.

## Launch a unit

From the repository root, on your own machine:

```
python scripts/scope_make_worker_packages.py build
```

Open `experiments/kaggle/build/workers/<unit folder>/kernel-metadata.json`. Replace `OWNER` with your Kaggle
username and `DATASET_OWNER` with the username that shares the dataset. Then run:

```
kaggle kernels push -p experiments/kaggle/build/workers/<unit folder>
kaggle kernels status <your username>/<unit folder>
```

When the status is complete:

```
kaggle kernels output <your username>/<unit folder> -p <output folder>
```

Send the `pred_*.parquet` and `runlog_*.json` files to the data holder. They go into
`$SCOPE_PRIVATE_DATA/kaggle_out/full/<model>/`, and `python -m scope.run E10T` scores every unit with the
same metric code. Record the unit, the kernel slug and the account alias (not the username) in
`docs/scope/EXPERIMENT_REGISTRY.csv`.

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
