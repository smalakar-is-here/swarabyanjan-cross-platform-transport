# Hand-off: the two outstanding GPU units (w05, w07)

Status 2026-11-27: these are the only GPU units still needed. w06 and w08 are not needed: source-only training
never sees a regime-B label, so they would reproduce w05 and w07 exactly (see `WORKER_UNITS.md`).

| Unit | Model | Upper estimate of GPU time (P100), from the pooled-ERM kernels | Assign to |
|---|---|---|---|
| w05 | BanglaBERT, source only | about 1 h | collaborator 1 (own Kaggle account) |
| w07 | XLM-R, source only | about 2.5 h | collaborator 2 (own Kaggle account) |

Each collaborator runs one unit on their own machine, under their own Kaggle login. Nobody sends a token,
password or browser session to anyone. If no collaborator is available, Account01 runs both automatically
after its quota reset (2026-11-28T00:00Z); tell the coordinator when you push, so the same unit is not run twice.

## Before you start (data holder)

Each collaborator needs read access to the private Kaggle dataset `scope-training-pool-v1`. It holds restricted
third-party text, so the data holder shares it only with team members whom the dataset licences cover.
The researcher confirms that before sharing. Sharing it is a data-transfer decision, not a technical step.

## Commands (collaborator, PowerShell, repository root, branch `research/scope-upgrade-2027`)

```
git pull
python scripts/scope_make_worker_packages.py build
```

Edit `experiments/kaggle/build/workers/<unit folder>/kernel-metadata.json`: replace `OWNER` with your Kaggle
username and `DATASET_OWNER` with the data holder's username. Unit folders:

- w05: `scope-e10-w05-banglabert-sourceonly-r6`
- w07: `scope-e10-w07-xlmr-sourceonly-r6`

```
kaggle kernels push -p experiments/kaggle/build/workers/<unit folder>
kaggle kernels status <your username>/<unit folder>
kaggle kernels output <your username>/<unit folder> -p <output folder>
```

## Return

Send the data holder every `pred_*.parquet`, `xcell_*.parquet` and `runlog_*.json` file (15 + 3 + 1 per unit).
They contain row ids and probabilities only, no text. The data holder puts them in
`$SCOPE_PRIVATE_DATA/kaggle_out/full/<model>/<unit folder>/` and runs `python -m scope.run E10T E17 E18 E19`.
The registry records the unit, kernel slug and account alias (never the username).
