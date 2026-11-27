# Reproducing the SCOPE Successor Study

This guide applies only to the SCOPE successor branch. R6 historical
reproduction is documented separately in `../REPRODUCIBILITY.md`.

## 1. Verify the checkout

Run commands from `github/scope_upgrade_checkout/`. Verify the branch,
working tree and expected source paths before execution.

``` powershell
git status --short --branch
git log -5 --oneline --decorate
Test-Path .\src\scope
Test-Path .\tests
Test-Path .\scripts\scope_make_worker_packages.py
Test-Path .\experiments\kaggle\WORKER_UNITS.md
```

If an expected path is missing, stop that specific workflow and record
the mismatch. Do not fabricate a path.

## 2. Environment (Windows PowerShell)

``` powershell
py -3.13 -m venv .venv-scope
.\.venv-scope\Scripts\python.exe -m pip install -r requirements-scope.txt
```

Use the versions pinned by the current requirements/lock file. Record
the actual installed versions. Do not assume the environment matches the
R6 Kaggle environment.

## 3. Private data location

The canonical private-data location is the G-drive workspace's
`private_data\scope\` directory, outside Git. When running from the
SCOPE repository root:

``` powershell
$workspaceRoot = (Resolve-Path (Join-Path (Get-Location) "..\..")).Path
$env:SCOPE_PRIVATE_DATA = Join-Path $workspaceRoot "private_data\scope"
New-Item -ItemType Directory -Force -Path $env:SCOPE_PRIVATE_DATA | Out-Null
```

Restore only authorised files. Verify each file against its recorded
SHA-256 manifest before analysis.

**External-transfer gate:** uploading restricted third-party text to a
private Kaggle dataset is still an external transfer. Do not upload
until the dataset terms and transfer permission are verified. Keep
credentials outside the workspace repository, manifests and ZIP
archives.

## 4. CPU-compatible experiments

Verify the experiment names against the current runner before launching.
The previously documented CPU group is:

``` powershell
$env:PYTHONPATH = "src"
.\.venv-scope\Scripts\python.exe -m scope.run E0 E1 E2 E3 E4 E5 E7 E11 E13
.\.venv-scope\Scripts\python.exe -m scope.run E10
```

Run independent CPU tasks concurrently only when they do not compete for
the same output paths or mutate shared state. Use separate logs and
experiment IDs.

## 5. GPU worker units

Read `experiments/kaggle/WORKER_UNITS.md`. Build worker packages using
the documented generator:

``` powershell
.\.venv-scope\Scripts\python.exe scripts\scope_make_worker_packages.py build
```

Each independently authorised team member launches their assigned unit
from their own authenticated Kaggle session and only if platform terms
permit. Do not share tokens, pool quotas or use multiple accounts
contrary to platform rules.

Typical commands, run by the authorised account owner:

``` powershell
kaggle kernels push -p experiments\kaggle\build\workers\<unit-folder>
kaggle kernels status <authorised-owner>/<kernel-slug>
kaggle kernels output <authorised-owner>/<kernel-slug> -p <local-output-folder>
```

Do not put usernames or credentials in the shared registry. Record
account aliases only. The data holder validates and imports returned
outputs into the canonical private workspace, verifies hashes, scores
them with the same metric code and updates
`docs/scope/EXPERIMENT_REGISTRY.csv`.

## 6. Scoring, assets and tests

After all required outputs are present and validated, run the documented
scoring stage (including `E10T` only if its prerequisites are
satisfied). Generate paper assets from verified result files.

``` powershell
$env:PYTHONPATH = "src"
.\.venv-scope\Scripts\python.exe -m pytest -q tests
```

Do not run asset generation against incomplete results. Every number in
the manuscript must map to a result file and manifest.

## 7. Output and status semantics

Record distinct statuses: planned, packaged, submitted, running,
completed, downloaded, scored, validated, failed or blocked. A kernel
submission is not a completed run. A completed run is not validated
until output hashes, configuration, metrics and expected row counts
pass.

Store versioned outputs under the current SCOPE results directory
(currently documented as `results/scope_v1/`, verify the
implementation). Never overwrite R6 results.
