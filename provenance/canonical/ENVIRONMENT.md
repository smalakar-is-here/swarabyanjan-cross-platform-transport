# Environment

## HISTORICALLY_RECORDED_ENVIRONMENT

The release preserves the environment metadata that the canonical stages actually recorded.
Stage 01/02 record Python 3.12.13 in the executed provenance; later stages preserve their own RUN_METADATA / preflight records where available.
No missing historical package version is inferred from the Stage 10 environment.

## STAGE10_ASSEMBLY_ENVIRONMENT

The file `STAGE10_ASSEMBLY_ENVIRONMENT.json` and `requirements.txt` describe only the environment used to assemble this release.
They are **not necessarily identical to the historical scientific stage environments**.
