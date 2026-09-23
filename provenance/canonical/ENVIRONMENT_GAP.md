# Environment gap

The canonical notebooks and stage outputs preserve substantial provenance, including Python/runtime and selected package metadata in several stages. However, the historical execution chain does **not** provide a single complete, immutable package lock covering every package and transitive dependency for Stages 03–09.

Therefore:

- `HISTORICALLY_RECORDED_ENVIRONMENT.json` contains only historically captured information.
- `STAGE10_ASSEMBLY_ENVIRONMENT.json` and `requirements.txt` describe the Stage 10 release-assembly runtime only.
- Missing historical package versions are not back-filled from the current environment.
- The Stage 10 `requirements.txt` MUST NOT be claimed to recreate the original Stage 03–09 software environments.
- Model, tokenizer, dataset, manifest, and checksum identities remain separately pinned where the original stages recorded them.

This gap is documentary, not a license to rerun or alter any scientific stage during release assembly.
