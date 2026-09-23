# Model artifacts — license and redistribution status

The 10 Stage03 `.joblib` artifacts that were present in the private/core build are **not included in this public-safe package** because their redistribution permission remained explicitly unresolved at the final push-safety gate.

Their exact former paths, sizes and SHA-256 values are recorded in `../../provenance/external_restricted/RESTRICTED_ARTIFACT_MANIFEST.json`. Their scientific configuration/state identities remain recorded by:

- `../../provenance/canonical/stage03/MODEL_ARTIFACT_MANIFEST.json`
- `../../provenance/canonical/stage03/STAGE03_SHA256SUMS.txt`
- `../../configs/MODEL_CONFIG_LOCK.json`

An authorised researcher can reproduce the Stage03 models from the frozen authorised inputs and verify the expected identity. The public repository does not claim that reproduction alone grants redistribution rights.

Transformer checkpoint packages were already outside this public core; their optional large-artifact identities remain documented by canonical provenance.
