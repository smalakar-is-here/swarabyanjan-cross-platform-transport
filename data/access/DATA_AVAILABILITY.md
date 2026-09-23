# Data availability — public-safe repository

Raw third-party dataset bytes are not redistributed in this public core.

The public-safe package also excludes frozen derived artifacts that contain literal third-party text or row-level source bytes where public redistribution permission remains unresolved. These include the clean BanglaBait text splits, Stage05 text-bearing prediction/perturbation caches, Rule A/B text-bearing branch caches, selected text-bearing audit/forensic JSON files, and Stage09 raw-URL row mapping/diagnostic files.

For every excluded artifact, `../../provenance/external_restricted/RESTRICTED_ARTIFACT_MANIFEST.json` records:

- expected repository path and filename;
- original byte size;
- SHA-256;
- scientific role;
- access/reproduction route;
- manifests/checksum files that record its identity;
- dependent notebooks/results;
- exact hash-verification instructions.

Authorised researchers should reacquire the original external datasets from the recorded source/revision, verify Stage01 identities, reproduce or obtain the required frozen derived artifact, place it at the expected path, and verify SHA-256 equality before running dependent stages.

Aggregate published results, figure/table source data, statistical summaries, and sensitivity comparison evidence remain in the repository. Public reproducibility metadata does not imply a right to redistribute third-party text.
