# Privacy / PII

Stage 10 is a release-engineering pass. It does not create new participant data or infer personal attributes.

The anonymized release process:
- redacts local Kaggle owner path segments and email-address patterns from text artifacts;
- flags probable secrets/API keys;
- does not rewrite scientific source repository identifiers, dataset revisions, hashes, DOI identifiers, or model repository identifiers;
- avoids redistributing raw source datasets unless redistribution is author-verified.

Any unresolved identity-bearing URL or free-text field is reported for author review in `ANONYMIZATION_REPORT.json`.
