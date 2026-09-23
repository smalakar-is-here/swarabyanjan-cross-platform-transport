# Row-Level / Decontamination Input Audit

This folder preserves the non-parquet audit and identity records from `NEWS_DECONTAMINATION_INPUTS.zip`.

Large clean-split and prediction Parquets are not duplicated here when they already exist at their frozen primary data paths. Their identity remains traceable through `INPUT_SHA256SUMS.txt` and the canonical manifests.
