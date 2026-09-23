# P0 Integrity, Provenance, and Leakage Audit Summary

**Decision:** `P0_NO_GO`

**Methodology:** `P0-v1.1.1`

**Methodology SHA-256:** `fcf658785c8a91ffa31a08b7861dbed07aac26f06b52d133ce257f3610d3ce6b`

**Frozen package:** `EXTERNAL-DATASET-BYTES-v1`

**Expected ZIP SHA-256:** `e0de5c224f406fe22d8ad30804217465f789c49c70e1d3afbe00afd28deb7ccf`

## Finding counts

- BLOCKER: 3
- MAJOR: 10
- MINOR: 7
- INFORMATIONAL: 24

## Stage 03 gate

Stage 03 model development remains blocked until every `stage03_blocking=true` finding is resolved through an explicit versioned protocol decision and Stage 02 is re-executed.

## Blocking findings

- `P0-LEAK-PROTECTED-BANGLABAIT_DEV-BANGLABAIT_TEST` — Exact or normalized protected-title overlap crosses the BanglaBait train/calibration/test firewall.
- `P0-LEAK-PROTECTED-BANGLABAIT_TRAIN-BANGLABAIT_DEV` — Exact or normalized protected-title overlap crosses the BanglaBait train/calibration/test firewall.
- `P0-LEAK-PROTECTED-BANGLABAIT_TRAIN-BANGLABAIT_TEST` — Exact or normalized protected-title overlap crosses the BanglaBait train/calibration/test firewall.

## Construct boundary

BanglaBait is treated as target-proximal clickbait supervision/proxy rather than direct proof of equivalence with the broader yellow-journalism construct. BanglaClick is external clickbait diagnostic evidence. BanMANI, BanFakeNews-2.0, and BaitBuster-Bangla remain adjacent/boundary constructs and are prohibited from model tuning.

## BaitBuster serialization rule

The pinned Version 3 Parquet file is the canonical research artifact. XLSX is secondary reference material. The unavailable publisher-listed CSV is not reconstructed and no cross-serialization byte-equivalence claim is made.
