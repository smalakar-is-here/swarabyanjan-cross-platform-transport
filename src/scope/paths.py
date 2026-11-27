"""Locations and integrity checks for restricted inputs (E0)."""
from __future__ import annotations
import csv, hashlib, json, os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIG = json.loads((REPO / "configs/scope/scope_config.json").read_text(encoding="utf-8"))
RESULTS = REPO / "results" / "scope_v1"
MANIFEST = REPO / "provenance/external_restricted/RESTRICTED_ARTIFACT_MANIFEST.csv"


def private_root() -> Path:
    root = os.environ.get(CONFIG["private_data_env"])
    if not root:
        raise RuntimeError("Set SCOPE_PRIVATE_DATA to the private data directory (see docs/scope/REPRODUCE.md).")
    return Path(root)


def restored(rel: str) -> Path:
    return private_root() / "restored" / rel


def raw(rel: str) -> Path:
    return private_root() / "raw_external" / rel


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_restricted() -> list[dict]:
    """Check every manifest artefact; return one status row per artefact (no paths outside the alias)."""
    rows = []
    for r in csv.DictReader(open(MANIFEST, encoding="utf-8")):
        p = restored(r["path"])
        status = "MISSING"
        if p.exists():
            status = "VERIFIED" if sha256(p) == r["sha256"] else "HASH_MISMATCH"
        rows.append({"path": r["path"], "expected_sha256": r["sha256"], "status": status, "role": r["role"]})
    return rows


def verify_raw() -> list[dict]:
    rows = []
    sums = raw("SHA256SUMS.txt")
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(maxsplit=1)
        rel = rel.lstrip("*")
        p = raw(rel)
        status = "MISSING" if not p.exists() else ("VERIFIED" if sha256(p) == digest else "HASH_MISMATCH")
        rows.append({"path": rel, "expected_sha256": digest, "status": status})
    return rows
