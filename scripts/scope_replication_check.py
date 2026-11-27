"""Compare independent Kaggle GPU sessions of the same transformer work units, file by file.

Writes results/scope_v1/REPLICATION/replication_check.json (aggregate numbers only, no text).
Usage: SCOPE_PRIVATE_DATA=<private_data/scope> python scripts/scope_replication_check.py
"""
import json, os
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
K = Path(os.environ["SCOPE_PRIVATE_DATA"]) / "kaggle_out"
UNITS = {"w05": "banglabert/scope-e10-w05-banglabert-sourceonly-r6", "w07": "xlmr/scope-e10-w07-xlmr-sourceonly-r6"}


def compare(a_dir, b_dir):
    rows = []
    for fa in sorted(Path(a_dir).rglob("*.parquet")):
        fb = next(Path(b_dir).rglob(fa.name), None)
        if fb is None or fa.name.startswith("srcval_"):
            continue
        a = pd.read_parquet(fa).set_index("row_id"); b = pd.read_parquet(fb).set_index("row_id")
        if not a.index.sort_values().equals(b.index.sort_values()):
            rows.append({"file": fa.name, "same_rows": False}); continue
        b = b.loc[a.index]
        rows.append({"file": fa.name, "same_rows": True, "n": len(a),
                     "max_abs_dp": float(np.abs(a.p_calibrated - b.p_calibrated).max()),
                     "max_abs_dlogit": float(np.abs(a.logit - b.logit).max())})
    return rows


out = {}
for u, rel in UNITS.items():
    orig = K / "full" / rel
    sessions = {"Account01 replicate (2026-11-27)": next((K / "replicates" / "Account01").glob(f"*{u}*"), None),
                "news-fold rerun w11/w12 (2026-11-27)": next((K / "replicates" / "srcval_rerun").glob(f"{'w11' if u == 'w05' else 'w12'}_*"), None)}
    for name, d in sessions.items():
        if d is None:
            continue
        r = compare(d, orig)
        out[f"{u} vs {name}"] = {"files_compared": len(r), "all_rows_match": all(x["same_rows"] for x in r),
                                 "max_abs_dp": max(x.get("max_abs_dp", np.inf) for x in r),
                                 "max_abs_dlogit": max(x.get("max_abs_dlogit", np.inf) for x in r)}
dst = ROOT / "results/scope_v1/REPLICATION"; dst.mkdir(parents=True, exist_ok=True)
(dst / "replication_check.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
