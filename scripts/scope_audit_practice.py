"""Practice set for E27 annotator training (ANNOTATION_PROTOCOL.md, step 2): 20 BanglaBait training headlines
(10 clickbait, 10 not), none of which is an anchor or in the audit sample. Writes a blank sheet and a separate answer
key to private_data/scope/annotation/. Seeded; refuses to overwrite."""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scope import run, paths  # noqa: E402

SEED = 20261129


def main():
    ann = paths.private_root() / "annotation"
    if (ann / "practice_sheet.csv").exists():
        sys.exit("practice set exists; refusing to overwrite")
    pool = run.training_pool()
    tr = pool[pool.env == "banglabait:news"]
    excl = set(run.anchors().normalized_title_sha256) | set(pd.read_csv(ann / "AUDIT_KEY_PRIVATE.csv").normalized_title_sha256)
    tr = tr[~tr.normalized_title_sha256.isin(excl)].drop_duplicates("normalized_title_sha256")
    pick = pd.concat([tr[tr.y_r6 == v].sample(10, random_state=SEED) for v in (1, 0)]).sample(frac=1, random_state=SEED)
    pick = pick.reset_index(drop=True); pick["practice_id"] = [f"P{i:02d}" for i in range(1, 21)]
    sheet = pick[["practice_id", "title"]].rename(columns={"title": "text"})
    for c in ("label_0_1", "category", "confidence_1_3", "notes"):
        sheet[c] = ""
    sheet.to_csv(ann / "practice_sheet.csv", index=False, encoding="utf-8-sig")
    pick[["practice_id", "row_id", "y_r6"]].rename(columns={"y_r6": "banglabait_label"}).to_csv(ann / "practice_key_PRIVATE.csv", index=False)
    print("practice set written:", len(pick), "items;", int(pick.y_r6.sum()), "clickbait")


if __name__ == "__main__":
    main()
