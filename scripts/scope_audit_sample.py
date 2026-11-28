"""E27 design: probability sample for the source-guideline human audit (docs/scope/ANNOTATION_PROTOCOL.md).

Blocks (target cells of regime B):
  facebook, twitter, youtube  : n = 100 each, stratified by rubric level L (proportional, >= 2 per stratum)
  news_nonanchor              : n = 100 from BanglaClick news items whose title is not an anchor
  gold                        : n = 50 anchors (Y_A known), stratified by level, hidden among the audit items;
                                used to check that the new annotators reproduce the BanglaBait label.
Inclusion probabilities pi = n_h / N_h are recorded for the difference estimator (Eq. 8 of the paper).

Outputs
  private_data/scope/annotation/AUDIT_KEY_PRIVATE.csv      audit_id -> row_id, block, stratum, pi, gold label
  private_data/scope/annotation/sheet_annotator_{1,2,3}.csv blinded sheets (text only, per-annotator random order)
  results/scope_v1/E27/audit_design.csv                    text-free design (no gold labels)
Re-running with the same seed reproduces the sample exactly; existing sheets are never overwritten.
Usage: SCOPE_PRIVATE_DATA=... PYTHONPATH=src python scripts/scope_audit_sample.py
"""
from __future__ import annotations
import hashlib, json, re, sys, datetime
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scope import run, data, construct, paths  # noqa: E402

SEED = 20261128
N_BLOCK = {"banglaclick:facebook": 100, "banglaclick:twitter": 100, "banglaclick:youtube": 100, "news_nonanchor": 100}
N_GOLD = 50
N_ANNOTATORS = 3
URL = re.compile(r"https?://\S+|www\.\S+")


def allocate(Nh: pd.Series, n: int, floor: int = 2) -> pd.Series:
    """Proportional allocation with a floor per stratum (largest-remainder rounding)."""
    Nh = Nh[Nh > 0]
    base = np.minimum(Nh, floor).astype(int)
    rest = n - base.sum()
    share = (Nh - base) / (Nh - base).sum() * rest
    nh = base + np.floor(share).astype(int)
    for k in (share - np.floor(share)).sort_values(ascending=False).index[: n - nh.sum()]:
        nh[k] += 1
    return np.minimum(nh, Nh).astype(int)


def stratified(frame: pd.DataFrame, n: int, rng, block: str) -> pd.DataFrame:
    Nh = frame.groupby("level").size()
    nh = allocate(Nh, n)
    out = []
    for lv, k in nh.items():
        pool = frame[frame.level == lv]
        take = pool.iloc[rng.choice(len(pool), size=k, replace=False)].copy()
        take["stratum"] = f"L{lv}"; take["N_h"] = int(Nh[lv]); take["n_h"] = int(k); take["pi"] = k / Nh[lv]
        out.append(take)
    s = pd.concat(out); s["block"] = block
    return s


def main():
    out_priv = paths.private_root() / "annotation"
    out_priv.mkdir(parents=True, exist_ok=True)
    if (out_priv / "AUDIT_KEY_PRIVATE.csv").exists():
        sys.exit("Audit sample already exists; refusing to overwrite (delete deliberately to redraw).")
    a = run.anchors(); g = construct.linking_function(a)
    it = run.add_labels(data.load_predictions(with_text=True), g).drop_duplicates("row_id")
    it = it[it.dataset == "banglaclick"][["row_id", "env", "cluster_id", "level", "title", "normalized_title_sha256"]].copy()
    it["level"] = it.level.astype(int)
    anchor_hash = set(a.normalized_title_sha256)
    rng = np.random.default_rng(SEED)
    parts = []
    for env in ("banglaclick:facebook", "banglaclick:twitter", "banglaclick:youtube"):
        parts.append(stratified(it[it.env == env], N_BLOCK[env], rng, env.split(":")[1]))
    news = it[it.env == "banglaclick:news"]
    parts.append(stratified(news[~news.normalized_title_sha256.isin(anchor_hash)], N_BLOCK["news_nonanchor"], rng, "news_nonanchor"))
    gold_pool = news[news.normalized_title_sha256.isin(anchor_hash)].drop_duplicates("normalized_title_sha256")
    gold = stratified(gold_pool, N_GOLD, rng, "gold")
    gold = gold.merge(a[["normalized_title_sha256", "bb_label"]], on="normalized_title_sha256", how="left")
    s = pd.concat(parts + [gold], ignore_index=True)
    # Sampled rows can share a normalised title (reposts within a cell, or one title in two blocks). Every sampled row
    # stays in the design; annotators see each distinct title once, and its label is applied to all rows with that title.
    s = s.sample(frac=1, random_state=SEED).reset_index(drop=True)
    s["row_key"] = [f"R{i:04d}" for i in range(1, len(s) + 1)]
    uniq = s.drop_duplicates("normalized_title_sha256")[["normalized_title_sha256"]].reset_index(drop=True)
    uniq["audit_id"] = [f"A{i:04d}" for i in range(1, len(uniq) + 1)]
    s = s.merge(uniq, on="normalized_title_sha256", how="left")
    s["text"] = s.title.astype(str).map(lambda t: URL.sub("", t).strip())
    print("sampled rows:", len(s), "distinct titles to annotate:", len(uniq))
    key = s[["row_key", "audit_id", "row_id", "block", "env", "cluster_id", "level", "stratum", "N_h", "n_h", "pi", "bb_label", "normalized_title_sha256"]]
    key.to_csv(out_priv / "AUDIT_KEY_PRIVATE.csv", index=False)
    for k in range(1, N_ANNOTATORS + 1):
        sh = s.drop_duplicates("audit_id")[["audit_id", "text"]].sample(frac=1, random_state=SEED + k).reset_index(drop=True)
        sh.insert(0, "order", range(1, len(sh) + 1))
        for c in ("label_0_1", "category", "confidence_1_3", "notes"):
            sh[c] = ""
        sh.to_csv(out_priv / f"sheet_annotator_{k}.csv", index=False, encoding="utf-8-sig")
    E = ROOT / "results/scope_v1/E27"; E.mkdir(parents=True, exist_ok=True)
    design = key.drop(columns=["bb_label", "normalized_title_sha256", "audit_id"]).assign(is_gold=key.block == "gold")  # no audit_id: keeps annotators blind to level
    design.to_csv(E / "audit_design.csv", index=False)
    summ = design.groupby(["block", "stratum"]).agg(N_h=("N_h", "first"), n_h=("n_h", "first")).reset_index()
    summ.to_csv(E / "audit_design_summary.csv", index=False)
    man = {"experiment": "E27-design", "created": datetime.datetime.now().isoformat(timespec="seconds"), "seed": SEED,
           "n_items": int(len(s)), "blocks": {b: int((s.block == b).sum()) for b in s.block.unique()},
           "private_key_sha256": hashlib.sha256((out_priv / "AUDIT_KEY_PRIVATE.csv").read_bytes()).hexdigest(),
           "sheets_sha256": {f"sheet_annotator_{k}.csv": hashlib.sha256((out_priv / f"sheet_annotator_{k}.csv").read_bytes()).hexdigest()
                             for k in range(1, N_ANNOTATORS + 1)}}
    (E / "MANIFEST_design.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(json.dumps(man["blocks"], indent=2)); print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
