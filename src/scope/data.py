"""Unified loaders for the frozen R6 prediction caches and the raw label sources.

Environment notation
--------------------
An environment e is a (dataset, platform) cell. Each dataset carries one annotation
regime r(e):
    A  BanglaBait   (news headlines, binary human label)            -- source
    B  BanglaClick  (news/facebook/youtube/twitter, ordinal 0-3 rubric level)
    C  BaitBuster   (YouTube titles, binary human label, 10k human subset)
The R6 binary target label for regime B is Y = 1[L >= 1] (verified on all 4,961 rows).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .paths import restored, raw

CUES = [
    "curiosity_information_withholding", "direct_address_imperative", "exaggeration_superlative",
    "interrogative_structure", "punctuation_intensity", "sensational_emotional_loading",
    "specificity_numerals", "vague_deictic_reference",
]
CONTROLS = ["token_count", "character_count", "mean_token_length", "latin_code_mixing_proportion"]
PRED = "data/derived/primary/stage05/"
KEEP = ["row_id", "platform", "label", "cluster_id", "normalized_title_sha256", "model", "seed",
        "logit", "p_uncalibrated", "temperature", "p_calibrated", *CUES, *CONTROLS]


def banglaclick_levels() -> pd.DataFrame:
    x = pd.read_excel(raw("BanglaClick/repository_snapshot/Data/Final_4961_Dataset.xlsx"))
    out = pd.DataFrame({
        "row_id": [f"bc_{i:04d}" for i in range(len(x))],
        "level": x["FINAL_CLICKBAIT_LEVEL"].astype(int).to_numpy(),
        "fast_level": x["FAST_LEVEL"].to_numpy(),
        "trigger_level": x["TRIGGER_LEVEL"].to_numpy(),
        "strong_triggers": x["STRONG_TRIGGER_COUNT"].to_numpy(),
        "weak_triggers": x["WEAK_TRIGGER_COUNT"].to_numpy(),
        "fast_fail": x["FAST_FAIL_COUNT"].astype(str).str.extract(r"([a-z,]+)")[0].fillna("").to_numpy(),
        "media_type": x["MEDIA_TYPE"].astype(str).to_numpy(),
    })
    return out


def load_predictions(with_text: bool = False) -> pd.DataFrame:
    """Long table: one row per (item, model, seed) over source test, BanglaClick and BaitBuster."""
    cols = KEEP + (["title"] if with_text else [])
    src = pd.read_parquet(restored(PRED + "LOCKED_SOURCE_TEST_PREDICTIONS.parquet"), columns=cols)
    src["dataset"], src["regime"], src["platform"] = "banglabait", "A", "news"
    src["row_id"] = "bb_" + src["row_id"].astype(str)
    bcp = pd.read_parquet(restored(PRED + "BANGLACLICK_PLATFORM_PREDICTIONS.parquet"), columns=cols)
    bcp["dataset"], bcp["regime"] = "banglaclick", "B"
    bcp = bcp.merge(banglaclick_levels(), on="row_id", how="left", validate="many_to_one")
    if not ((bcp["level"] >= 1).astype(int) == bcp["label"]).all():
        raise AssertionError("R6 BanglaClick label is not 1[level>=1]")
    bbt = pd.read_parquet(restored(PRED + "BAITBUSTER_HUMAN_PREDICTIONS.parquet"), columns=cols)
    bbt["dataset"], bbt["regime"], bbt["platform"] = "baitbuster", "C", "youtube"
    df = pd.concat([src, bcp, bbt], ignore_index=True)
    df["env"] = df["dataset"] + ":" + df["platform"]
    df["seed"] = df["seed"].astype(int)
    return df


ENV_ORDER = ["banglabait:news", "banglaclick:news", "banglaclick:facebook", "banglaclick:youtube",
             "banglaclick:twitter", "baitbuster:youtube"]


def banglabait_all_rows() -> pd.DataFrame:
    """All 15,056 raw BanglaBait rows with label, split, domain, date and normalised-title hash."""
    man = pd.read_parquet(restored("CLEAN_ROW_MANIFEST.parquet"))
    parts = []
    for s in ["train", "dev", "test"]:
        d = pd.read_csv(raw(f"BanglaBait/repository_snapshot/{s}.csv"), usecols=["domain", "date", "label"])
        d["source_artifact"], d["source_row_id"] = "banglabait_" + s, np.arange(len(d))
        parts.append(d)
    bb = pd.concat(parts, ignore_index=True).merge(
        man[["source_artifact", "source_row_id", "normalized_title_sha256", "decision", "original_split"]],
        on=["source_artifact", "source_row_id"], how="left", validate="one_to_one")
    if bb["normalized_title_sha256"].isna().any():
        raise AssertionError("BanglaBait rows missing from clean-row manifest")
    return bb


def anchors() -> pd.DataFrame:
    """Anchor items: normalised-title classes labelled by both regime A and regime B.

    One row per title class: BanglaBait label (unanimous within class), BanglaClick level
    (max within class; min reported for audit) and the BanglaClick platform.
    """
    bb = banglabait_all_rows()
    g = bb.groupby("normalized_title_sha256")
    a = pd.DataFrame({"bb_label_mean": g["label"].mean(), "bb_rows": g.size(),
                      "bb_splits": g["original_split"].agg(lambda s: ",".join(sorted(set(s))))})
    p = pd.read_parquet(restored(PRED + "BANGLACLICK_PLATFORM_PREDICTIONS.parquet"),
                        columns=["row_id", "platform", "normalized_title_sha256", "cluster_id"]).drop_duplicates("row_id")
    p = p.merge(banglaclick_levels()[["row_id", "level", "fast_fail", "fast_level", "trigger_level"]], on="row_id")
    h = p.groupby("normalized_title_sha256")
    b = pd.DataFrame({"bc_level": h["level"].max(), "bc_level_min": h["level"].min(), "bc_rows": h.size(),
                      "bc_platform": h["platform"].first(), "bc_cluster": h["cluster_id"].first(),
                      "fast_fail": h["fast_fail"].first(), "fast_level": h["fast_level"].first(),
                      "trigger_level": h["trigger_level"].first()})
    j = a.join(b, how="inner").reset_index()
    j["bb_label"] = (j["bb_label_mean"] >= 0.5).astype(int)
    return j
