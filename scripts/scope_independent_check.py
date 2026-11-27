"""Independent re-derivation of the headline phenomenon numbers from the raw third-party files.

Deliberately does NOT import the scope package, the R6 clean-row manifest or any derived cache: it reads
the raw BanglaBait CSVs and the raw BanglaClick workbook, derives platform from the post URL, applies its
own title normalisation, and recomputes overlap, label conflict, kappa and prevalence. It then compares
with the pipeline outputs in results/scope_v1/. Differences are reported, not hidden.

Usage: SCOPE_PRIVATE_DATA=... python scripts/scope_independent_check.py
Writes results/scope_v1/INDEPENDENT_CHECK/independent_check.json (aggregates only).
"""
import json, os, re, unicodedata
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ["SCOPE_PRIVATE_DATA"]) / "raw_external"
OUT = ROOT / "results/scope_v1/INDEPENDENT_CHECK"


def kappa(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe))


def norm_strict(s):  # NFC, collapse whitespace, strip
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(s))).strip()


def norm_loose(s):  # additionally drop punctuation/symbols and case-fold
    s = unicodedata.normalize("NFKC", str(s)).casefold()
    s = "".join(ch for ch in s if not unicodedata.category(ch).startswith(("P", "S", "Z", "C")))
    return s


def platform(url):
    u = str(url).lower()
    if "facebook.com" in u or "fb.com" in u or "fb.watch" in u:
        return "facebook"
    if "youtube.com" in u or "youtu.be" in u:
        return "youtube"
    if "twitter.com" in u or "x.com/" in u:
        return "twitter"
    return "news"


def main():
    bb = pd.concat([pd.read_csv(RAW / f"BanglaBait/repository_snapshot/{s}.csv", usecols=["title", "label", "domain"]).assign(split=s)
                    for s in ("train", "dev", "test")], ignore_index=True)
    bc = pd.read_excel(RAW / "BanglaClick/repository_snapshot/Data/Final_4961_Dataset.xlsx")
    bc = pd.DataFrame({"title": bc.TEXT, "L": bc.FINAL_CLICKBAIT_LEVEL.astype(int), "platform": bc.POST_URL.fillna(bc.SOURCE_URL).map(platform),
                       "source": bc.SOURCE.astype(str)})
    res = {"banglabait_rows": len(bb), "banglaclick_rows": len(bc),
           "banglaclick_platform_counts": bc.platform.value_counts().to_dict(),
           "prevalence_r6_by_platform": bc.assign(y=(bc.L >= 1)).groupby("platform").y.mean().round(4).to_dict(),
           "prevalence_aligned_by_platform": bc.assign(y=(bc.L >= 2)).groupby("platform").y.mean().round(4).to_dict(),
           "banglabait_prevalence_by_split": bb.groupby("split").label.mean().round(4).to_dict(), "overlap": {}}
    for name, fn in (("strict_nfc_whitespace", norm_strict), ("loose_no_punct_casefold", norm_loose)):
        b = bb.assign(k=bb.title.map(fn)); c = bc[bc.platform == "news"].assign(k=lambda d: d.title.map(fn))
        bl = b.groupby("k").label.agg(["mean", "size"]); cl = c.groupby("k").L.agg(["max", "min", "size"])
        j = bl.join(cl, how="inner", lsuffix="_bb", rsuffix="_bc")
        ya = (j["mean"] >= 0.5).astype(int)
        r = {"shared_title_classes": int(len(j)), "bb_label_inconsistent_within_class": int(((j["mean"] > 0) & (j["mean"] < 1)).sum()),
             "bc_level_inconsistent_within_class": int((j["max"] != j["min"]).sum())}
        for t in (1, 2, 3):
            yb = (j["max"] >= t).astype(int)
            r[f"cut{t}"] = {"conflicts": int((ya != yb).sum()), "disagreement": round(float((ya != yb).mean()), 4),
                            "kappa": round(kappa(ya, yb), 4), "bc_prev": round(float(yb.mean()), 4)}
        r["bb_prev_on_anchors"] = round(float(ya.mean()), 4)
        # anchors restricted to titles in the BanglaBait train split only (one candidate origin of the 1,058 figure)
        tr = b[b.split == "train"].groupby("k").label.mean(); jt = j.loc[j.index.isin(tr.index)]
        yat = (jt["mean"] >= 0.5).astype(int); ybt = (jt["max"] >= 1).astype(int)
        r["train_split_only"] = {"classes": int(len(jt)), "conflicts_cut1": int((yat != ybt).sum())}
        # all platforms (not only news-derived rows)
        cal = bc.assign(k=bc.title.map(fn)).groupby("k").L.max(); ja = bl.join(cal, how="inner")
        r["any_platform_classes"] = int(len(ja))
        res["overlap"][name] = r
    # pipeline values for comparison
    e1 = pd.read_csv(ROOT / "results/scope_v1/E1/threshold_agreement.csv")
    res["pipeline_E1"] = e1.round(4).to_dict(orient="records")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "independent_check.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
