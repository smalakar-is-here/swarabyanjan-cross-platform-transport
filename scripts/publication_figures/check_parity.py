"""Independent numeric parity test for the redesigned main figures.

Re-reads the frozen source CSVs (raw csv module, not the plotting helpers) and compares every plotted
element recorded by make_main_figures.py: element counts, point values, interval endpoints, NE sets,
and degradation counts. Also checks manuscript-stated invariants (SCIENTIFIC_LOCKS).
Exit code 0 only if every check passes. Writes output/PARITY_REPORT.json and .md.
"""
import csv
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "results/primary/stage08/figure_source_data"
import os
OUT = pathlib.Path(os.environ.get("SWFIG_OUT", str(ROOT / "build" / "publication_figures")))
P = json.loads((OUT / "main_figures_plotted_elements.json").read_text(encoding="utf8"))
plotted = P["plotted"]
results = []


def check(name, ok, detail=""):
    results.append({"check": name, "pass": bool(ok), "detail": detail})


def rows(fn):
    with open(SRC / fn, newline="", encoding="utf8") as f:
        return list(csv.DictReader(f))


def ci(s):
    if not s or not s.strip().startswith("["):
        return (None, None)
    a, b = json.loads(s)
    return float(a), float(b)


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a == b  # exact float equality: values must be passed through untouched


# hashes of everything used
idx = json.loads((SRC / "FIGURE_SOURCE_DATA_INDEX.json").read_text(encoding="utf8"))
idxh = {e["source_csv"]: e["source_csv_sha256"] for e in idx["source_files"]}
for name, h in P["sources_sha256"].items():
    if name.startswith("tables/"):
        continue
    check(f"hash {name}", hashlib.sha256((SRC / name).read_bytes()).hexdigest() == h == idxh[name])

# ---------------- FIG02
f2 = plotted["FIG02"]
A = [e for e in f2 if e["panel"] == "A"]
ra = rows("FIG02_H1_H2_A_source.csv")
check("FIG02a count = 120 (6 domains x 4 models x 5 seeds)", len(A) == len(ra) == 120, f"{len(A)}")
ka = {(r["domain"], r["model"], int(r["seed"])): float(r["calibrated_auroc"]) for r in ra}
check("FIG02a values exact", all(ka[(e["domain"], e["model"], e["seed"])] == e["y"] for e in A))
rbc = {(r["domain"], r["model"], int(r["seed"])): r for r in rows("FIG02_H1_H2_B_C_source.csv")}
for letter, y, c, est in [("b", "nll_relative_change", "nll_relative_change_ci95", "nll_relative_change_estimable"),
                          ("c", "balanced_brier_relative_change", "balanced_brier_relative_change_ci95",
                           "balanced_brier_relative_change_estimable")]:
    E = [e for e in f2 if e["panel"] == letter]
    okv = all(float(rbc[(e["domain"], e["model"], e["seed"])][y]) == e["y"] for e in E)
    oke = all((rbc[(e["domain"], e["model"], e["seed"])][est] == "True") == e["estimable"] for e in E)
    okc = all(e["lo"] is None or ci(rbc[(e["domain"], e["model"], e["seed"])][c]) == (e["lo"], e["hi"]) for e in E)
    nne = sum(not e["estimable"] for e in E)
    check(f"FIG02{letter} count=100, values/CIs/estimability exact", len(E) == 100 and okv and oke and okc,
          f"n={len(E)} NE={nne}")
check("FIG02c BaitBuster balanced-Brier NE = 20/20",
      sum(1 for e in f2 if e["panel"] == "c" and not e["estimable"] and e["domain"].startswith("bait")) == 20)
check("FIG02b all 100 NLL endpoints estimable", all(e["estimable"] for e in f2 if e["panel"] == "b"))
rd = {(r["domain"], r["model"], int(r["seed"])): r for r in rows("FIG02_H1_H2_D_source.csv")}
D = [e for e in f2 if e["panel"] == "d"]
check("FIG02d count=100, values/CIs exact",
      len(D) == 100 and all(float(rd[(e["domain"], e["model"], e["seed"])]["target_minus_source_gain"]) == e["y"]
                            and ci(rd[(e["domain"], e["model"], e["seed"])]["h2_target_minus_source_gain_ci95"]) == (e["lo"], e["hi"])
                            for e in D))
check("H2 support 0/100 in source", sum(r["setting_supported"] == "True" for r in rd.values()) == 0)
check("H1 supported seeds = 5 (YouTube TF-IDF LR only)",
      sorted({(k[0], k[1]) for k, r in rbc.items() if r["setting_supported"] == "True"}) == [("banglaclick:youtube", "tfidf_lr")]
      and sum(r["setting_supported"] == "True" for r in rbc.values()) == 5)

# ---------------- FIG03
f3 = plotted["FIG03"]
r3 = {(r["cue"], r["platform"]): r for r in rows("FIG03_H3_CUE_TRANSPORT_A_B_source.csv")}
for letter, v, c in [("a", "delta_rd", "delta_rd_ci95"), ("b", "ebm_delta_nll_gap_mean_5seeds", "ebm_delta_nll_gap_ci95")]:
    E = [e for e in f3 if e["panel"] == letter]
    ok = True
    for e in E:
        r = r3[(e["cue"], e["platform"])]
        if r["inferential_status"] == "ESTIMABLE":
            ok &= e["value"] == float(r[v]) and (e["lo"], e["hi"]) == ci(r[c])
        else:
            ok &= e["value"] is None and e["status"] == r["inferential_status"]
    ne = sorted((e["cue"], e["platform"]) for e in E if e["value"] is None)
    check(f"FIG03{letter} 32 cells; estimable values+CIs exact; NE cells drawn without value", len(E) == 32 and ok,
          f"NE={len(ne)}")
EXPECTED_NE = sorted([("curiosity_information_withholding", "youtube"), ("curiosity_information_withholding", "twitter"),
                      ("direct_address_imperative", "youtube"), ("exaggeration_superlative", "youtube"),
                      ("interrogative_structure", "youtube"), ("sensational_emotional_loading", "youtube"),
                      ("sensational_emotional_loading", "twitter"), ("vague_deictic_reference", "youtube")])
check("FIG03 NE set equals the 8 cells listed in results_discussion.tex",
      sorted((e["cue"], e["platform"]) for e in f3 if e["panel"] == "a" and e["value"] is None) == EXPECTED_NE)

# ---------------- FIG04
f4 = plotted["FIG04"]
pr = rows("FIG04_H4_FIXED_BUDGET_primary_10pct_source.csv")
se = rows("FIG04_H4_FIXED_BUDGET_budget_sensitivity_source.csv")
kp = {(r["domain"], r["model"], int(r["seed"])): r for r in pr}
fields = {"lift": ("target_minus_source_lift", "lift_ci95", "lift_p", -1),
          "error_capture": ("target_minus_source_error_capture", "error_capture_ci95", "error_capture_p", -1),
          "aurc": ("target_minus_source_aurc", "aurc_ci95", "aurc_p", +1)}
for m, (f, c, p, sign) in fields.items():
    E = [e for e in f4 if e["panel"] == f"primary_{m}"]
    ok = all(float(kp[(e["domain"], e["model"], e["seed"])][f]) == e["y"]
             and ci(kp[(e["domain"], e["model"], e["seed"])][c]) == (e["lo"], e["hi"]) for e in E)
    okd = all(e["degrading"] == (float(kp[(e["domain"], e["model"], e["seed"])][p]) < 0.05 and
                                 sign * float(kp[(e["domain"], e["model"], e["seed"])][f]) > 0) for e in E)
    check(f"FIG04 primary {m}: 80 points, values/CIs/status exact", len(E) == 80 and ok and okd)
    S = [e for e in f4 if e["panel"] == f"budget_{m}"]
    ks = {(r["domain"], float(r["budget"]), r["model"], int(r["seed"])): float(r[f]) for r in se}
    check(f"FIG04 sensitivity {m}: 240 points exact",
          len(S) == 240 and all(ks[(e["domain"], e["budget"], e["model"], e["seed"])] == e["y"] for e in S))
counts = {(e["panel"].replace("count_", ""), e["domain"].split(":")[1]): (e["n_degrading"], e["n"])
          for e in f4 if e["panel"].startswith("count_")}
EXPECTED_H4 = {(m, p): (20, 20) for m in fields for p in ("news", "facebook", "twitter")}
EXPECTED_H4.update({("lift", "youtube"): (20, 20), ("error_capture", "youtube"): (15, 20), ("aurc", "youtube"): (15, 20)})
check("FIG04 degradation counts equal SCIENTIFIC_LOCKS (News/Facebook/Twitter 20/20; YouTube 20,15,15)",
      counts == EXPECTED_H4, json.dumps({f"{k[0]}|{k[1]}": v for k, v in counts.items()}))

# ---------------- FIG05
f5 = plotted["FIG05"]
specs = {"a": ("FIG05_H3_H4_BRIDGE_abs_delta_rd__vs__lift_deterioration_source.csv", "abs_delta_rd",
               "lift_deterioration_source_minus_target", 0.235172),
         "b": ("FIG05_H3_H4_BRIDGE_abs_delta_rd__vs__error_capture_deterioration_source.csv", "abs_delta_rd",
               "error_capture_deterioration_source_minus_target", -0.015018),
         "c": ("FIG05_H3_H4_BRIDGE_abs_delta_nll_gap__vs__lift_deterioration_source.csv", "abs_delta_nll_gap",
               "lift_deterioration_source_minus_target", 0.136024),
         "d": ("FIG05_H3_H4_BRIDGE_abs_delta_nll_gap__vs__error_capture_deterioration_source.csv", "abs_delta_nll_gap",
               "error_capture_deterioration_source_minus_target", -0.005515)}
for k, (fn, xc, yc, text_slope) in specs.items():
    rr = {(r["cue"], r["platform"]): r for r in rows(fn)}
    E = [e for e in f5 if e["panel"] == k and "x" in e]
    S = [e for e in f5 if e["panel"] == k and "slope" in e][0]
    ok = all(float(rr[(e["cue"], e["platform"])][xc]) == e["x"] and float(rr[(e["cue"], e["platform"])][yc]) == e["y"]
             for e in E)
    check(f"FIG05{k}: 32 points exact; slope {S['slope']:.6f} matches manuscript {text_slope}; status NE",
          len(E) == 32 and ok and round(S["slope"], 6) == text_slope and S["status"].startswith("NON_ESTIMABLE"))

ok_all = all(r["pass"] for r in results)
(OUT / "PARITY_REPORT.json").write_text(json.dumps({"all_pass": ok_all, "results": results}, indent=1), encoding="utf8")
md = ["# Numeric parity report — redesigned main figures", "",
      f"**Overall: {'PASS' if ok_all else 'FAIL'}** ({sum(r['pass'] for r in results)}/{len(results)} checks)", "",
      "| Check | Result | Detail |", "|---|---|---|"]
md += [f"| {r['check']} | {'PASS' if r['pass'] else '**FAIL**'} | {r['detail']} |" for r in results]
(OUT / "PARITY_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf8")
print("\n".join(md))
sys.exit(0 if ok_all else 1)
