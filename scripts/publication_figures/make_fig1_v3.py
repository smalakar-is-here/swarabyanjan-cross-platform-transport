"""Figure 1 v3, "Transport Engine" (approved Concept A; schematic centrepiece; no worked example).

Presentation only. Every number and status shown is read from an upstream canonical record and asserted
(see SRC below), and every displayed label is logged with its source in
output/FIG01_v3_SOURCE_MANIFEST.{csv,json}. Nothing is recomputed: values are read or counted from
stored statuses. The within-domain contrasts in the centrepiece are schematic (no prevalences, no NLL-gap values).

Run: python scripts/publication_figures/make_fig1_v3.py   (output: $SWFIG_OUT)
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

import swfig_common as C

REPO = C.ROOT
SRC = {
    "H1": "results/primary/stage05/H1_RESULTS.json",
    "H2": "results/primary/stage05/H2_RESULTS.json",
    "H3": "results/primary/stage05/H3_RESULTS.json",
    "H3P": "results/primary/stage05/H3_PERTURBATION_RESULTS.json",
    "H4": "results/primary/stage06/H4_RESULTS.json",
    "BRIDGE": "results/primary/stage07/H3_H4_BRIDGE_RESULTS.json",
    "TEMPS": "results/primary/stage04/TEMPERATURES.json",
    "SPLITS": "data/derived/primary/stage03/CLEAN_SPLIT_SUMMARY.csv",
    "SPARSITY": "results/primary/stage08/figure_source_data/SUPP_H3_SPARSITY_support_source.csv",
    "SOURCES": "provenance/canonical/stage01/SOURCES_LOCK.json",
    "STAGE09": "results/primary/stage09/SENSITIVITY_SUMMARY.md",
    "RELEASE": "provenance/release/RELEASE_STATUS.json",
    "P0SUM": "provenance/canonical/stage02/P0_AUDIT_SUMMARY.md",
    "CONSTRUCT": "provenance/canonical/stage02/CONSTRUCT_ALIGNMENT_MATRIX.csv",
    "FIREWALL": "provenance/canonical/stage02/SPLIT_FIREWALL.json",
}
MANUSCRIPT = {"EXP": "manuscript sections/experiments.tex (private; not distributed in this repository)",
              "FRAME": "manuscript sections/framework.tex (private; not distributed in this repository)"}
SHA = {k: hashlib.sha256((REPO / v).read_bytes()).hexdigest() for k, v in SRC.items()}
SHA.update({k: "not available (private manuscript source)" for k in MANUSCRIPT})
J = {k: json.loads((REPO / v).read_text(encoding="utf8")) for k, v in SRC.items() if v.endswith(".json")}
MANIFEST: list[dict] = []


def src(key, field, value, element, text):
    path = SRC.get(key) or MANUSCRIPT.get(key)
    MANIFEST.append(dict(element=element, displayed_text=text, source_file=path, source_field=field,
                         verified_value=str(value), source_sha256=SHA[key]))


# ------------------------------------------------------------------ read and assert every displayed fact
splits = {r["split"]: r for r in csv.DictReader(open(REPO / SRC["SPLITS"], encoding="utf8")) if r["original_rows"]}
n_train, n_dev, n_test = (int(float(splits[s]["clean_rows"])) for s in ("train", "dev", "test"))
q = {s: int(float(splits[s]["quarantined_rows"])) for s in ("train", "dev", "test")}
assert (n_train, n_dev, n_test) == (10544, 1100, 2784) and sum(q.values()) == 628
temps = J["TEMPS"]["temperatures"]
n_temps = sum(len(v) for v in temps.values())
assert n_temps == 20 and "DEV" in J["TEMPS"]["source"]
h1 = J["H1"]["confirmatory_assessment_rows"]
models = sorted({r["model"] for r in h1}); seeds = sorted({r["seed"] for r in h1})
assert models == ["banglabert", "ebm", "tfidf_lr", "xlmr"] and seeds == [42, 137, 2718, 31415, 65537]
h1_est = sum(r["inferential_status"] == "ESTIMABLE" for r in h1)
h1_sup = sorted({(r["domain"], r["model"]) for r in h1 if r["setting_supported"]})
assert h1_est == 80 and h1_sup == [("banglaclick:youtube", "tfidf_lr")]
h2 = J["H2"]["confirmatory_assessment_rows"]
h2_est = sum(r["inferential_status"] == "ESTIMABLE" for r in h2); h2_sup = sum(bool(r["setting_supported"]) for r in h2)
assert (len(h2), h2_est, h2_sup) == (100, 100, 0)
H3 = J["H3"]
cells = H3["banglaclick_cue_platform_contrasts"]
status = {(r["cue"], r["target"].split(":")[1]): r["inferential_status"] for r in cells}
n_est = sum(v == "ESTIMABLE" for v in status.values())
assert len(status) == 32 and n_est == 24
wald = H3["global_banglaclick_outcome_wald"]["status"]
bh = [H3["family_fdr_outcome_full"]["status"], H3["family_fdr_reliability_full"]["status"]]
assert wald == "NON_ESTIMABLE" and all(s.startswith("NON_ESTIMABLE") for s in bh)
bb_est = sum(r["inferential_status"] == "ESTIMABLE" for r in H3["baitbuster_replication"])
assert (len(H3["baitbuster_replication"]), bb_est) == (8, 5)
assert H3["bootstrap_replicates"] == 5000
cues_h3 = sorted({r["cue"] for r in cells})
assert cues_h3 == sorted(C.CUES)
h3p = J["H3P"]
h3p_est = sum(r["inferential_status"] == "ESTIMABLE" for r in h3p["rows"])
h3p_targets = sorted({r["target"] for r in h3p["rows"]})
assert (len(h3p["rows"]), h3p_est, len(h3p["families"]), len(h3p["models"]), len(h3p_targets)) == (150, 150, 3, 2, 5)
assert h3p["causal_real_world_claim"] is False
H4 = J["H4"]
assert H4["primary_budget"] == 0.1 and sorted(H4["budgets"]) == [0.05, 0.1, 0.2]
h4_domains = sorted({r["domain"] for r in H4["bootstrap_rows"]})
assert len(h4_domains) == 5
br = J["BRIDGE"]
br_status = [v["bootstrap"]["platform_fe_slope"]["estimability_status"] for v in br["pairs"].values()]
assert len(br_status) == 4 and all(s.startswith("NON_ESTIMABLE") for s in br_status) and br["cells"] == 32
assert br["causal"] is False
plat_n = {}
for r in csv.DictReader(open(REPO / SRC["SPARSITY"], encoding="utf8")):
    plat_n[r["domain"]] = int(r["rows"])
assert [plat_n[f"banglaclick:{p}"] for p in C.PLATFORMS] == [1500, 1961, 1000, 500]
assert plat_n["baitbuster:youtube_human"] == 10000
sources_lock = set(J["SOURCES"]["sources"].keys())
assert {"BanglaBait", "BanglaClick", "BanMANI", "BanFakeNews-2.0", "BaitBuster-Bangla"} <= sources_lock
s09 = (REPO / SRC["STAGE09"]).read_text(encoding="utf8")
assert "Source-site host clusters: 40" in s09
rel = J["RELEASE"]
assert rel["scientific_computation_executed"] is False and rel["qa"]["stage09_sensitivity_only"] is True
# public repository evidence for facts the manuscript text also states
p0_summary = (REPO / SRC["P0SUM"]).read_text(encoding="utf8")
assert "`P0_NO_GO`" in p0_summary
roles = {r["dataset"]: r["permitted_role"] for r in csv.DictReader(open(REPO / SRC["CONSTRUCT"], encoding="utf8"))}
assert roles["BanMANI"] == roles["BanFakeNews-2.0"] == "construct-boundary external diagnostic"
assert roles["BaitBuster-Bangla"] == "cross-platform construct diagnostic only"
assert J["FIREWALL"]["protected_scientific_fields"]["dataset"] == "BanglaBait"
src("P0SUM", "Decision", "P0_NO_GO", "integrity chip 1 (public evidence)", "Stage02 audit P0_NO_GO")
src("CONSTRUCT", "permitted_role[BanMANI, BanFakeNews-2.0]", "construct-boundary external diagnostic",
    "construct-boundary note (public evidence)", "Construct boundary only: BanMANI, BanFakeNews-2.0")
src("CONSTRUCT", "permitted_role[BaitBuster-Bangla]", roles["BaitBuster-Bangla"], "BaitBuster tile (public evidence)",
    "BaitBuster: related construct; corroboration only")
src("FIREWALL", "protected_scientific_fields.dataset", "BanglaBait", "firewall (public evidence)",
    "target-blind firewall")

# ------------------------------------------------------------------ drawing helpers
mpl.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                     "mathtext.bf": "Arial:bold"})
# Sub/superscripts are rendered at SHRINK_FACTOR x base size (default 0.7). With 0.86 and math text at 9.4 pt,
# scripts such as the c in C_c print at 8.1 pt, meeting the 8 pt floor while keeping the paper's notation.
import matplotlib._mathtext as _mathtext  # noqa: E402
_mathtext.SHRINK_FACTOR = 0.86
BM = 9.4
W_IN, H_IN = C.FIG_W, 7.34
XMAX, YMAX = 100.0, 100.0 * H_IN / W_IN          # 1 unit = 0.064 in in both directions
fig = plt.figure(figsize=(W_IN, H_IN))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, XMAX); ax.set_ylim(0, YMAX); ax.axis("off")

BLUE, BLUE_BG = "#1F4E79", "#EEF3F8"
ACCENT, ACCENT_BG = "#9C4A00", "#FFF6EA"
GREY, FIRE = "#6E6E6E", "#B22222"
B8, B85, B9, B10 = 8.0, 8.5, 9.0, 10.0


def box(x, y, w, h, fc="white", ec="black", lw=0.8, ls="-", r=0.8, z=1, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                ls=ls, zorder=z, alpha=alpha))


def txt(x, y, s, size=B8, w="normal", ha="left", va="center", color="black", style="normal", z=5, **kw):
    return ax.text(x, y, s, fontsize=size, fontweight=w, ha=ha, va=va, color=color, style=style, zorder=z, **kw)


def arrow(x1, y1, x2, y2, color="black", lw=1.0, ls="-", z=4, head=7):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=head, lw=lw, color=color,
                                 ls=ls, zorder=z, shrinkA=0, shrinkB=0))


# ================================================================== band 1: data, integrity, development
TOP = YMAX - 1.0
B1_BOT = TOP - 36.5
# --- source panel
box(1, B1_BOT, 47.2, TOP - B1_BOT, fc=BLUE_BG, ec=BLUE, lw=1.0)
txt(2.5, TOP - 2.4, "SOURCE: BanglaBait", B10, "bold", color=BLUE)
src("SOURCES", "sources/BanglaBait", "present", "source title", "SOURCE: BanglaBait")
# integrity column
ix, iw = 2.5, 15.5
chips = [(TOP - 13.0, 7.2, ["Stage02 audit", "P0_NO_GO", "(title overlap)"], "#FFF3D6", "#B07A00"),
         (TOP - 23.6, 7.2, ["Stage03", "decontamination", f"{sum(q.values())} quarantined"], "white", BLUE),
         (TOP - 34.6, 7.6, ["Clean splits:", "normalised-title", "overlap = 0"], "white", BLUE)]
for k, (y0, hh, lines, fc, ec) in enumerate(chips):
    box(ix, y0, iw, hh, fc=fc, ec=ec, lw=0.8, r=0.6)
    for li, ln in enumerate(lines):
        txt(ix + iw / 2, y0 + hh - 1.6 - li * 2.25, ln, B8, "bold" if li == 0 else "normal", ha="center")
    if k < len(chips) - 1:
        arrow(ix + iw / 2, y0, ix + iw / 2, chips[k + 1][0] + chips[k + 1][1] + 0.1, color=BLUE, lw=0.9)
src("EXP", "Data integrity paragraph: P0_NO_GO", "P0_NO_GO", "integrity chip 1", "Stage02 audit P0_NO_GO (title overlap)")
src("SPLITS", "quarantined_rows (train+dev+test)", f"{q['train']}+{q['dev']}+{q['test']}={sum(q.values())}",
    "integrity chip 2", f"Stage03 decontamination; {sum(q.values())} quarantined")
src("EXP", "Data integrity: audited normalised cross-split title overlap is zero (lock M05; not semantic duplicates)",
    "0", "integrity chip 3", "Clean splits: normalised-title overlap = 0")
# splits bar (proportional to the frozen clean counts)
sx, sw = 20.0, 26.5
tot = n_train + n_dev + n_test
wt, wd, ws = (sw * n / tot for n in (n_train, n_dev, n_test))
by = TOP - 10.2
txt(sx, TOP - 6.2, "Clean source splits", B8, "bold", color=BLUE)
for x0, w_, fc in [(sx, wt, "#2E6DA4"), (sx + wt, wd, "#7FA7CC"), (sx + wt + wd, ws, "#B9CDE3")]:
    ax.add_patch(Rectangle((x0, by), w_, 2.6, fc=fc, ec="white", lw=0.6, zorder=3))
txt(sx + wt / 2, by + 1.3, f"TRAIN {n_train:,}", B8, "bold", ha="center", color="white")
txt(sx + sw, by - 1.7, f"DEV {n_dev:,} · TEST {n_test:,}", B8, ha="right")
src("SPLITS", "clean_rows", f"{n_train}/{n_dev}/{n_test}", "splits bar (widths proportional to clean_rows)",
    f"TRAIN {n_train:,} DEV {n_dev:,} TEST {n_test:,}")
dev_lines = [("4 model families × 5 seeds", True), ("EBM · TF-IDF + LR", False), ("BanglaBERT · XLM-R", False),
             (f"DEV-only calibration", True), (f"{n_temps} scalar temperatures", False),
             ("8 frozen cue families", True), ("presence = count > 0", False)]
yy = by - 5.0
for ln, bold in dev_lines:
    if bold and ln != dev_lines[0][0]:
        yy -= 0.7
    txt(sx, yy, ln, B8, "bold" if bold else "normal")
    yy -= 2.25
src("H1", "confirmatory_assessment_rows: model and seed sets", f"{models}; {seeds}", "development text",
    "4 model families × 5 seeds; EBM · TF-IDF + LR · BanglaBERT · XLM-R")
src("TEMPS", "temperatures (count); source", f"{n_temps}; {J['TEMPS']['source']}", "development text",
    f"DEV-only calibration; {n_temps} scalar temperatures")
src("H3", "banglaclick_cue_platform_contrasts[].cue (distinct)", len(cues_h3), "development text",
    "8 frozen cue families; presence = count > 0 (Eq. 1)")

# --- firewall
FX = 50.0
ax.plot([FX, FX], [B1_BOT + 0.6, TOP - 0.6], color=FIRE, lw=1.5, ls=(0, (5, 3)), zorder=6)
txt(FX, (B1_BOT + TOP) / 2, "target-blind firewall", B8, "bold", ha="center", color=FIRE, rotation=90,
    bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="none"), z=7)
src("EXP", "Data integrity and external-evaluation firewall", "targets held out", "firewall", "target-blind firewall")

# --- target panel
TX = 51.8
box(TX, B1_BOT, 99 - TX, TOP - B1_BOT, fc="white", ec="0.35", lw=1.0)
txt(TX + 1.5, TOP - 2.4, "TARGETS: BanglaClick, 4 platforms", B10, "bold")
txt(TX + 1.5, TOP - 5.3, "each evaluated separately, never pooled; no target", B8, color="0.25")
txt(TX + 1.5, TOP - 7.6, "data in fitting, calibration, thresholds or budgets", B8, color="0.25")
tw, th = 22.0, 6.4
for i, p in enumerate(C.PLATFORMS):
    x0 = TX + 1.5 + (i % 2) * (tw + 1.5)
    y0 = TOP - 16.6 - (i // 2) * (th + 1.4)
    box(x0, y0, tw, th, fc="white", ec=C.PLATFORM_COLOR[p], lw=1.4, r=0.6)
    ax.plot(x0 + 2.6, y0 + th / 2, marker=C.PLATFORM_MARKER[p], color=C.PLATFORM_COLOR[p], ms=7, mec="black",
            mew=0.4, zorder=6)
    txt(x0 + 5.0, y0 + th / 2 + 1.25, C.PLATFORM_LABEL[p], B85, "bold")
    txt(x0 + 5.0, y0 + th / 2 - 1.3, f"n = {plat_n['banglaclick:' + p]:,}", B8)
    src("SPARSITY", f"rows where domain=banglaclick:{p}", plat_n["banglaclick:" + p], f"platform tile {p}",
        f"{C.PLATFORM_LABEL[p]} n = {plat_n['banglaclick:' + p]:,}")
# BaitBuster tile (separate; related construct)
bby = TOP - 30.8
box(TX + 1.5, bby, 99 - TX - 3.0, 5.8, fc="#F4F4F4", ec=GREY, lw=1.0, ls=(0, (3, 2)), r=0.6)
ax.plot(TX + 4.1, bby + 2.9, marker="P", color="0.35", ms=7, mec="black", mew=0.4, zorder=6)
txt(TX + 6.5, bby + 4.15, "BaitBuster: related construct", B85, "bold")
txt(TX + 6.5, bby + 1.65, f"{plat_n['baitbuster:youtube_human']:,} human-labelled rows; corroboration only", B8)
src("SPARSITY", "rows where domain=baitbuster:youtube_human", plat_n["baitbuster:youtube_human"], "BaitBuster tile",
    "BaitBuster: related construct; 10,000 human-labelled rows; corroboration only")
txt(TX + 1.5, B1_BOT + 3.9, "Construct boundary only (never trained or tuned on):", B8, color="0.3", style="italic")
txt(TX + 1.5, B1_BOT + 1.6, "BanMANI, BanFakeNews-2.0", B8, color="0.3", style="italic")
src("SOURCES", "sources/BanMANI, sources/BanFakeNews-2.0", "present", "construct-boundary note",
    "Construct boundary only: BanMANI, BanFakeNews-2.0")
src("EXP", "Data paragraph 1: construct-boundary resources, not clickbait training or tuning data", "stated",
    "construct-boundary note", "never trained or tuned on")

# ================================================================== band 2: contrast-of-contrasts (focal)
B2_TOP = B1_BOT - 2.3
B2_BOT = B2_TOP - 32.6
Y2 = lambda rel: B2_TOP + rel          # offsets measured down from the band top
box(1, B2_BOT, 98, B2_TOP - B2_BOT, fc=ACCENT_BG, ec=ACCENT, lw=1.6, r=1.0)
txt(50, Y2(-2.3), "Source-referenced contrast of contrasts", B10, "bold", ha="center", color=ACCENT)
txt(50, Y2(-5.2), r"within each domain $d$, for each cue family $c$:   "
    r"$RD(c,d) = P(Y{=}1\,|\,C_c{=}1,d) - P(Y{=}1\,|\,C_c{=}0,d)$", BM, ha="center")
txt(50, Y2(-8.2), r"$G_s(c,d) = E[L_s\,|\,C_c{=}1,d] - E[L_s\,|\,C_c{=}0,d]$   ($L_s$: calibrated NLL of EBM seed $s$)",
    BM, ha="center")
src("FRAME", "Eqs. cue-indicator, cue-rd and cue-loss-gap", "definitions", "centrepiece definitions",
    "RD(c,d) = P(Y=1|C=1,d) - P(Y=1|C=0,d); G_s(c,d) = E[L_s|C=1,d] - E[L_s|C=0,d]")
# entry arrows: source (left) and targets (right) enter the blocks outside the centred text
arrow(10.3, B1_BOT, 10.3, Y2(-10.0), color=BLUE, lw=1.2)
arrow(95.0, B1_BOT, 95.0, Y2(-10.0), color="0.25", lw=1.2)
BLK_TOP, BLK_H = Y2(-10.0), 11.8


def contrast_block(x0, label, sub, col, result):
    """Schematic within-domain contrast: no values, no bar heights."""
    w = 39.5
    y0 = BLK_TOP - BLK_H
    box(x0, y0, w, BLK_H, fc="white", ec=col, lw=1.1, r=0.6, z=2)
    txt(x0 + 1.4, BLK_TOP - 1.9, label, B9, "bold", color=col)
    txt(x0 + w - 1.4, BLK_TOP - 1.9, sub, B8, ha="right", color="0.3")
    for j, (lab1, lab2) in enumerate([("cue present", r"$C_c{=}1$"), ("cue absent", r"$C_c{=}0$")]):
        bx = x0 + 1.4 + j * 13.6
        box(bx, y0 + 1.0, 11.0, 6.8, fc="#FAFAFA", ec="0.45", lw=0.7, r=0.5, z=3)
        txt(bx + 5.5, y0 + 5.8, lab1, B8, ha="center")
        txt(bx + 5.5, y0 + 2.9, lab2, BM, ha="center")
    txt(x0 + 13.4, y0 + 4.4, "−", B10, "bold", ha="center")
    txt(x0 + 27.0, y0 + 4.4, "→", B10, ha="center")
    txt(x0 + 34.2, y0 + 5.8, result[0], BM, "bold", ha="center")
    txt(x0 + 34.2, y0 + 2.9, result[1], BM, "bold", ha="center")


contrast_block(3.0, "Source S", "clean BanglaBait TEST", BLUE, (r"$RD(c,S)$", r"$G_s(c,S)$"))
contrast_block(57.5, "Target T", "one platform at a time", "0.2", (r"$RD(c,T)$", r"$G_s(c,T)$"))
src("EXP", "external evaluation: each target analysed separately; source TEST locked", "stated",
    "contrast block labels", "Source S: clean BanglaBait TEST; Target T: one platform at a time")
# operator: target minus source, placed between the two blocks
OPY = BLK_TOP - BLK_H / 2 - 0.6
ax.add_patch(Circle((50, OPY), 3.4, fc="white", ec=ACCENT, lw=1.4, zorder=4))
txt(50, OPY, "T − S", B85, "bold", ha="center", color=ACCENT)
arrow(42.5, OPY, 46.6, OPY, color=ACCENT, lw=1.1)
arrow(57.5, OPY, 53.4, OPY, color=ACCENT, lw=1.1)
CARD_TOP, CARD_H = B2_BOT + 8.7, 8.1
arrow(48.0, OPY - 2.9, 38.0, CARD_TOP, color=ACCENT, lw=1.1)
arrow(52.0, OPY - 2.9, 62.0, CARD_TOP, color=ACCENT, lw=1.1)
cards = [(3.0, r"$\Delta RD(c,T,S) = RD(c,T) - RD(c,S)$", "cue–label transport; model-independent"),
         (52.0, r"$\Delta$NLL-gap $= \frac{1}{5}\sum_{s}\,[\,G_s(c,T) - G_s(c,S)\,]$",
          "proper-loss transport; EBM only, 5-seed average")]
for x0, f, scope in cards:
    box(x0, CARD_TOP - CARD_H, 45.0, CARD_H, fc="white", ec=ACCENT, lw=1.2, r=0.5, z=2)
    txt(x0 + 22.5, CARD_TOP - 2.5, f, BM, "bold", ha="center")
    txt(x0 + 22.5, CARD_TOP - 6.3, scope, B8, ha="center", color="0.2")
src("FRAME", "Eq. delta-rd", "definition", "estimand card 1", "ΔRD(c,T,S) = RD(c,T) - RD(c,S)")
src("FRAME", "Eq. delta-gap (five-seed EBM average of per-seed gap differences)", "definition", "estimand card 2",
    "ΔNLL-gap = (1/5) Σ_s [G_s(c,T) - G_s(c,S)]")
src("FRAME", "text after Eq. delta-gap; Limitations ¶2 (ΔRD model-independent; ΔNLL-gap EBM five-seed)", "stated",
    "estimand scope lines", "model-independent / EBM only, 5-seed average")

# ================================================================== band 3: evidence layers (parallel)
txt(50, B2_BOT - 2.0, f"Inference gate: a contrast is adjudicated only if all {H3['bootstrap_replicates']:,} "
    "locked bootstrap draws are valid;", B8, "bold", ha="center", color=ACCENT)
txt(50, B2_BOT - 4.3, "otherwise it is NON_ESTIMABLE (no redraw, no dimension dropped)", B8, "bold", ha="center",
    color=ACCENT)
src("H3", "bootstrap_replicates; estimability_policy", H3["bootstrap_replicates"], "inference gate",
    "adjudicated only if all 5,000 locked bootstrap draws are valid; otherwise NON_ESTIMABLE")
BAR = B2_BOT - 6.3
ax.plot([24.0, 87.0], [BAR, BAR], color="black", lw=0.9, zorder=3)
ax.plot([50, 50], [B2_BOT - 5.2, BAR], color="black", lw=0.9)
txt(50, BAR, "  same frozen system, parallel evidence layers (not a sequence)  ", B8, ha="center", color="0.25",
    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"), z=6)
for xd in (24.0, 61.9, 87.0):
    arrow(xd, BAR, xd, BAR - 1.5, lw=0.9)

# --- H3 primary lane (dominant)
L3_TOP, L3_BOT = BAR - 1.5, 6.6
box(1, L3_BOT, 47.0, L3_TOP - L3_BOT, fc="#E9E9E9", ec="black", lw=1.8, r=0.8)
txt(2.5, L3_TOP - 2.3, "H3 PRIMARY: cue transport", B10, "bold")
src("H3", "role (primary analysis)", "primary", "H3 lane title", "H3 PRIMARY: cue transport")
short = {"curiosity_information_withholding": "Curiosity", "direct_address_imperative": "Direct address",
         "exaggeration_superlative": "Exaggeration", "interrogative_structure": "Interrogative",
         "punctuation_intensity": "Punctuation", "sensational_emotional_loading": "Sensational",
         "specificity_numerals": "Specificity", "vague_deictic_reference": "Vague/deictic"}
gx, cw, ch = 14.7, 3.9, 2.15
gy_top = L3_TOP - 7.0
for j, p in enumerate(C.PLATFORMS):
    ax.plot(gx + j * cw + cw / 2, gy_top + 1.5, marker=C.PLATFORM_MARKER[p], color=C.PLATFORM_COLOR[p], ms=6.5,
            mec="black", mew=0.4, zorder=6)
for i, cue in enumerate(C.CUES):
    y0 = gy_top - (i + 1) * ch
    txt(gx - 0.8, y0 + ch / 2, short[cue], B8, ha="right")
    for j, p in enumerate(C.PLATFORMS):
        est = status[(cue, p)] == "ESTIMABLE"
        ax.add_patch(Rectangle((gx + j * cw, y0), cw, ch, fc="#7A9CBF" if est else "white",
                               ec="white" if est else "0.3", lw=0.8, hatch=None if est else "////", zorder=4))
        if not est:
            txt(gx + j * cw + cw / 2, y0 + ch / 2, "NE", B8, "bold", ha="center", z=6,
                bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"))
        src("H3", f"banglaclick_cue_platform_contrasts[cue={cue},target=banglaclick:{p}].inferential_status",
            status[(cue, p)], f"H3 grid cell {cue} x {p}", "estimable" if est else "NE")
GRID_BOT = gy_top - 8 * ch
txt(2.5, GRID_BOT - 2.3, "hatched = NE; columns = platforms", B8, color="0.25")
gx1 = gx + 4 * cw
stats = [("24/32 estimable", "bold"), ("8/32 NE", "bold"), ("Wald, 32-D:", "normal"),
         ("NON_ESTIMABLE", "bold"), ("Simes/BH, 8 cue", "normal"), ("families, ΔRD and", "normal"),
         ("ΔNLL-gap: both NE", "bold"), ("BaitBuster: 5/8", "normal"), ("(corroboration)", "normal")]
assert n_est == 24 and 32 - n_est == 8 and bb_est == 5
yy = gy_top + 0.4
for s_, wgt in stats:
    txt(gx1 + 1.3, yy, s_, B8, wgt)
    yy -= 2.0
src("H3", "count of inferential_status==ESTIMABLE over 32 cells", n_est, "H3 stats", "24/32 estimable; 8/32 NE")
src("H3", "global_banglaclick_outcome_wald.status", wald, "H3 stats", "global 32-D Wald: NON_ESTIMABLE")
src("H3", "family_fdr_outcome_full.status; family_fdr_reliability_full.status", "; ".join(bh), "H3 stats",
    "8-family Simes/BH (ΔRD and ΔNLL-gap): NON_ESTIMABLE")
src("H3", "baitbuster_replication: ESTIMABLE count", f"{bb_est}/8", "H3 stats", "BaitBuster cues: 5/8 estimable")

# --- other layers: 2 x 2 cards; coverage markers = which external datasets contribute
COV = {
    "H1/H2": sorted({r["domain"] for r in h1} | {r["domain"] for r in h2}),
    "H3-P": h3p_targets,
    "H4": h4_domains,
    "BRIDGE": [f"banglaclick:{p}" for p in C.PLATFORMS],
}
assert len(COV["H1/H2"]) == 5 and len(COV["H3-P"]) == 5 and len(COV["H4"]) == 5
ALL_DOMS = [f"banglaclick:{p}" for p in C.PLATFORMS] + ["baitbuster:youtube_human"]
MK = {**{f"banglaclick:{p}": (C.PLATFORM_MARKER[p], C.PLATFORM_COLOR[p]) for p in C.PLATFORMS},
      "baitbuster:youtube_human": ("P", "0.35")}
cx0, cgap = 49.5, 0.8
cw2 = (99.0 - cx0 - cgap) / 2
chh = (L3_TOP - L3_BOT - 1.0) / 2
card_spec = [
    ("H1/H2", "SUPPORTING", (cx0, L3_TOP - chh), (1.0, (0, (4, 2))),
     [f"H1: {h1_est}/100 estimable;", "localised support", f"H2: {h2_est}/100 estimable;", f"{h2_sup}/100 support"]),
    ("H3-P", "CORROBORATIVE", (cx0 + cw2 + cgap, L3_TOP - chh), (1.0, (0, (5, 1.5, 1, 1.5))),
     ["controlled cue edits;", "3 cues × 2 models;", f"{h3p_est}/150 estimable;", "non-causal"]),
    ("H4", "OPERATIONAL", (cx0, L3_BOT), (1.2, "-"),
     ["fixed review budget;", "3 review metrics;", "5%, 20% sensitivity;", "10% primary"]),
    ("H3→H4", "EXPLORATORY", (cx0 + cw2 + cgap, L3_BOT), (1.2, (0, (1, 1.5))),
     ["32 cells, 4 slopes;", "descriptive only;", "joint inference NE;", "non-causal"]),
]
for name, role, (x0, y0), (lw_, ls_), lines in card_spec:
    box(x0, y0, cw2, chh, fc="white", ec="black", lw=lw_, ls=ls_, r=0.6)
    txt(x0 + 1.2, y0 + chh - 2.0, name, B9, "bold")
    txt(x0 + cw2 - 1.2, y0 + chh - 2.0, role, B8, "bold", ha="right", color="0.35")
    for li, ln in enumerate(lines):
        txt(x0 + 1.2, y0 + chh - 4.7 - li * 2.15, ln, B8)
    key = name.replace("H3→H4", "BRIDGE")
    for di, dname in enumerate(ALL_DOMS):
        mk, col = MK[dname]
        inc = dname in COV[key]
        ax.plot(x0 + cw2 - 10.3 + di * 2.2, y0 + 1.9, marker=mk, ms=4.8, mew=0.5,
                mfc=col if inc else "white", mec="black" if inc else "0.6", ls="None", zorder=6)
src("H1", "inferential_status count; setting_supported", f"{h1_est}/100; supported={h1_sup}", "H1/H2 card",
    "H1: 80/100 estimable; localised support")
src("H2", "inferential_status count; setting_supported count", f"{h2_est}/100; {h2_sup}/100", "H1/H2 card",
    "H2: 100/100 estimable; 0/100 support")
src("H3P", "rows (count), inferential_status, families, models, causal_real_world_claim",
    f"{len(h3p['rows'])}; {h3p_est}; {h3p['families']}; {h3p['models']}; {h3p['causal_real_world_claim']}",
    "H3-P card", "controlled cue edits; 3 cues × 2 models; 150/150 estimable; non-causal")
src("H4", "primary_budget; budgets; metrics (lift, error capture, AURC)", f"{H4['primary_budget']}; {H4['budgets']}",
    "H4 card", "fixed review budget; 3 review metrics; 5%, 20% sensitivity; 10% primary")
src("BRIDGE", "cells; pairs[*].bootstrap.platform_fe_slope.estimability_status; causal",
    f"{br['cells']}; {br_status}; {br['causal']}", "H3→H4 card",
    "32 cells, 4 slopes; descriptive only; joint inference NE; non-causal")
for k, v in COV.items():
    src({"H1/H2": "H1", "H3-P": "H3P", "H4": "H4", "BRIDGE": "BRIDGE"}[k], "distinct domains in stored rows",
        "|".join(v), f"coverage markers {k}", "filled marker = dataset contributes; hollow = not in this analysis")

# ================================================================== band 4: subordinate stages
box(1, 0.6, 98, 4.6, fc="#F5F5F5", ec="0.75", lw=0.6, r=0.5)
txt(2.5, 2.9, "Stage09: source-site cluster sensitivity; uncertainty only", B8, color="0.3")
txt(97.5, 2.9, "Stage10: release packaging; no scientific computation", B8, ha="right", color="0.3")
src("STAGE09", "Source-site host clusters: 40; summary", "sensitivity only", "Stage09 strip",
    "Stage09: source-site cluster sensitivity (uncertainty only)")
src("RELEASE", "scientific_computation_executed; qa.stage09_sensitivity_only",
    f"{rel['scientific_computation_executed']}; {rel['qa']['stage09_sensitivity_only']}", "Stage10 strip",
    "Stage10: release packaging, no scientific computation")

if __name__ == "__main__":
    paths = C.export(fig, "FIG01_STUDY_DESIGN_v3")
    fields = list(MANIFEST[0].keys())
    with open(C.OUT / "FIG01_v3_SOURCE_MANIFEST.csv", "w", newline="", encoding="utf8") as f:
        wr = csv.DictWriter(f, fieldnames=fields, lineterminator="\n"); wr.writeheader(); wr.writerows(MANIFEST)
    (C.OUT / "FIG01_v3_SOURCE_MANIFEST.json").write_text(json.dumps(
        {"figure": "FIG01_STUDY_DESIGN_v3", "design": "Concept A, Transport Engine (approved 2026-11-15)",
         "presentation_only": True, "recomputation": "none (stored values and status counts only)",
         "schematic_elements": "within-domain cue-present/cue-absent contrasts carry no values",
         "sources": {k: {"path": SRC.get(k) or MANUSCRIPT.get(k), "sha256": v} for k, v in SHA.items()},
         "elements": MANIFEST, "text_qa": C.TEXT_QA.get("FIG01_STUDY_DESIGN_v3", [])}, indent=1),
        encoding="utf8")
    print("wrote", [p.name for p in paths.values()], "| manifest rows:", len(MANIFEST),
          "| text QA issues:", len(C.TEXT_QA.get("FIG01_STUDY_DESIGN_v3", [])))
