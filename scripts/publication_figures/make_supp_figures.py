"""Redesigned supplementary Figures S1, S2, S4, S5, S6 (presentation only; frozen Stage08 source data).

Each figure records its plotted elements and is checked here against the raw CSV text (csv module):
counts, exact values, interval endpoints and NE sets. Exit code 1 on any mismatch.
"""
from __future__ import annotations

import csv
import json
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

import swfig_common as C
from make_main_figures import H4_METRICS, MODEL_DODGE, degrading, model_handles, seed_offsets

SEEDS = [42, 137, 2718, 31415, 65537]
REC: dict[str, list] = {}
CHECKS: list = []


def raw(name):
    with open(C.SRC / name, newline="", encoding="utf8") as f:
        return list(csv.DictReader(f))


def chk(name, ok, detail=""):
    CHECKS.append({"check": name, "pass": bool(ok), "detail": detail})


# ------------------------------------------------------------------ S1: H1 relative NLL by model x seed
def s1():
    df = C.load("SUPP_H1_H2_MODEL_SEED_all_source.csv")
    fig, axs = plt.subplots(2, 2, figsize=(C.FIG_W, 4.6), sharey=True, constrained_layout=True)
    so = seed_offsets(width=0.36)
    seed_col = ["#1B1B1B", "#555555", "#8A8A8A", "#B0B0B0", "#D0D0D0"]
    for ax, m in zip(axs.ravel(), C.MODELS):
        sub = df[df.model == m]
        bb = C.TARGETS.index("baitbuster:youtube_human")
        ax.axvspan(bb - 0.5, bb + 0.5, color="#F2F2F2", lw=0, zorder=0)
        for r in sub.itertuples():
            x = C.TARGETS.index(r.domain) + so[SEEDS.index(r.seed)]
            lo, hi = C.ci(r.nll_relative_change_ci95)
            est = C.is_true(r.nll_relative_change_estimable)
            ax.plot([x, x], [lo, hi], color=C.MODEL_COLOR[m], lw=0.9)
            ax.plot(x, r.nll_relative_change, marker=C.MODEL_MARKER[m], color=C.MODEL_COLOR[m], mec="black",
                    mew=0.3, ms=3.6, ls="None")
            if C.is_true(r.setting_supported):
                ax.plot(x, r.nll_relative_change, marker="o", mfc="none", mec="black", ms=7.5, mew=0.8, ls="None")
            REC.setdefault("S1", []).append(dict(domain=r.domain, model=m, seed=int(r.seed),
                                                 y=float(r.nll_relative_change), lo=lo, hi=hi, est=est,
                                                 overall=r.inferential_status))
        ax.axhline(0, color="black", lw=0.6)
        ax.axhline(0.10, color="black", lw=0.7, ls=(0, (4, 2)))
        ax.set_xticks(range(5), [C.DOMAIN_LABEL[d] for d in C.TARGETS])
        ax.set_xlim(-0.5, 4.5)
        ax.set_title(C.MODEL_LABEL[m], loc="left", color=C.MODEL_COLOR[m], fontweight="bold")
        for i in range(1, 5):
            ax.axvline(i - 0.5, color="0.9", lw=0.5, zorder=0)
    for ax in axs[:, 0]:
        ax.set_ylabel("H1 relative NLL change\n(target − source) / source")
    fig.legend(handles=[Line2D([0], [0], color="black", lw=0.7, ls=(0, (4, 2)), label="H1 threshold +0.10"),
                        Line2D([0], [0], marker="o", mfc="none", mec="black", ls="None", ms=6.5,
                               label="H1-supporting seed"),
                        Line2D([0], [0], color="0.4", lw=0.9, label="95% bootstrap interval")],
               loc="outside upper center", ncol=3, frameon=False,
               title="Within each target, the five seeds (42, 137, 2718, 31415, 65537) are drawn left to right")
    rows = raw("SUPP_H1_H2_MODEL_SEED_all_source.csv")
    k = {(r["domain"], r["model"], int(r["seed"])): r for r in rows}
    ok = all(float(k[(e["domain"], e["model"], e["seed"])]["nll_relative_change"]) == e["y"] and
             C.ci(k[(e["domain"], e["model"], e["seed"])]["nll_relative_change_ci95"]) == (e["lo"], e["hi"])
             for e in REC["S1"])
    chk("S1: 100 points; NLL values and CIs exact", len(REC["S1"]) == 100 and ok)
    chk("S1: BaitBuster overall H1 NE 20/20 retained in source (status not recoded)",
        sum(e["overall"].startswith("NON") for e in REC["S1"] if e["domain"].startswith("bait")) == 20)
    return fig


# ------------------------------------------------------------------ S2: cue support heatmap
def s2():
    sp = C.load("SUPP_H3_SPARSITY_support_source.csv")
    h3 = C.load("FIG03_H3_CUE_TRANSPORT_A_B_source.csv")
    bb = C.load("SUPP_BAITBUSTER_H3_A_B_source.csv")
    doms = ["source_test"] + C.TARGETS
    ne = {(r.cue, r.target) for r in h3.itertuples() if r.inferential_status != "ESTIMABLE"}
    ne |= {(r.cue, "baitbuster:youtube_human") for r in bb.itertuples() if r.inferential_status != "ESTIMABLE"}
    M = np.zeros((8, 6))
    fig, ax = plt.subplots(figsize=(C.FIG_W, 4.4), constrained_layout=True)
    for r in sp.itertuples():
        M[C.CUES.index(r.cue), doms.index(r.domain)] = r.cue_present_rows
    im = ax.imshow(M, cmap="Blues", norm=LogNorm(vmin=max(1, M[M > 0].min()), vmax=M.max()), aspect="auto")
    for r in sp.itertuples():
        i, j = C.CUES.index(r.cue), doms.index(r.domain)
        v = int(r.cue_present_rows)
        dark = M[i, j] > np.sqrt(M[M > 0].min() * M.max()) * 2
        is_ne = (r.cue, r.domain) in ne
        if is_ne:  # hatch + outline: survives grayscale
            ax.add_patch(plt.Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fill=False, ec="#B22222", lw=1.4,
                                       hatch="////", alpha=0.55, zorder=2))
        ax.text(j, i - 0.15, f"{v:,}" + (" (NE)" if is_ne else ""), ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="white" if dark and not is_ne else "black", zorder=3,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85) if is_ne else None)
        ax.text(j, i + 0.24, f"of {int(r.rows):,}", ha="center", va="center", fontsize=8,
                color="white" if dark and not is_ne else "0.2", zorder=3,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85) if is_ne else None)
        REC.setdefault("S2", []).append(dict(cue=r.cue, domain=r.domain, present=v, rows=int(r.rows), ne=is_ne))
    ax.set_xticks(range(6), ["Source TEST", "News", "Facebook", "YouTube", "Twitter", "BaitBuster"])
    ax.set_yticks(range(8), [C.CUE_LABEL[c] for c in C.CUES])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 6), minor=True); ax.set_yticks(np.arange(-0.5, 8), minor=True)
    ax.grid(which="minor", color="white", lw=1.2); ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
    cb.set_label("cue-present rows (log scale)")
    # plain tick labels (no 10^n superscripts, which would print below 8 pt)
    cb.set_ticks([1, 10, 100, 1000])
    cb.set_ticklabels(["1", "10", "100", "1,000"])
    cb.ax.minorticks_off()
    ax.set_title("Cue-present rows (bold) out of all rows; hatched outline = H3 cell NE", loc="left", fontsize=8.5)
    rows = raw("SUPP_H3_SPARSITY_support_source.csv")
    k = {(r["cue"], r["domain"]): r for r in rows}
    ok = all(int(k[(e["cue"], e["domain"])]["cue_present_rows"]) == e["present"] and
             int(k[(e["cue"], e["domain"])]["rows"]) == e["rows"] for e in REC["S2"])
    n_ne = sum(e["ne"] for e in REC["S2"])
    chk("S2: 48 cells; counts exact; NE outlines = 8 BanglaClick + 3 BaitBuster", len(REC["S2"]) == 48 and ok and n_ne == 11,
        f"NE outlined={n_ne}")
    return fig


# ------------------------------------------------------------------ S4: BaitBuster H3 (external corroboration)
def s4():
    bb = C.load("SUPP_BAITBUSTER_H3_A_B_source.csv")
    fig, axs = plt.subplots(1, 2, figsize=(C.FIG_W, 3.6), sharey=True, constrained_layout=True)
    specs = [("delta_rd", "delta_rd_ci95", "ΔRD (target − source)", "Cue–label transport, ΔRD"),
             ("ebm_delta_nll_gap_mean_5seeds", "ebm_delta_nll_gap_ci95", "ΔNLL-gap (target − source)",
              "Proper-loss transport, EBM ΔNLL-gap")]
    for k_, (ax, (v, c, xl, title)) in enumerate(zip(axs, specs)):
        ax.axvline(0, color="black", lw=0.7)
        for r in bb.itertuples():
            y = C.CUES.index(r.cue)
            if r.inferential_status == "ESTIMABLE":
                lo, hi = C.ci(getattr(r, c))
                ax.plot([lo, hi], [y, y], color="0.25", lw=1.1)
                ax.plot(getattr(r, v), y, marker="o", color="0.25", mec="black", mew=0.4, ms=4.5, ls="None")
                REC.setdefault("S4", []).append(dict(panel=k_, cue=r.cue, v=float(getattr(r, v)), lo=lo, hi=hi))
            else:
                ax.text(0.985, y, "NE", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8,
                        fontweight="bold", color="0.25",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=C.NE_GREY, lw=0.6))
                REC.setdefault("S4", []).append(dict(panel=k_, cue=r.cue, v=None, lo=None, hi=None))
        ax.set_xlabel(xl); ax.set_title(title, loc="left")
        ax.grid(axis="x", color="0.88", lw=0.5)
        ax.set_ylim(7.6, -0.6); ax.tick_params(axis="y", length=0)
        C.panel_label(ax, "ab"[k_])
    axs[0].set_yticks(range(8), [C.CUE_LABEL[c] for c in C.CUES])
    fig.supxlabel("BaitBuster human-labelled subset: external corroboration from a related construct,\n"
                  "not a construct-equivalent replication. 5/8 cues estimable, 3/8 NE.", fontsize=8)
    rows = {r["cue"]: r for r in raw("SUPP_BAITBUSTER_H3_A_B_source.csv")}
    ok = True
    for e in REC["S4"]:
        r = rows[e["cue"]]; vcol, ccol = (specs[e["panel"]][0], specs[e["panel"]][1])
        if e["v"] is None:
            ok &= r["inferential_status"] != "ESTIMABLE"
        else:
            ok &= float(r[vcol]) == e["v"] and C.ci(r[ccol]) == (e["lo"], e["hi"])
    est = sum(e["v"] is not None for e in REC["S4"] if e["panel"] == 0)
    chk("S4: 8 cues x 2 panels; values/CIs exact; 5 estimable, 3 NE", len(REC["S4"]) == 16 and ok and est == 5)
    return fig


# ------------------------------------------------------------------ S5 / S6: H4 at 5% and 20%
def s_h4(fname, budget_label, key):
    d = C.load(fname)
    assert len(d) == 80
    fig, axs = plt.subplots(1, 3, figsize=(C.FIG_W, 2.9), constrained_layout=True)
    doms = [f"banglaclick:{p}" for p in C.PLATFORMS]
    so = seed_offsets(width=0.1)
    for j, (ax, (field, cicol, st, pc, title, ylab, sign)) in enumerate(zip(axs, H4_METRICS)):
        for r in d.to_dict("records"):
            x = doms.index(r["domain"]) + MODEL_DODGE[r["model"]] + so[SEEDS.index(r["seed"])]
            y = float(r[field]); lo, hi = C.ci(r[cicol]); deg = degrading(r, field, st, pc, sign)
            est = r[st] == "ESTIMABLE"
            col = C.MODEL_COLOR[r["model"]]
            if est:
                ax.plot([x, x], [lo, hi], color=col, lw=0.6)
                ax.plot(x, y, marker=C.MODEL_MARKER[r["model"]], ms=2.8, ls="None", mew=0.5,
                        mfc=col if deg else "white", mec="black" if deg else col)
            else:
                ax.plot(x, y, marker="x", color=C.NE_GREY, ms=3.4, mew=0.8, ls="None")
            REC.setdefault(key, []).append(dict(metric=title, domain=r["domain"], model=r["model"], seed=int(r["seed"]),
                                                y=y, lo=None if math.isnan(lo) else lo, hi=None if math.isnan(hi) else hi))
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(range(4), ["News", "Faceb.", "YouT.", "Twitter"])
        for i in range(1, 4):
            ax.axvline(i - 0.5, color="0.9", lw=0.5, zorder=0)
        ax.set_xlim(-0.5, 3.5)
        ax.set_title(f"{title} ({'lower' if sign < 0 else 'higher'} = worse)", loc="left", fontsize=8)
        ax.set_ylabel(ylab)
        C.panel_label(ax, "abc"[j])
    extra = [Line2D([0], [0], marker="o", mfc="0.3", mec="black", ls="None", ms=4.5, label="p < 0.05 and degrading"),
             Line2D([0], [0], marker="o", mfc="white", mec="0.3", ls="None", ms=4.5, label="other estimable")]
    fig.legend(handles=model_handles(extra), loc="outside upper center", ncol=6, frameon=False, handletextpad=0.3,
               columnspacing=0.8, fontsize=8,
               title=f"Prespecified {budget_label} sensitivity budget (the 10% budget is primary; Figure 4)")
    rows = raw(fname)
    kk = {(r["domain"], r["model"], int(r["seed"])): r for r in rows}
    fmap = {"Lift": ("target_minus_source_lift", "lift_ci95"), "Error capture": ("target_minus_source_error_capture",
            "error_capture_ci95"), "AURC": ("target_minus_source_aurc", "aurc_ci95")}
    ok = all(float(kk[(e["domain"], e["model"], e["seed"])][fmap[e["metric"]][0]]) == e["y"] and
             (e["lo"] is None or C.ci(kk[(e["domain"], e["model"], e["seed"])][fmap[e["metric"]][1]]) == (e["lo"], e["hi"]))
             for e in REC[key])
    chk(f"{key}: 3 metrics x 80 settings; values/CIs exact", len(REC[key]) == 240 and ok)
    return fig


# ------------------------------------------------------------------ S3a-c: H3-P controlled cue neutralisation
H3P_TARGETS = ["banglaclick:news", "banglaclick:facebook", "banglaclick:youtube", "banglaclick:twitter",
               "baitbuster:youtube_human"]
DOM_COLOR = {**{f"banglaclick:{p}": C.PLATFORM_COLOR[p] for p in C.PLATFORMS}, "baitbuster:youtube_human": "0.35"}
DOM_MARK = {**{f"banglaclick:{p}": C.PLATFORM_MARKER[p] for p in C.PLATFORMS}, "baitbuster:youtube_human": "P"}


def s3(cue, key):
    d = C.load("SUPP_H3P_SHARED_all_source.csv")
    d = d[d.cue == cue]
    assert len(d) == 50
    fig, axs = plt.subplots(1, 2, figsize=(C.FIG_W, 3.7), sharey=True, constrained_layout=True)
    for k_, (ax, m) in enumerate(zip(axs, ["ebm", "banglabert"])):
        bb = H3P_TARGETS.index("baitbuster:youtube_human")
        ax.axhspan(bb - 0.45, bb + 0.45, color="#F2F2F2", lw=0, zorder=0)
        ax.axvline(0, color="black", lw=0.7)
        for r in d[d.model == m].itertuples():
            y = H3P_TARGETS.index(r.target) + (SEEDS.index(r.seed) - 2) * 0.15
            col = DOM_COLOR[r.target]
            if r.inferential_status == "ESTIMABLE":
                lo, hi = C.ci(r.ci95)
                ax.plot([lo, hi], [y, y], color=col, lw=1.0)
                ax.plot(r.target_minus_source_delta_z, y, marker=DOM_MARK[r.target], color=col, mec="black", mew=0.3,
                        ms=4, ls="None")
            else:
                ax.text(0.985, y, "NE", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8)
                lo = hi = None
            REC.setdefault(key, []).append(dict(model=m, target=r.target, seed=int(r.seed),
                                                v=float(r.target_minus_source_delta_z), lo=lo, hi=hi,
                                                status=r.inferential_status))
        for i in range(1, len(H3P_TARGETS)):
            ax.axhline(i - 0.5, color="0.9", lw=0.5, zorder=0)
        ax.set_title(C.MODEL_LABEL[m], loc="left", fontweight="bold")
        ax.set_xlabel("target − source mean Δz\n(Δz = normalised-original − neutralised logit)")
        ax.grid(axis="x", color="0.9", lw=0.5)
        C.panel_label(ax, "ab"[k_])
    axs[0].set_yticks(range(5), ["News", "Facebook", "YouTube", "Twitter", "BaitBuster"])
    axs[0].set_ylim(4.6, -0.6)
    fig.suptitle(f"H3-P, {C.CUE_LABEL[cue]}: controlled model-sensitivity corroboration (non-causal)\n"
                 "Five seeds per target, top to bottom 42, 137, 2718, 31415, 65537; bars = 95% bootstrap intervals",
                 fontsize=8.5)
    rows = [x for x in raw("SUPP_H3P_SHARED_all_source.csv") if x["cue"] == cue]
    kk = {(x["model"], x["target"], int(x["seed"])): x for x in rows}
    ok = all(float(kk[(e["model"], e["target"], e["seed"])]["target_minus_source_delta_z"]) == e["v"] and
             (e["lo"] is None or C.ci(kk[(e["model"], e["target"], e["seed"])]["ci95"]) == (e["lo"], e["hi"]))
             for e in REC[key])
    chk(f"{key}: 50 contrasts (2 models x 5 targets x 5 seeds); values/CIs/status exact",
        len(REC[key]) == 50 and ok and all(e["status"] == "ESTIMABLE" for e in REC[key]))
    return fig


# ------------------------------------------------------------------ S7: H4 primary 10% model x seed (all 5 domains)
def s7():
    d = C.load("SUPP_H4_MODEL_SEED_all_source.csv")
    assert len(d) == 100
    doms = H3P_TARGETS
    fig, axs = plt.subplots(1, 3, figsize=(C.FIG_W, 7.6), sharey=True, constrained_layout=True)
    ypos = {}
    y = 0
    for di, dname in enumerate(doms):
        for s in SEEDS:
            ypos[(dname, s)] = y; y += 1
        y += 0.8
    for j, (ax, (field, cicol, st, pc, title, ylab, sign)) in enumerate(zip(axs, H4_METRICS)):
        for di, dname in enumerate(doms):
            ys = [ypos[(dname, s)] for s in SEEDS]
            if dname.startswith("bait"):
                ax.axhspan(min(ys) - 0.6, max(ys) + 0.6, color="#F2F2F2", lw=0, zorder=0)
        ax.axvline(0, color="black", lw=0.7)
        for r in d.to_dict("records"):
            yy = ypos[(r["domain"], int(r["seed"]))] + (C.MODELS.index(r["model"]) - 1.5) * 0.17
            v = float(r[field]); col = C.MODEL_COLOR[r["model"]]
            if r[st] == "ESTIMABLE":
                lo, hi = C.ci(r[cicol]); deg = degrading(r, field, st, pc, sign)
                ax.plot([lo, hi], [yy, yy], color=col, lw=0.6)
                ax.plot(v, yy, marker=C.MODEL_MARKER[r["model"]], ms=3.2, ls="None", mew=0.5,
                        mfc=col if deg else "white", mec="black" if deg else col)
            else:
                lo = hi = None; deg = False
                ax.plot(v, yy, marker="x", color=C.NE_GREY, ms=3.6, mew=0.9, ls="None")
            REC.setdefault("S7", []).append(dict(metric=title, domain=r["domain"], model=r["model"],
                                                 seed=int(r["seed"]), v=v, lo=lo, hi=hi, status=r[st], deg=deg))
        ax.set_title(f"{title} ({'lower' if sign < 0 else 'higher'} = worse)", loc="left")
        ax.set_xlabel(ylab)
        ax.grid(axis="x", color="0.9", lw=0.5)
        C.panel_label(ax, "abc"[j])
    ticks = [ypos[(dn, s)] for dn in doms for s in SEEDS]
    axs[0].set_yticks(ticks, [str(s) for dn in doms for s in SEEDS])
    axs[0].set_ylim(max(ticks) + 0.8, -0.8)
    for dname in doms:
        ym = np.mean([ypos[(dname, s)] for s in SEEDS])
        axs[0].text(-0.36, ym, C.DOMAIN_LABEL[dname].replace("\n", " "), transform=axs[0].get_yaxis_transform(),
                    rotation=90, ha="center", va="center", fontsize=8.5, fontweight="bold")
    axs[0].set_ylabel("")
    extra = [Line2D([0], [0], marker="o", mfc="0.3", mec="black", ls="None", ms=4.5, label="p < 0.05 and degrading"),
             Line2D([0], [0], marker="o", mfc="white", mec="0.3", ls="None", ms=4.5, label="p ≥ 0.05"),
             Line2D([0], [0], marker="x", color=C.NE_GREY, ls="None", ms=4.5, mew=0.9, label="NE")]
    fig.legend(handles=model_handles(extra), loc="outside upper center", ncol=4, frameon=False, handletextpad=0.3,
               columnspacing=0.9, title="Primary 10% budget; seeds listed within each domain; bars = 95% intervals")
    rows = raw("SUPP_H4_MODEL_SEED_all_source.csv")
    kk = {(x["domain"], x["model"], int(x["seed"])): x for x in rows}
    fmap = {t: (f, c, s_) for f, c, s_, _, t, _, _ in H4_METRICS}
    ok = all(float(kk[(e["domain"], e["model"], e["seed"])][fmap[e["metric"]][0]]) == e["v"] and
             kk[(e["domain"], e["model"], e["seed"])][fmap[e["metric"]][2]] == e["status"] and
             (e["lo"] is None or C.ci(kk[(e["domain"], e["model"], e["seed"])][fmap[e["metric"]][1]]) == (e["lo"], e["hi"]))
             for e in REC["S7"])
    ne_lift = sum(1 for e in REC["S7"] if e["metric"] == "Lift" and e["status"] != "ESTIMABLE")
    chk("S7: 5 domains x 20 settings x 3 metrics; values/CIs/status exact; BaitBuster lift NE 20/20",
        len(REC["S7"]) == 300 and ok and ne_lift == 20, f"lift NE={ne_lift}")
    return fig


# ------------------------------------------------------------------ S8: risk-coverage curves
RISK_DOMS = ["source_test"] + H3P_TARGETS
SEED_STYLE = {42: ("#000000", "-"), 137: ("#3B528B", (0, (5, 1.5))), 2718: ("#21908C", (0, (1.5, 1.2))),
              31415: ("#5DC863", (0, (6, 1.5, 1.5, 1.5))), 65537: ("#B8A000", (0, (3, 1)))}


def s8():
    d = C.load("SUPP_RISK_COVERAGE_all_source.csv")
    fig, axs = plt.subplots(6, 4, figsize=(C.FIG_W, 8.6), sharex=True, sharey="row", constrained_layout=True)
    n_pts = 0
    for i, dname in enumerate(RISK_DOMS):
        for j, m in enumerate(C.MODELS):
            ax = axs[i, j]
            sub = d[(d.domain == dname) & (d.model == m)]
            for s in SEEDS:
                g = sub[sub.seed == s].sort_values("coverage", kind="mergesort")
                assert len(g) > 0
                col, ls = SEED_STYLE[s]
                ax.plot(g.coverage, g.risk, color=col, ls=ls, lw=0.8)
                n_pts += len(g)
                REC.setdefault("S8", []).append(dict(domain=dname, model=m, seed=s, n=len(g),
                                                     first=(float(g.coverage.iloc[0]), float(g.risk.iloc[0])),
                                                     last=(float(g.coverage.iloc[-1]), float(g.risk.iloc[-1]))))
            ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
            ax.grid(color="0.92", lw=0.5)
            if i == 0:
                ax.set_title(C.MODEL_LABEL[m], color=C.MODEL_COLOR[m], fontweight="bold")
            if j == 0:
                ax.set_ylabel(C.DOMAIN_LABEL[dname].replace("\n", " ") + "\nrisk")
            if i == 5:
                ax.set_xlabel("coverage")
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    hs = [Line2D([0], [0], color=SEED_STYLE[s][0], ls=SEED_STYLE[s][1], lw=1.2, label=str(s)) for s in SEEDS]
    fig.legend(handles=hs, loc="outside upper center", ncol=5, frameon=False,
               title="Seed (every model × seed curve shown; no averaging); y-axis shared within each domain row")
    rows = raw("SUPP_RISK_COVERAGE_all_source.csv")
    chk("S8: 120 curves (6 domains x 4 models x 5 seeds); all 24,120 points plotted",
        len(REC["S8"]) == 120 and n_pts == len(rows) == 24120, f"points={n_pts}")
    return fig


# ------------------------------------------------------------------ S9: bridge leave-one-platform-out
LOO_PAIRS = [("abs_delta_rd__vs__lift_deterioration", "|ΔRD| vs lift deterioration",
              "FIG05_H3_H4_BRIDGE_abs_delta_rd__vs__lift_deterioration_source.csv"),
             ("abs_delta_rd__vs__error_capture_deterioration", "|ΔRD| vs error-capture deterioration",
              "FIG05_H3_H4_BRIDGE_abs_delta_rd__vs__error_capture_deterioration_source.csv"),
             ("abs_delta_nll_gap__vs__lift_deterioration", "|ΔNLL-gap| vs lift deterioration",
              "FIG05_H3_H4_BRIDGE_abs_delta_nll_gap__vs__lift_deterioration_source.csv"),
             ("abs_delta_nll_gap__vs__error_capture_deterioration", "|ΔNLL-gap| vs error-capture deterioration",
              "FIG05_H3_H4_BRIDGE_abs_delta_nll_gap__vs__error_capture_deterioration_source.csv")]


def s9():
    d = C.load("SUPP_BRIDGE_LOO_all_source.csv")
    assert len(d) == 16
    fig, axs = plt.subplots(2, 2, figsize=(C.FIG_W, 4.6), constrained_layout=True)
    for k_, (ax, (pair, title, full_csv)) in enumerate(zip(axs.ravel(), LOO_PAIRS)):
        full = float(C.load(full_csv).canonical_platform_fe_slope.iloc[0])
        ax.axhline(0, color="black", lw=0.6)
        ax.axhline(full, color="0.35", lw=0.9, ls=(0, (4, 2)))
        sub = d[d.pair == pair].set_index("omitted_platform")
        for i, p in enumerate(C.PLATFORMS):
            v = float(sub.loc[p, "point_platform_fe_slope"])
            ax.plot(i, v, marker=C.PLATFORM_MARKER[p], color=C.PLATFORM_COLOR[p], mec="black", mew=0.4, ms=6, ls="None")
            REC.setdefault("S9", []).append(dict(pair=pair, omitted=p, v=v, retained=int(sub.loc[p, "retained_cells"])))
        REC["S9"].append(dict(pair=pair, full=full))
        ax.set_xticks(range(4), [f"without\n{C.PLATFORM_LABEL[p]}" for p in C.PLATFORMS])
        ax.set_xlim(-0.5, 3.5)
        ax.set_title(title, loc="left")
        ax.set_ylabel("platform-FE point slope")
        lo, hi = ax.get_ylim(); pad = 0.12 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        C.panel_label(ax, "abcd"[k_])
    fig.legend(handles=[Line2D([0], [0], color="0.35", lw=0.9, ls=(0, (4, 2)),
                               label="slope with all four platforms (Figure 5)")],
               loc="outside upper center", frameon=False,
               title="Leave-one-platform-out point slopes (24 retained cells each)")
    fig.supxlabel("Secondary point-estimate robustness checks only; every bridge inference is NON_ESTIMABLE "
                  "(no interval or p-value).", fontsize=8.5)
    rows = raw("SUPP_BRIDGE_LOO_all_source.csv")
    kk = {(x["pair"], x["omitted_platform"]): x for x in rows}
    pts = [e for e in REC["S9"] if "omitted" in e]
    ok = all(float(kk[(e["pair"], e["omitted"])]["point_platform_fe_slope"]) == e["v"] and
             int(kk[(e["pair"], e["omitted"])]["retained_cells"]) == e["retained"] == 24 for e in pts)
    chk("S9: 16 LOO points exact (4 pairs x 4 omitted platforms); 24 retained cells each", len(pts) == 16 and ok)
    return fig


# ------------------------------------------------------------------ S10: prevalence / structural sensitivity
ADJ = ["unadjusted", "source_prior_standardized", "structural_composition_standardized"]
ADJ_LABEL = ["Unadjusted", "Source-prior", "Structural"]


def s10():
    d = C.load("SUPP_SENSITIVITY_A_B_source.csv")
    assert len(d) == 300
    doms = H3P_TARGETS
    fig, axs = plt.subplots(5, 2, figsize=(C.FIG_W, 8.4), sharex=True, constrained_layout=True)
    for i, dname in enumerate(doms):
        for j, met in enumerate(["auroc", "nll"]):
            ax = axs[i, j]
            for m in C.MODELS:
                for s in SEEDS:
                    ss = d[(d.domain == dname) & (d.model == m) & (d.seed == s)].set_index("adjustment").reindex(ADJ)
                    ax.plot(range(3), ss[met], color=C.MODEL_COLOR[m], ls=SEED_STYLE[s][1], lw=0.7, alpha=0.9,
                            marker=C.MODEL_MARKER[m], ms=3, mec="black", mew=0.25)
                    for a_, v in zip(ADJ, ss[met]):
                        REC.setdefault("S10", []).append(dict(domain=dname, model=m, seed=s, adj=a_, met=met,
                                                              v=float(v)))
            ax.set_xticks(range(3), ADJ_LABEL)
            ax.set_xlim(-0.25, 2.25)
            ax.grid(axis="y", color="0.92", lw=0.5)
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            if i == 0:
                ax.set_title("AUROC" if met == "auroc" else "NLL", fontweight="bold")
            if j == 0:
                ax.set_ylabel(C.DOMAIN_LABEL[dname].replace("\n", " ") + "\nAUROC")
            else:
                ax.set_ylabel("NLL")
    seed_h = [Line2D([0], [0], color="0.3", ls=SEED_STYLE[s][1], lw=1, label=str(s)) for s in SEEDS]
    fig.legend(handles=model_handles() + seed_h, loc="outside upper center", ncol=5, frameon=False,
               title="Colour/marker = model; line style = seed; every domain × model × seed row shown")
    rows = raw("SUPP_SENSITIVITY_A_B_source.csv")
    kk = {(x["domain"], x["model"], int(x["seed"]), x["adjustment"]): x for x in rows}
    ok = all(float(kk[(e["domain"], e["model"], e["seed"], e["adj"])][e["met"]]) == e["v"] for e in REC["S10"])
    chk("S10: 300 rows x {AUROC, NLL} = 600 values exact", len(REC["S10"]) == 600 and ok)
    return fig


if __name__ == "__main__":
    out = {}
    for stem, f in [("SUPP_S1_H1_MODEL_SEED_v2", s1), ("SUPP_S2_H3_SPARSITY_v2", s2),
                    ("SUPP_S4_BAITBUSTER_H3_v2", s4),
                    ("SUPP_S5_H4_05PCT_v2", lambda: s_h4("SUPP_H4_05PCT_A_C_source.csv", "5%", "S5")),
                    ("SUPP_S6_H4_20PCT_v2", lambda: s_h4("SUPP_H4_20PCT_A_C_source.csv", "20%", "S6")),
                    ("SUPP_S3a_H3P_CURIOSITY_v2", lambda: s3("curiosity_information_withholding", "S3a")),
                    ("SUPP_S3b_H3P_DIRECT_ADDRESS_v2", lambda: s3("direct_address_imperative", "S3b")),
                    ("SUPP_S3c_H3P_PUNCTUATION_v2", lambda: s3("punctuation_intensity", "S3c")),
                    ("SUPP_S7_H4_MODEL_SEED_v2", s7), ("SUPP_S8_RISK_COVERAGE_v2", s8),
                    ("SUPP_S9_BRIDGE_LOO_v2", s9), ("SUPP_S10_SENSITIVITY_v2", s10)]:
        fig = f(); out[stem] = {k: v.name for k, v in C.export(fig, stem).items()}; plt.close(fig)
    ok = all(c["pass"] for c in CHECKS)
    (C.OUT / "SUPP_PARITY_REPORT.json").write_text(json.dumps({"all_pass": ok, "checks": CHECKS,
                                                              "sources_sha256": C.SOURCE_LOG, "outputs": out}, indent=1),
                                                  encoding="utf8")
    for c in CHECKS:
        print("PASS" if c["pass"] else "FAIL", c["check"], c["detail"])
    sys.exit(0 if ok else 1)
