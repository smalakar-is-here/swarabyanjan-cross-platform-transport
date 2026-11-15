"""Redesigned main Figures 2-5 (presentation only; frozen data; no new statistics).

Run:  python scripts/publication_figures/make_main_figures.py   (output: $SWFIG_OUT, default build/publication_figures)
Outputs PDF/SVG/PNG plus a JSON record of every plotted element.
"""
from __future__ import annotations

import json
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import swfig_common as C

PLOTTED: dict[str, list] = {}   # figure -> list of plotted element records (for parity tests)


def rec(fig, **kw):
    PLOTTED.setdefault(fig, []).append(kw)


def model_handles(extra=()):
    hs = [Line2D([0], [0], marker=C.MODEL_MARKER[m], color=C.MODEL_COLOR[m], linestyle="None",
                 markersize=5, markeredgecolor="black", markeredgewidth=0.4, label=C.MODEL_LABEL[m])
          for m in C.MODELS]
    return hs + list(extra)


def seed_offsets(n=5, width=0.12):
    return np.linspace(-width / 2, width / 2, n)


MODEL_DODGE = {m: o for m, o in zip(C.MODELS, np.linspace(-0.3, 0.3, 4))}


# --------------------------------------------------------------------------- Figure 2 (H1/H2)
def fig2():
    a = C.load("FIG02_H1_H2_A_source.csv")
    bc = C.load("FIG02_H1_H2_B_C_source.csv")
    d = C.load("FIG02_H1_H2_D_source.csv")
    fig, axs = plt.subplots(2, 2, figsize=(C.FIG_W, 5.3), constrained_layout=True)
    domains_a = ["source_test"] + C.TARGETS

    def frame(ax, domains):
        ax.set_xticks(range(len(domains)), [C.DOMAIN_LABEL[x] for x in domains])
        ax.set_xlim(-0.55, len(domains) - 0.45)
        bb = domains.index("baitbuster:youtube_human")
        ax.axvspan(bb - 0.5, bb + 0.5, color="#F2F2F2", zorder=0, lw=0)
        ax.axvline(bb - 0.5, color="0.55", lw=0.6, ls=(0, (3, 2)))
        for i in range(len(domains) - 1):
            if i + 0.5 != bb - 0.5:
                ax.axvline(i + 0.5, color="0.9", lw=0.5, zorder=0)
        return bb

    # A: calibrated AUROC, every model x seed point (no aggregation)
    ax = axs[0, 0]
    frame(ax, domains_a)
    so = seed_offsets()
    for r in a.itertuples():
        x = domains_a.index(r.domain) + MODEL_DODGE[r.model] + so[[42, 137, 2718, 31415, 65537].index(r.seed)]
        ax.plot(x, r.calibrated_auroc, marker=C.MODEL_MARKER[r.model], color=C.MODEL_COLOR[r.model],
                ms=3.4, mec="black", mew=0.3, ls="None")
        rec("FIG02", panel="A", domain=r.domain, model=r.model, seed=int(r.seed), y=float(r.calibrated_auroc))
    ax.axhline(0.70, color="black", lw=0.7, ls=(0, (4, 2)))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Calibrated AUROC")
    ax.set_title("H1 discrimination (point estimates)", loc="left")
    # BaitBuster: the dots are calibrated-AUROC point estimates; the H1 AUROC-drop inference is NE (20/20).
    bbx = domains_a.index("baitbuster:youtube_human")
    lo_y, hi_y = ax.get_ylim()
    ax.set_ylim(lo_y, hi_y + 0.07 * (hi_y - lo_y))
    ax.text(bbx, hi_y + 0.035 * (hi_y - lo_y), "drop: NE", ha="center", va="center", fontsize=8, color="0.25")
    C.panel_label(ax, "a")

    # B, C: H1 relative changes with frozen 95% CIs; NE endpoints drawn as grey crosses without CI
    def interval_panel(ax, df, ycol, cicol, estcol, title, ylab, letter, thr=None, thr_label=None, mark_support=False):
        frame(ax, C.TARGETS)
        for r in df.itertuples():
            m = r.model
            x = C.TARGETS.index(r.domain) + MODEL_DODGE[m] + so[[42, 137, 2718, 31415, 65537].index(r.seed)]
            y = float(getattr(r, ycol))
            est = C.is_true(getattr(r, estcol)) if estcol else True
            lo, hi = C.ci(getattr(r, cicol))
            if est and not math.isnan(lo):
                ax.plot([x, x], [lo, hi], color=C.MODEL_COLOR[m], lw=0.7, alpha=0.9, solid_capstyle="butt")
                ax.plot(x, y, marker=C.MODEL_MARKER[m], color=C.MODEL_COLOR[m], ms=3.0, mec="black", mew=0.3, ls="None")
            else:
                ax.plot(x, y, marker="x", color=C.NE_GREY, ms=3.4, mew=0.8, ls="None")
            if mark_support and C.is_true(r.setting_supported):
                ax.plot(x, y, marker="o", mfc="none", mec="black", ms=7, mew=0.8, ls="None")
            rec("FIG02", panel=letter, domain=r.domain, model=m, seed=int(r.seed), y=y,
                lo=None if math.isnan(lo) else lo, hi=None if math.isnan(hi) else hi, estimable=bool(est))
        ax.axhline(0, color="black", lw=0.6)
        if thr is not None:
            ax.axhline(thr, color="black", lw=0.7, ls=(0, (4, 2)))
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left")
        C.panel_label(ax, letter)

    interval_panel(axs[0, 1], bc, "nll_relative_change", "nll_relative_change_ci95", "nll_relative_change_estimable",
                   "H1 relative NLL change", "(target − source) / source", "b", 0.10, "H1 threshold +0.10",
                   mark_support=True)
    interval_panel(axs[1, 0], bc, "balanced_brier_relative_change", "balanced_brier_relative_change_ci95",
                   "balanced_brier_relative_change_estimable", "H1 relative balanced-Brier change",
                   "(target − source) / source", "c", 0.10, "H1 threshold +0.10")
    axs[1, 0].text(4, axs[1, 0].get_ylim()[1] * 0.97, "NE\n(20/20)", ha="center", va="top", fontsize=8, color="0.3")
    interval_panel(axs[1, 1], d, "target_minus_source_gain", "h2_target_minus_source_gain_ci95", None,
                   "H2 calibration-gain transport", "target − source NLL gain", "d")
    axs[1, 1].text(0.02, 0.98, "0/100 settings meet the locked H2 criterion", transform=axs[1, 1].transAxes,
                   ha="left", va="top", fontsize=8)

    extra = [Line2D([0], [0], marker="x", color=C.NE_GREY, ls="None", ms=4, mew=0.8, label="NE (no interval)"),
             Line2D([0], [0], marker="o", mfc="none", mec="black", ls="None", ms=6, mew=0.8, label="H1-supporting seed"),
             Line2D([0], [0], color="black", lw=0.7, ls=(0, (4, 2)), label="H1 threshold"),
             Patch(facecolor="#F2F2F2", edgecolor="0.55", ls="--", lw=0.6, label="Related construct")]
    fig.legend(handles=model_handles(extra), loc="outside upper center", ncol=4, frameon=False,
               handletextpad=0.3, columnspacing=0.9, fontsize=8)
    return fig


# --------------------------------------------------------------------------- Figure 3 (H3)
def fig3():
    h3 = C.load("FIG03_H3_CUE_TRANSPORT_A_B_source.csv")
    assert len(h3) == 32
    gap = 0.9
    rows = []  # (y, cue, platform)
    y = 0.0
    cue_mid = {}
    for cue in C.CUES:
        ys = []
        for p in C.PLATFORMS:
            rows.append((y, cue, p)); ys.append(y); y += 1
        cue_mid[cue] = (ys[0], ys[-1])
        y += gap
    ymax = y - gap
    fig, axs = plt.subplots(1, 2, figsize=(C.FIG_W, 7.1), sharey=True, constrained_layout=True,
                            gridspec_kw={"wspace": 0.04})
    specs = [("delta_rd", "delta_rd_ci95", "ΔRD (target − source)", "Cue–label transport, ΔRD", "a"),
             ("ebm_delta_nll_gap_mean_5seeds", "ebm_delta_nll_gap_ci95", "ΔNLL-gap (target − source)",
              "Proper-loss transport, EBM ΔNLL-gap", "b")]
    for ax, (vcol, cicol, xlab, title, letter) in zip(axs, specs):
        for i, cue in enumerate(C.CUES):
            lo_y, hi_y = cue_mid[cue]
            if i % 2 == 0:
                ax.axhspan(lo_y - 0.5 - gap / 2 + 0.45, hi_y + 0.5 + gap / 2 - 0.45, color="#F5F5F5", zorder=0, lw=0)
        ax.axvline(0, color="black", lw=0.7, zorder=1)
        for (yy, cue, p) in rows:
            r = h3[(h3.cue == cue) & (h3.platform == p)].iloc[0]
            est = r.inferential_status == "ESTIMABLE"
            if est:
                lo, hi = C.ci(r[cicol])
                v = float(r[vcol])
                ax.plot([lo, hi], [yy, yy], color=C.PLATFORM_COLOR[p], lw=1.1, solid_capstyle="butt", zorder=2)
                ax.plot(v, yy, marker=C.PLATFORM_MARKER[p], color=C.PLATFORM_COLOR[p], ms=4.2, mec="black",
                        mew=0.4, ls="None", zorder=3)
                rec("FIG03", panel=letter, cue=cue, platform=p, value=v, lo=lo, hi=hi, status=r.inferential_status)
            else:
                ax.text(0.985, yy, "NE", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8,
                        color="0.25", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=C.NE_GREY, lw=0.6), zorder=4)
                rec("FIG03", panel=letter, cue=cue, platform=p, value=None, lo=None, hi=None, status=r.inferential_status)
        ax.set_xlabel(xlab)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", color="0.88", lw=0.5, zorder=0)
        ax.set_ylim(ymax + 0.6, -0.6)
        ax.tick_params(axis="y", length=0)
        C.panel_label(ax, letter)
    axs[0].set_yticks([(a + b) / 2 for a, b in cue_mid.values()], [C.CUE_LABEL[c] for c in C.CUES])
    plat_h = [Line2D([0], [0], marker=C.PLATFORM_MARKER[p], color=C.PLATFORM_COLOR[p], mec="black", mew=0.4,
                     lw=1.1, ms=5, label=C.PLATFORM_LABEL[p]) for p in C.PLATFORMS]
    plat_h.append(Patch(fc="white", ec=C.NE_GREY, lw=0.6, label="NE: non-estimable under the locked bootstrap"))
    fig.supxlabel("Cells: 24/32 estimable, 8/32 NE.  Global 32-dimensional ΔRD Wald test: NON_ESTIMABLE.\n"
                  "Eight-family Simes/BH: NON_ESTIMABLE for ΔRD and for ΔNLL-gap.", fontsize=8.5, fontweight="bold")
    fig.legend(handles=plat_h, loc="outside upper center", ncol=5, frameon=False, handletextpad=0.4,
               columnspacing=1.0, title="Platform (rows within each cue: News, Facebook, YouTube, Twitter); "
                                        "bars = frozen 95% bootstrap intervals")
    return fig


# --------------------------------------------------------------------------- Figure 4 (H4)
H4_METRICS = [
    # field, ci, status, p, title, ylabel, degradation sign (-1: lower is worse, +1: higher is worse)
    ("target_minus_source_lift", "lift_ci95", "lift_estimability_status", "lift_p", "Lift", "target − source lift", -1),
    ("target_minus_source_error_capture", "error_capture_ci95", "error_capture_estimability_status",
     "error_capture_p", "Error capture", "target − source error capture", -1),
    ("target_minus_source_aurc", "aurc_ci95", "aurc_estimability_status", "aurc_p", "AURC", "target − source AURC", +1),
]


def degrading(r, field, status, p, sign) -> bool:
    """Table 4 definition: canonical p < 0.05 AND point estimate in the degradation direction."""
    if r[status] != "ESTIMABLE" or (isinstance(r[p], float) and math.isnan(r[p])):
        return False
    return float(r[p]) < 0.05 and sign * float(r[field]) > 0


def fig4():
    prim = C.load("FIG04_H4_FIXED_BUDGET_primary_10pct_source.csv")
    sens = C.load("FIG04_H4_FIXED_BUDGET_budget_sensitivity_source.csv")
    t4 = C.load_table("main/TABLE04_H4_PRIMARY_10PCT_SUMMARY.csv")
    assert len(prim) == 80 and len(sens) == 240
    fig = plt.figure(figsize=(C.FIG_W, 6.6), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.7, 1.0])
    doms = [f"banglaclick:{p}" for p in C.PLATFORMS]
    so = seed_offsets(width=0.1)
    seeds = [42, 137, 2718, 31415, 65537]
    metric_key = {"Lift": "lift", "Error capture": "error_capture", "AURC": "aurc"}
    for j, (field, cicol, st, pc, title, ylab, sign) in enumerate(H4_METRICS):
        ax = fig.add_subplot(gs[j, 0])
        for r in prim.to_dict("records"):
            x = doms.index(r["domain"]) + MODEL_DODGE[r["model"]] + so[seeds.index(r["seed"])]
            y = float(r[field]); lo, hi = C.ci(r[cicol]); deg = degrading(r, field, st, pc, sign)
            col = C.MODEL_COLOR[r["model"]]
            ax.plot([x, x], [lo, hi], color=col, lw=0.7, solid_capstyle="butt")
            ax.plot(x, y, marker=C.MODEL_MARKER[r["model"]], ms=3.2, ls="None", mew=0.5,
                    mfc=col if deg else "white", mec="black" if deg else col)
            rec("FIG04", panel=f"primary_{metric_key[title]}", domain=r["domain"], model=r["model"], seed=int(r["seed"]),
                y=y, lo=lo, hi=hi, degrading=deg)
        ax.axhline(0, color="black", lw=0.6)
        # y-range = union of primary intervals and all sensitivity points, so nothing is clipped in either panel.
        lo_all = min([C.ci(v)[0] for v in prim[cicol]] + list(sens[field]) + [0.0])
        hi_all = max([C.ci(v)[1] for v in prim[cicol]] + list(sens[field]) + [0.0])
        m = 0.05 * (hi_all - lo_all)
        ymin, ymax = lo_all - m, hi_all + m; pad = 0.2 * (ymax - ymin)
        ax.set_ylim(ymin, ymax + pad)
        for i, dname in enumerate(doms):
            sub = prim[prim.domain == dname]
            n_deg = sum(degrading(r, field, st, pc, sign) for r in sub.to_dict("records"))
            t4row = t4[(t4.domain == dname) & (t4.metric == metric_key[title])].iloc[0]
            assert n_deg == int(t4row.canonical_p_lt_0_05_and_degradation_direction), (dname, title, n_deg)
            ax.text(i, ymax + pad * 0.55, f"{n_deg}/{len(sub)} degrading", ha="center", va="center", fontsize=8,
                    fontweight="bold")
            rec("FIG04", panel=f"count_{metric_key[title]}", domain=dname, n_degrading=n_deg, n=len(sub))
            if i:
                ax.axvline(i - 0.5, color="0.9", lw=0.5, zorder=0)
        ax.set_xticks(range(4), [C.PLATFORM_LABEL[p] for p in C.PLATFORMS])
        ax.set_xlim(-0.5, 3.5)
        arrow = "lower = worse" if sign < 0 else "higher = worse"
        ax.set_title(f"{title} at the primary 10% budget ({arrow})", loc="left")
        ax.set_ylabel(ylab)
        C.panel_label(ax, "ace"[j])

        # Sensitivity: 5% / 10% / 20%, no intervals, smaller and lighter so the primary panel dominates.
        ax2 = fig.add_subplot(gs[j, 1], sharey=ax)
        boff = {0.05: -0.28, 0.10: 0.0, 0.20: 0.28}
        for r in sens.to_dict("records"):
            b = float(r["budget"])
            x = doms.index(r["domain"]) + boff[b] + so[seeds.index(r["seed"])] * 0.4 + MODEL_DODGE[r["model"]] * 0.12
            y = float(r[field]); deg = degrading(r, field, st, pc, sign)
            if math.isclose(b, 0.10):
                ax2.plot(x, y, marker="o", ms=1.8, ls="None", color="black", mew=0)
            else:
                ax2.plot(x, y, marker="o", ms=1.8, ls="None", mfc="#9A9A9A" if deg else "white", mec="#6E6E6E", mew=0.4)
            rec("FIG04", panel=f"budget_{metric_key[title]}", domain=r["domain"], budget=b, model=r["model"],
                seed=int(r["seed"]), y=y, degrading=deg)
        ax2.axhline(0, color="black", lw=0.6)
        ax2.set_xticks(range(4), ["News", "Faceb.", "YouT.", "Twitter"])
        ax2.tick_params(axis="y", labelleft=False)
        for i in range(1, 4):
            ax2.axvline(i - 0.5, color="0.9", lw=0.5, zorder=0)
        ax2.set_xlim(-0.5, 3.5)
        ax2.set_title("Budgets 5 | 10 | 20% (left to right)", loc="left", fontsize=8, color="0.25")
        C.panel_label(ax2, "bdf"[j])
    extra = [Line2D([0], [0], marker="o", mfc="0.3", mec="black", ls="None", ms=4.5, label="p < 0.05 and degrading"),
             Line2D([0], [0], marker="o", mfc="white", mec="0.3", ls="None", ms=4.5, label="other estimable"),
             Line2D([0], [0], marker="o", color="black", ls="None", ms=3, label="10% (right panels)"),
             Line2D([0], [0], marker="o", mfc="white", mec="#6E6E6E", ls="None", ms=3, label="5% / 20% (right panels)")]
    fig.legend(handles=model_handles(extra), loc="outside upper center", ncol=4, frameon=False,
               handletextpad=0.3, columnspacing=1.0)
    return fig


# --------------------------------------------------------------------------- Figure 5 (bridge)
BRIDGE = [
    ("FIG05_H3_H4_BRIDGE_abs_delta_rd__vs__lift_deterioration_source.csv", "abs_delta_rd",
     "lift_deterioration_source_minus_target", "|ΔRD|", "Lift deterioration\n(source − target)"),
    ("FIG05_H3_H4_BRIDGE_abs_delta_rd__vs__error_capture_deterioration_source.csv", "abs_delta_rd",
     "error_capture_deterioration_source_minus_target", "|ΔRD|", "Error-capture deterioration\n(source − target)"),
    ("FIG05_H3_H4_BRIDGE_abs_delta_nll_gap__vs__lift_deterioration_source.csv", "abs_delta_nll_gap",
     "lift_deterioration_source_minus_target", "|ΔNLL-gap|", "Lift deterioration\n(source − target)"),
    ("FIG05_H3_H4_BRIDGE_abs_delta_nll_gap__vs__error_capture_deterioration_source.csv", "abs_delta_nll_gap",
     "error_capture_deterioration_source_minus_target", "|ΔNLL-gap|", "Error-capture deterioration\n(source − target)"),
]


def fig5():
    fig, axs = plt.subplots(2, 2, figsize=(C.FIG_W, 5.0), constrained_layout=True)
    for k, (ax, (fn, xc, yc, xl, yl)) in enumerate(zip(axs.ravel(), BRIDGE)):
        df = C.load(fn)
        assert len(df) == 32 and df.inferential_status.nunique() == 1
        for r in df.itertuples():
            p = r.platform
            ax.plot(getattr(r, xc), getattr(r, yc), marker=C.PLATFORM_MARKER[p], color=C.PLATFORM_COLOR[p],
                    ms=4.2, mec="black", mew=0.35, ls="None", alpha=0.95)
            rec("FIG05", panel="abcd"[k], cue=r.cue, platform=p, x=float(getattr(r, xc)), y=float(getattr(r, yc)))
        slope = float(df.canonical_platform_fe_slope.iloc[0])
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(f"Platform-FE point slope = {slope:.6f}", loc="left")
        rec("FIG05", panel="abcd"[k], slope=slope, status=df.inferential_status.iloc[0])
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_xlim(left=-0.03)
        C.panel_label(ax, "abcd"[k])
    hs = [Line2D([0], [0], marker=C.PLATFORM_MARKER[p], color=C.PLATFORM_COLOR[p], mec="black", mew=0.35,
                 ls="None", ms=5, label=C.PLATFORM_LABEL[p]) for p in C.PLATFORMS]
    fig.legend(handles=hs, loc="outside upper center", ncol=4, frameon=False,
               title="Platform (one point per cue × platform cell, 32 per panel; no fitted line)")
    fig.supxlabel("Descriptive only: all four joint-bootstrap analyses are NON_ESTIMABLE "
                  "(no interval or p-value for any slope).", fontsize=8, fontweight="bold")
    return fig


if __name__ == "__main__":
    made = {}
    for stem, fn in [("FIG02_H1_H2_v2", fig2), ("FIG03_H3_CUE_TRANSPORT_v2", fig3),
                     ("FIG04_H4_FIXED_BUDGET_v2", fig4), ("FIG05_H3_H4_BRIDGE_v2", fig5)]:
        f = fn(); made[stem] = {k: v.name for k, v in C.export(f, stem).items()}
        plt.close(f)
        print("wrote", stem)
    (C.OUT / "main_figures_plotted_elements.json").write_text(json.dumps(
        {"sources_sha256": C.SOURCE_LOG, "outputs": made, "plotted": PLOTTED}, indent=1, default=float), encoding="utf8")
