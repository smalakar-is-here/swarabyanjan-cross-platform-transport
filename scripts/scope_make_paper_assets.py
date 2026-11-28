"""Generate every figure and table of the SCOPE manuscript from results/scope_v1 (aggregate files only).

Usage: python scripts/scope_make_paper_assets.py
Writes $SCOPE_PAPER_DIR/figures/*.pdf, tables/*.tex and numbers.tex. No value is typed by hand.
The manuscript itself is not part of the public repository; SCOPE_PAPER_DIR points at its private location.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results/scope_v1"
import os
PAPER = Path(os.environ.get("SCOPE_PAPER_DIR", ROOT / "paper_scope"))
FIG = PAPER / "figures"; TAB = PAPER / "tables"; FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "axes.linewidth": 0.6})
C_R6, C_AL, C_SRC, C_GREY = "#B4532A", "#1F5F8B", "#2E2E2E", "#9A9A9A"
MODELS = ["ebm", "tfidf_lr", "banglabert", "xlmr"]
MNAME = {"ebm": "EBM", "tfidf_lr": "TF-IDF LR", "banglabert": "BanglaBERT", "xlmr": "XLM-R", "majority": "Majority"}
ENVS = ["banglaclick:news", "banglaclick:facebook", "banglaclick:youtube", "banglaclick:twitter"]
ENAME = {"banglabait:news": "BanglaBait News (A)", "banglaclick:news": "News (B)", "banglaclick:facebook": "Facebook (B)",
         "banglaclick:youtube": "YouTube (B)", "banglaclick:twitter": "Twitter (B)", "baitbuster:youtube": "YouTube (C)",
         "banglaclick:news_nonanchor": "News (B), non-anchor"}
W1, W2 = 3.45, 7.0  # IEEE column / page width (in)


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight"); fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def tex_table(df, name, caption, label, colfmt=None, notes=None, wide=False):
    colfmt = colfmt or "l" + "r" * (df.shape[1] - 1)
    env = "table*" if wide else "table"
    lines = [r"\begin{" + env + "}[!htbp]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}", r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{" + colfmt + "}", r"\toprule",
             " & ".join(df.columns) + r" \\", r"\midrule"]
    for _, row in df.iterrows():
        lines.append(" & ".join(str(v) for v in row.values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}"]
    if notes:
        lines.append(r"\par\vspace{2pt}{\footnotesize " + notes + "}")
    lines.append(r"\end{" + env + "}")
    (TAB / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def f3(x):
    return f"{x:.3f}"


# ---------------------------------------------------------------- Fig 1: the phenomenon
lk = pd.read_csv(R / "E1/linking_function.csv"); ta = pd.read_csv(R / "E1/threshold_agreement.csv")
e3 = pd.read_csv(R / "E3/prevalence_estimates_summary.csv")
fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3))
a = ax[0]
a.bar(lk.level, lk.g_isotonic, color=C_AL, width=0.6)
a.errorbar(lk.level, lk.g_isotonic, yerr=[lk.g_isotonic - lk.ci_cluster_low, lk.ci_cluster_high - lk.g_isotonic], fmt="none", ecolor="k", lw=0.8, capsize=2)
for l, n in zip(lk.level, lk.n):
    a.text(l, 0.02, f"n={n}", ha="center", va="bottom", fontsize=6.5, color="white")
a.axhline(0.5, color=C_GREY, lw=0.6, ls=":")
a.set_xticks(range(4)); a.set_xlabel("BanglaClick rubric level $L$")
a.set_ylabel(r"$\hat g(L)=P(Y_A=1\mid L)$"); a.set_ylim(0, 1)
a.set_title("(a) Source label on 1,083 anchor title classes", loc="left")
a = ax[1]
cells = ENVS
x = np.arange(len(cells)); w = 0.26
r6 = [e3[(e3.env == c) & (e3.estimator == "pcc")].true_r6.iloc[0] for c in cells]
al = [e3[(e3.env == c) & (e3.estimator == "pcc")].true_aligned_hard.iloc[0] for c in cells]
pc = [e3[(e3.env == c) & (e3.estimator == "pcc")].estimate.iloc[0] for c in cells]
a.bar(x - w, r6, w, color=C_R6, label=r"published label $1[L\geq1]$")
a.bar(x, al, w, color=C_AL, label=r"aligned label $1[L\geq2]$")
a.bar(x + w, pc, w, color=C_GREY, label="unlabelled estimate (PCC)")
a.axhline(0.341, color=C_SRC, lw=0.8, ls="--"); a.text(3.45, 0.355, "source", fontsize=6.5, ha="right")
a.set_xticks(x); a.set_xticklabels([ENAME[c].replace(" (B)", "") for c in cells]); a.set_ylim(0, 1)
a.set_ylabel("positive prevalence"); a.legend(frameon=False, loc="upper left", ncol=1)
a.set_title("(b) Target prevalence under two label definitions", loc="left")
fig.tight_layout(); save(fig, "fig1_phenomenon")

t1 = ta.copy(); t1["threshold"] = [r"$L\geq1$ (published)", r"$L\geq2$", r"$L\geq3$"]
bs = json.loads((R / "E1/bootstrap_summary.json").read_text())
kc = bs["publisher_cluster_bootstrap"]["kappa_ci"]
t1["kappa"] = [f"{k:.3f} [{kc[str(t)][0]:.2f}, {kc[str(t)][1]:.2f}]" for k, t in zip(ta.kappa, (1, 2, 3))]
t1 = t1.rename(columns={"threshold": "BanglaClick cut", "disagreement": "Disagree", "kappa": r"$\kappa$ [95\% CI]",
                        "bc_prevalence": "Prev.\\ B", "bb_prevalence": "Prev.\\ A"})
for c in ("Disagree", "Prev.\\ B", "Prev.\\ A"):
    t1[c] = t1[c].map(f3)
tex_table(t1, "tab_anchor_agreement", r"Agreement between BanglaBait (regime A) and BanglaClick (regime B) on the 1,083 shared news title classes, by the cut used to binarise the BanglaClick rubric level. Intervals: publisher-cluster bootstrap (2,000 draws).",
          "tab:anchor", "lrrrr")

# ---------------------------------------------------------------- Table: rubric components
rl = pd.read_csv(R / "E1/rubric_logit_HC1.csv")
rl = rl[rl.term != "const"].copy()
names = {"fail_factual": "Fails Factual", "fail_attribution": "Fails Attribution", "fail_specificity": "Fails Specificity",
         "fail_tone": "Fails Tone", "trigger_level": "Trigger level (0--3)"}
rl["term"] = rl.term.map(names)
rl["OR [95\\% CI]"] = [f"{np.exp(b):.2f} [{np.exp(l):.2f}, {np.exp(h):.2f}]" for b, l, h in zip(rl.log_odds, rl.ci_low, rl.ci_high)]
rd = pd.read_csv(R / "E1/residual_disagreement_by_rubric.csv")
prev = {r.dimension: r.n for r in rd[(rd.failed == 1)].itertuples()}
rl["Anchors with component"] = [prev.get({"Fails Factual": "factual", "Fails Attribution": "attribution", "Fails Specificity": "specificity", "Fails Tone": "tone"}.get(t, ""), "--") for t in rl.term]
rl["$p$"] = rl.p.map(lambda v: "$<$0.001" if v < 0.001 else f"{v:.3f}")
tex_table(rl[["term", "Anchors with component", "OR [95\\% CI]", "$p$"]].rename(columns={"term": "BanglaClick rubric component"}),
          "tab_rubric", r"Which BanglaClick rubric components predict the BanglaBait label on anchors? Logistic regression, HC1 errors ($n=1{,}083$ title classes).", "tab:rubric", "lrrr", wide=True)

# ---------------------------------------------------------------- Fig 3 + Table: AUROC under two constructs (frozen R6 models)
e5 = pd.read_csv(R / "E5/calibration_adaptation_summary.csv")
ts = e5[e5.method == "temperature_scaled_source"]
fig, axs = plt.subplots(1, 4, figsize=(W2, 1.9), sharey=True)
for a, m in zip(axs, MODELS):
    s = ts[ts.model == m]
    src = s[s.env == "banglabait:news"].auroc.iloc[0]
    for i, c in enumerate(ENVS):
        v6 = s[(s.env == c) & (s.label == "r6")].auroc.iloc[0]; va = s[(s.env == c) & (s.label == "aligned")].auroc.iloc[0]
        a.plot([i, i], [v6, va], color=C_GREY, lw=0.8, zorder=1)
        a.scatter(i, v6, color=C_R6, s=16, zorder=2, label="published label" if i == 0 else None)
        a.scatter(i, va, color=C_AL, s=16, zorder=2, marker="s", label="aligned label" if i == 0 else None)
    bbv = s[s.env == "baitbuster:youtube"].auroc.iloc[0]
    a.scatter(4, bbv, color=C_SRC, s=16, marker="^", label="regime C" if m == "ebm" else None)
    a.axhline(src, color=C_SRC, lw=0.7, ls="--")
    a.set_xticks(range(5)); a.set_xticklabels(["News", "FB", "YT", "TW", "YT-C"], rotation=0)
    a.set_title(MNAME[m]); a.set_ylim(0.4, 1.0); a.axhline(0.5, color=C_GREY, lw=0.5, ls=":")
axs[0].set_ylabel("AUROC (frozen source models)"); axs[0].legend(frameon=False, loc="lower left", fontsize=6)
fig.tight_layout(); save(fig, "fig3_auroc_constructs")

rows = []
for c in ["banglabait:news"] + ENVS + ["baitbuster:youtube"]:
    r = {"Cell": ENAME[c]}
    for m in MODELS:
        s = ts[(ts.model == m) & (ts.env == c)]
        v6 = s[s.label == "r6"].auroc.iloc[0]
        r[MNAME[m]] = f"{v6:.3f}" + (f" / {s[s.label == 'aligned'].auroc.iloc[0]:.3f}" if (s.label == "aligned").any() else "")
    rows.append(r)
tex_table(pd.DataFrame(rows), "tab_auroc", r"AUROC of the frozen source models (mean of five seeds). Regime-B cells: published label / aligned label.", "tab:auroc", "lrrrr", wide=True)

# ---------------------------------------------------------------- Fig 4 + Table: decomposition
dec = pd.read_csv(R / "E4/decomposition_summary.csv"); dec = dec[dec.aligned_label == "y_al"]
ci = pd.read_csv(R / "E4/decomposition_cluster_bootstrap_ci.csv")
fig, axs = plt.subplots(1, 4, figsize=(W2, 2.0), sharey=True)
order = ENVS + (["banglaclick:news_nonanchor"] if (dec.env == "banglaclick:news_nonanchor").any() else [])
for a, m in zip(axs, MODELS):
    s = dec[dec.model == m].set_index("env").loc[order]
    x = np.arange(len(order))
    pos_base = np.zeros(len(x)); neg_base = np.zeros(len(x))
    for comp, col, lab in (("nll_construct", C_R6, "construct"), ("nll_prior", "#D9A441", "prior"), ("nll_residual", C_AL, "residual")):
        v = s[comp].to_numpy()
        base = np.where(v >= 0, pos_base, neg_base)
        a.bar(x, v, 0.65, bottom=base, color=col, label=lab)
        pos_base += np.where(v >= 0, v, 0); neg_base += np.where(v < 0, v, 0)
    a.scatter(x, s.nll_gap, color="k", s=10, zorder=3, label="total gap")
    a.axhline(0, color="k", lw=0.5)
    a.set_xticks(x); a.set_xticklabels(["News", "FB", "YT", "TW", "News*"][:len(x)])
    a.set_title(MNAME[m])
axs[0].set_ylabel(r"$\Delta$ NLL (target $-$ source)")
h, l = axs[0].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=7)
save(fig, "fig4_decomposition")

rows = []
for c in order:
    for m in ("banglabert", "xlmr", "tfidf_lr", "ebm"):
        s = dec[(dec.env == c) & (dec.model == m)].iloc[0]
        k = ci[(ci.env == c) & (ci.model == m)]
        fmt = lambda comp: f"{s[comp]:+.3f}" + (f" [{k[comp + '_lo'].iloc[0]:+.2f}, {k[comp + '_hi'].iloc[0]:+.2f}]" if len(k) else "")
        rows.append({"Cell": ENAME[c], "Model": MNAME[m], "Gap": f"{s.nll_gap:+.3f}", "Construct $\\phi_A$": fmt("nll_construct"),
                     "Prior $\\phi_P$": fmt("nll_prior"), "Residual $\\rho$": fmt("nll_residual")})
tex_table(pd.DataFrame(rows), "tab_decomposition", r"Exact two-player Shapley decomposition of the target$-$source NLL gap of frozen source models (mean of five seeds). Brackets: 95\% publisher/channel-cluster bootstrap intervals (500 draws).",
          "tab:decomp", "llrrrr", wide=True)

# ---------------------------------------------------------------- Fig 5: cue transport
cr = pd.read_csv(R / "E2/crossed_platform_regime_contrasts.csv")
cue_short = {"curiosity_information_withholding": "Curiosity", "direct_address_imperative": "Direct address",
             "exaggeration_superlative": "Exaggeration", "interrogative_structure": "Interrogative",
             "punctuation_intensity": "Punctuation", "sensational_emotional_loading": "Sensational",
             "specificity_numerals": "Numerals", "vague_deictic_reference": "Deictic"}
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.2))
a = axs[0]
p6 = cr[cr.regime_B_label == "y_r6"].set_index("cue"); pa = cr[cr.regime_B_label == "y_al"].set_index("cue")
y = np.arange(len(p6))
a.barh(y + 0.2, p6.platform_sd_in_B, 0.4, color=C_R6, label="published label")
a.barh(y - 0.2, pa.loc[p6.index].platform_sd_in_B, 0.4, color=C_AL, label="aligned label")
a.set_yticks(y); a.set_yticklabels([cue_short[c] for c in p6.index]); a.set_xlabel("SD of cue log-OR across 4 platforms (regime B)")
a.legend(frameon=False); a.set_title("(a) Platform heterogeneity within one regime", loc="left")
a = axs[1]
a.scatter(pa.loc[p6.index].regime_B_minus_A_on_news, y, color=C_AL, s=14, label="B $-$ A on News")
a.scatter(pa.loc[p6.index].regime_C_minus_B_on_youtube, y, color=C_SRC, marker="^", s=14, label="C $-$ B on YouTube")
a.axvline(0, color="k", lw=0.5); a.set_yticks(y); a.set_yticklabels([])
a.set_xlabel(r"difference in cue log-OR (aligned label)"); a.legend(frameon=False, fontsize=6)
a.set_title("(b) Regime contrasts at a fixed platform", loc="left")
fig.tight_layout(); save(fig, "fig5_cue_transport")

# ---------------------------------------------------------------- Fig 6: audit
au = pd.read_csv(R / "E11/audit_results.csv")
fig, axs = plt.subplots(1, 3, figsize=(W2, 1.9))
meth = {"classical": ("Classical SRS", C_SRC, "o"), "stratified": ("Stratified", C_GREY, "s"), "ppi": ("PPI", C_AL, "D"), "active_ppi": ("Active PPI", C_R6, "^")}
for k, (lab, col, mk) in meth.items():
    for a, metric in zip(axs[:2], ("mae", "coverage")):
        s = au[(au.method == k) & (au.label == "y_al")].groupby("budget")[metric].mean()
        a.plot(s.index, s.values, marker=mk, color=col, ms=3, lw=0.9, label=lab)
for lab_, col, ls in (("y_r6", C_R6, "-"), ("y_al", C_AL, "-")):
    s = au[(au.method == "classical") & (au.label == lab_)].groupby("budget").decision_accuracy.mean()
    axs[2].plot(s.index, s.values, marker="o", ms=3, color=col, lw=0.9, label="published label" if lab_ == "y_r6" else "aligned label")
axs[0].set_ylabel("MAE of target Brier"); axs[1].set_ylabel("95% CI coverage"); axs[1].axhline(0.95, color=C_GREY, ls=":", lw=0.6)
axs[2].set_ylabel("deploy/relabel accuracy")
for a in axs:
    a.set_xscale("log"); a.set_xticks([25, 50, 100, 200]); a.set_xticklabels(["25", "50", "100", "200"]); a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); a.set_xlabel("audit labels $n$")
axs[0].legend(frameon=False, fontsize=6); axs[2].legend(frameon=False, fontsize=6)
fig.tight_layout(); save(fig, "fig6_audit")

# LOEO tables: see the benchmark block at the end of this script.

# ---------------------------------------------------------------- few-cluster simulation table
if (R / "E13/few_cluster_size.csv").exists():
    fc = pd.read_csv(R / "E13/few_cluster_size.csv")
    t = fc.pivot(index="G", columns="method", values="size_at_5pct").reset_index()
    t.columns = ["Clusters $G$"] + [{"cr1": "CR1 $t_{G-1}$", "pairs": "Pairs bootstrap", "wcr": "Wild cluster (WCR)"}[c] for c in t.columns[1:]]
    for c in t.columns[1:]:
        t[c] = t[c].map(lambda v: f"{v:.3f}")
    tex_table(t, "tab_fewcluster", r"Empirical size of nominal 5\% tests of a zero risk-difference cue effect in simulated clustered data (1,000 replications per $G$; Monte Carlo SE $\approx 0.007$).", "tab:fewcluster", "rrrr")

# ---------------------------------------------------------------- cue extractor table (supplement)
ch = pd.read_csv(R / "E7/cue_presence_change_by_cell.csv")
t = ch.pivot_table(index="cue", columns="env", values="presence_changed")
t = t[[c for c in ["banglabait:news"] + ENVS + ["baitbuster:youtube"] if c in t.columns]]
t.columns = [ENAME[c] for c in t.columns]; t.index = [cue_short[c] for c in t.index]
t = t.map(lambda v: f"{100 * v:.1f}").reset_index().rename(columns={"index": "Cue family"})
tex_table(t, "tab_cue_extractor", r"Share of items (\%) whose cue presence changes when punctuation-attached tokens are split (extractor v1.1 vs.\ the frozen v1.0).", "tab:cues", "l" + "r" * (t.shape[1] - 1))
print("assets written:", sorted(p.name for p in FIG.glob("*.pdf")), sorted(p.name for p in TAB.glob("*.tex")))


# ---------------------------------------------------------------- numbers.tex: every prose number as a macro
def nm(name, val):
    return "\\newcommand{\\" + name + "}{" + val + "}\n"


N = ""
anc = json.loads((R / "E1/anchor_summary.json").read_text())
N += nm("NAnchors", f"{anc['anchor_classes']:,}".replace(",", "{,}"))
for t, word in ((1, "One"), (2, "Two"), (3, "Three")):
    r = ta[ta.threshold == t].iloc[0]
    N += nm(f"Disagree{word}", f"{100 * r.disagreement:.1f}") + nm(f"Kappa{word}", f"{r.kappa:.2f}") + nm(f"PrevB{word}", f"{r.bc_prevalence:.3f}")
N += nm("PrevAAnchor", f"{ta.bb_prevalence.iloc[0]:.3f}")
kd = bs["publisher_cluster_bootstrap"]["kappa_diff_2_minus_1_ci"]
N += nm("KappaGainLo", f"{kd[0]:.2f}") + nm("KappaGainHi", f"{kd[1]:.2f}")
N += nm("ArgmaxTwo", f"{100 * bs['item_bootstrap']['argmax_threshold_freq']['2']:.0f}")
for l in range(4):
    N += nm("G" + "abcd"[l], f"{lk.g_isotonic.iloc[l]:.2f}")
rlt = pd.read_csv(R / "E1/rubric_logit_HC1.csv").set_index("term")
N += nm("ORattr", f"{np.exp(rlt.loc['fail_attribution', 'log_odds']):.2f}") + nm("ORtrig", f"{np.exp(rlt.loc['trigger_level', 'log_odds']):.2f}")
N += nm("Pattr", f"{rlt.loc['fail_attribution', 'p']:.2f}")
N += nm("NattrFail", str(int(rd[(rd.dimension == 'attribution') & (rd.failed == 1)].n.iloc[0])))
for c, tag in zip(ENVS, ("News", "Fb", "Yt", "Tw")):
    s = e3[(e3.env == c)].set_index("estimator")
    N += nm("PrevRsix" + tag, f"{s.loc['pcc', 'true_r6']:.3f}") + nm("PrevAl" + tag, f"{s.loc['pcc', 'true_aligned_hard']:.3f}")
    N += nm("Pcc" + tag, f"{s.loc['pcc', 'estimate']:.3f}")
pe = e3[(e3.estimator == "pcc") & e3.env.isin(ENVS)]
N += nm("PccErrAlMax", f"{pe.abs_err_aligned.max():.3f}") + nm("PccErrRsixMin", f"{pe.abs_err_r6.min():.3f}")
for m in MODELS:
    s = ts[ts.model == m]; M_ = {"ebm": "Ebm", "tfidf_lr": "Tfidf", "banglabert": "Bbert", "xlmr": "Xlmr"}[m]
    N += nm("AucSrc" + M_, f"{s[s.env == 'banglabait:news'].auroc.iloc[0]:.3f}")
    N += nm("AucC" + M_, f"{s[s.env == 'baitbuster:youtube'].auroc.iloc[0]:.3f}")
    for c, tag in zip(ENVS, ("News", "Fb", "Yt", "Tw")):
        N += nm("AucRsix" + tag + M_, f"{s[(s.env == c) & (s.label == 'r6')].auroc.iloc[0]:.3f}")
        N += nm("AucAl" + tag + M_, f"{s[(s.env == c) & (s.label == 'aligned')].auroc.iloc[0]:.3f}")
    for c, tag in zip(order, ("News", "Fb", "Yt", "Tw", "NewsNa")):
        d_ = dec[(dec.env == c) & (dec.model == m)].iloc[0]
        for comp, ct in (("nll_gap", "Gap"), ("nll_construct", "Con"), ("nll_prior", "Pri"), ("nll_residual", "Res")):
            N += nm("Dec" + ct + tag + M_, f"{d_[comp]:.3f}")
        N += nm("DecShare" + tag + M_, f"{100 * (d_.nll_construct + d_.nll_prior) / d_.nll_gap:.0f}")
        k_ = ci[(ci.env == c) & (ci.model == m)]
        if len(k_):  # cluster-bootstrap interval of the residual
            N += nm("DecResLo" + tag + M_, f"{k_.nll_residual_lo.iloc[0]:.3f}") + nm("DecResHi" + tag + M_, f"{k_.nll_residual_hi.iloc[0]:.3f}")
            N += nm("DecResShare" + tag + M_, f"{100 * d_.nll_residual / d_.nll_gap:.0f}")
for _, r in cr[cr.regime_B_label == "y_al"].iterrows():
    pass
sd6 = cr[cr.regime_B_label == "y_r6"].set_index("cue").platform_sd_in_B; sda = cr[cr.regime_B_label == "y_al"].set_index("cue").platform_sd_in_B
N += nm("CuesSdDown", str(int((sda < sd6).sum()))) + nm("MeanSdRsix", f"{sd6.mean():.2f}") + nm("MeanSdAl", f"{sda.mean():.2f}")
pa_ = cr[cr.regime_B_label == "y_al"]
N += nm("MeanAbsBA", f"{pa_.regime_B_minus_A_on_news.abs().mean():.2f}") + nm("MeanAbsCB", f"{pa_.regime_C_minus_B_on_youtube.abs().mean():.2f}")
N += nm("MeanAbsPlat", f"{pa_[[c for c in pa_.columns if c.startswith('platform_') and c.endswith('_in_B') and 'sd' not in c]].abs().mean().mean():.2f}")
au_c = au[au.method == "classical"].groupby(["label", "budget"]).agg(acc=("decision_accuracy", "mean"), cov=("coverage", "mean"), mae=("mae", "mean"))
for lab, tag in (("y_r6", "Rsix"), ("y_al", "Al")):
    for n in (25, 200):
        N += nm(f"AudAcc{tag}{'Small' if n == 25 else 'Large'}", f"{100 * au_c.loc[(lab, n), 'acc']:.0f}")
        N += nm(f"AudCov{tag}{'Small' if n == 25 else 'Large'}", f"{au_c.loc[(lab, n), 'cov']:.2f}")
mm = au.groupby(["method", "budget"]).mae.mean().unstack()
N += nm("AudMaeClassicalSmall", f"{mm.loc['classical', 25]:.3f}") + nm("AudMaePpiSmall", f"{mm.loc['ppi', 25]:.3f}") + nm("AudMaeStratSmall", f"{mm.loc['stratified', 25]:.3f}") + nm("AudMaeActiveSmall", f"{mm.loc['active_ppi', 25]:.3f}")
od = au.drop_duplicates(["env", "model", "label"])
N += nm("DeployRsix", str(int(((od.label == "y_r6") & (od.oracle_decision == "DEPLOY")).sum()))) + nm("TotRsix", str(int((od.label == "y_r6").sum())))
N += nm("DeployAl", str(int(((od.label == "y_al") & (od.oracle_decision == "DEPLOY")).sum()))) + nm("TotAl", str(int((od.label == "y_al").sum())))
mx = ch.presence_changed.max(); N += nm("CueChangeMax", f"{100 * mx:.1f}")
if (R / "E13/few_cluster_size.csv").exists():
    fc = pd.read_csv(R / "E13/few_cluster_size.csv").set_index(["G", "method"]).size_at_5pct
    for G in (10, 13, 26, 40):
        for meth, t in (("pairs", "Pairs"), ("cr1", "Crone"), ("wcr", "Wcr")):
            if (G, meth) in fc.index:
                N += nm(f"Size{t}{ {10: 'Ten', 13: 'Thirteen', 26: 'TwentySix', 40: 'Forty'}[G]}", f"{fc.loc[(G, meth)]:.3f}")
(PAPER / "numbers.tex").write_text("% generated by scripts/scope_make_paper_assets.py -- do not edit\n" + N, encoding="utf-8")
print("numbers.tex written:", N.count("newcommand"), "macros")


# ---------------------------------------------------------------- supplementary tables S2, S3
cal = pd.read_csv(R / "E5/calibration_adaptation_summary.csv")
mname = {"uncalibrated": "Uncalibrated", "temperature_scaled_source": "Source TS", "prior_bbse": "TS + BBSE prior",
         "prior_mlls": "TS + MLLS prior", "prior_oracle": "TS + oracle prior"}
rows = []
for c in ENVS + ["baitbuster:youtube"]:
    for lab in ("r6", "aligned"):
        for meth in mname:
            s = cal[(cal.env == c) & (cal.label == lab) & (cal.method == meth) & (cal.model.isin(["banglabert", "xlmr"]))]
            if not len(s):
                continue
            rows.append({"Cell": ENAME[c], "Label": lab, "Method": mname[meth], "NLL": f"{s.nll.mean():.3f}", "Brier": f"{s.brier.mean():.3f}",
                         "ECE": f"{s.ece.mean():.3f}", "Macro-F1": f"{s.macro_f1.mean():.3f}", "Recall": f"{s.recall.mean():.3f}"})
tex_table(pd.DataFrame(rows), "supp_tab_calibration", r"Calibration and prior-adaptation methods on each target cell (mean of BanglaBERT and XLM-R over five seeds), under the published (r6) and aligned labels.", "tab:s-cal", "lllrrrrr")
dsoft = pd.read_csv(R / "E4/decomposition_summary.csv"); dsoft = dsoft[dsoft.aligned_label == "q_al"]
rows = []
for c in order:
    for m in MODELS:
        s = dsoft[(dsoft.env == c) & (dsoft.model == m)].iloc[0]
        rows.append({"Cell": ENAME[c], "Model": MNAME[m], "Gap": f"{s.nll_gap:+.3f}", "Construct": f"{s.nll_construct:+.3f}", "Prior": f"{s.nll_prior:+.3f}", "Residual": f"{s.nll_residual:+.3f}"})
tex_table(pd.DataFrame(rows), "supp_tab_soft_decomp", r"Decomposition with the soft linked label $q=\hat g(L)$ (sensitivity analysis; mean of five seeds).", "tab:s-soft", "llrrrr")
e2 = pd.read_csv(R / "E2/cue_association_by_cell.csv")
rows = []
for cue in e2.cue.unique():
    r = {"Cue": cue_short[cue]}
    for env in ["banglabait:news"] + ENVS + ["baitbuster:youtube"]:
        lab = "y_al" if env.startswith("banglaclick") else "y_r6"
        s = e2[(e2.env == env) & (e2.cue == cue) & (e2.label == lab)].iloc[0]
        r[ENAME[env].replace(" (", "\newline(") if False else ENAME[env]] = f"{s.log_or:.2f} [{s.log_or_lo:.1f}, {s.log_or_hi:.1f}]"
    rows.append(r)
tex_table(pd.DataFrame(rows), "supp_tab_cue_logor", r"Cue--label log odds ratios by cell (aligned label for regime B) with 95\% cluster-bootstrap intervals (1,000 draws).", "tab:s-cue", "l" + "r" * 6)
print("supplementary tables written")

# ---------------------------------------------------------------- E14 macros (appended to numbers.tex)
h = pd.read_csv(R / "E14/cluster_label_homogeneity.csv").set_index(["env", "label"])
extra = ""
extra += nm("EtaBaitBuster", f"{h.loc[('baitbuster:youtube', 'y_r6'), 'eta2_between_cluster']:.3f}")
extra += nm("EtaYtB", f"{h.loc[('banglaclick:youtube', 'y_r6'), 'eta2_between_cluster']:.2f}")
extra += nm("EtaFbB", f"{h.loc[('banglaclick:facebook', 'y_r6'), 'eta2_between_cluster']:.2f}")
extra += nm("TopChannelYtB", f"{100 * h.loc[('banglaclick:youtube', 'y_r6'), 'largest_cluster_share']:.0f}")
with open(PAPER / "numbers.tex", "a", encoding="utf-8") as fh:
    fh.write(extra)
print("E14 macros appended")

# ================================================================ benchmark, synthetic and anchor-proxy assets
PNAME = {"majority": "majority", "source_only": "source only", "pooled_erm": "pooled ERM", "group_dro": "Group DRO",
         "group_balanced": "env.-balanced ERM", "iw_source": "IW (source)", "iw_pooled": "IW (pooled)",
         "source_mlls": "source + MLLS prior", "source_oracle": "source + oracle prior", "in_domain": "in domain (ref.)",
         "latent_cuts": "regime cuts"}
TGTS = ENVS + ["baitbuster:youtube"]
TSHORT = {"banglaclick:news": "News", "banglaclick:facebook": "FB", "banglaclick:youtube": "YT-B",
          "banglaclick:twitter": "TW", "baitbuster:youtube": "YT-C"}
TMAC = {"banglaclick:news": "News", "banglaclick:facebook": "Fb", "banglaclick:youtube": "Yt",
        "banglaclick:twitter": "Tw", "baitbuster:youtube": "C"}
PMAC = {"source_only": "Src", "pooled_erm": "Pool", "group_dro": "Dro", "iw_source": "Iws", "iw_pooled": "Iwp", "latent_cuts": "Lat"}
LAB_TEX = {"r6": "$Y$", "aligned": r"$Y^\star$"}
X = ""
parts = [pd.read_csv(f) for f in [R / "E10/loeo_tfidf_summary.csv", R / "E16/shift_baselines_summary.csv",
                                  R / "E10/loeo_transformer_summary.csv", R / "E22/latent_cuts_summary.csv"] if f.exists()]
if parts:
    bm = pd.concat(parts, ignore_index=True)

    def bench_table(metric, protos, name, caption, label):
        rows = []
        for model in ["tfidf_lr", "banglabert", "xlmr"]:
            for pr in protos:
                g = bm[(bm.model == model) & (bm.protocol == pr)]
                if g.empty:
                    continue
                r = {"Model": MNAME[model], "Protocol": PNAME[pr]}
                for lab in ("r6", "aligned"):
                    for tg in TGTS:
                        v = g[(g.label == lab) & (g.target == tg)][metric]
                        r[TSHORT[tg] + " " + LAB_TEX[lab]] = f"{v.iloc[0]:.3f}" if len(v) else "--"
                rows.append(r)
        t = pd.DataFrame(rows)
        tex_table(t, name, caption, label, "ll" + "r" * (t.shape[1] - 2), wide=True)

    bench_table("auroc", ["source_only", "pooled_erm", "group_dro", "iw_source", "iw_pooled", "latent_cuts", "in_domain"], "tab_loeo",
                r"AUROC of models retrained leave-one-environment-out under the published label $Y$ (left block) and the aligned label $Y^\star$ (right block), mean over three seeds. Training excludes the held-out cell and every title that occurs in it. Prior corrections leave AUROC unchanged and are omitted. \emph{Regime cuts}: the latent-intensity model of Section~\ref{sec:latent}. \emph{In domain} uses target labels (cluster-grouped cross-fitting) and is a reference, not a baseline. YT-B: BanglaClick YouTube; YT-C: BaitBuster YouTube.", "tab:loeo")
    bench_table("nll", ["majority", "source_only", "source_mlls", "source_oracle", "pooled_erm", "group_dro", "iw_source", "iw_pooled", "latent_cuts"], "supp_tab_loeo_nll",
                r"Log loss of the LOEO models (mean over three seeds). \emph{Source + MLLS prior}: source-only model with its prior moved to the MLLS estimate from unlabelled target scores. \emph{Oracle prior} uses the true target prevalence and is a reference.", "tab:s-loeo-nll")
    base5 = ["source_only", "pooled_erm", "group_dro", "iw_source", "iw_pooled"]
    tf = bm[bm.model == "tfidf_lr"].pivot_table(index=["label", "target"], columns="protocol", values="auroc")
    gain = (tf.loc["aligned"] - tf.loc["r6"]).loc[["banglaclick:news", "banglaclick:facebook", "banglaclick:twitter"], base5]
    spread = tf[base5].max(axis=1) - tf[base5].min(axis=1)
    X += nm("BenchLabelGainMin", f"{gain.values.min():.2f}") + nm("BenchLabelGainMax", f"{gain.values.max():.2f}")
    X += nm("BenchMethodSpreadMedian", f"{spread.median():.3f}")
    nl = bm[bm.model == "tfidf_lr"].pivot_table(index=["label", "target"], columns="protocol", values="nll")
    X += nm("MllsWorseCells", str(int((nl.source_mlls > nl.source_only).sum()))) + nm("MllsCells", str(int(nl.source_mlls.notna().sum())))
    X += nm("MllsNllMax", f"{(nl.source_mlls - nl.source_only).max():.1f}")  # largest increase over the uncorrected model
    X += nm("OracleBetterCellsRsix", str(int((nl.loc["r6"].source_oracle < nl.loc["r6"].source_only).sum())))
    if "iw_relative_ess" in bm:
        e = bm[bm.protocol.isin(["iw_source", "iw_pooled"])].iw_relative_ess
        X += nm("IwEssMin", f"{100 * e.min():.0f}") + nm("IwEssMax", f"{100 * e.max():.0f}")
    for tg in TGTS:
        for lab in ("r6", "aligned"):
            for pr in base5 + ["latent_cuts"]:
                v = bm[(bm.model == "tfidf_lr") & (bm.protocol == pr) & (bm.label == lab) & (bm.target == tg)].auroc
                if len(v):
                    X += nm("Lo" + PMAC[pr] + ("Rsix" if lab == "r6" else "Al") + TMAC[tg], f"{v.iloc[0]:.3f}")
    for model, mt in [("banglabert", "Bbert"), ("xlmr", "Xlmr")]:
        for tg in TGTS:
            for lab in ("r6", "aligned"):
                for pr in ("pooled_erm", "group_dro", "source_only", "latent_cuts"):
                    v = bm[(bm.model == model) & (bm.protocol == pr) & (bm.label == lab) & (bm.target == tg)].auroc
                    if len(v):
                        X += nm("Lo" + PMAC[pr] + ("Rsix" if lab == "r6" else "Al") + TMAC[tg] + mt, f"{v.iloc[0]:.3f}")

# ---- E17 forest plot of paired contrasts
if (R / "E17/protocol_contrasts_cluster_bootstrap.csv").exists():
    ct = pd.read_csv(R / "E17/protocol_contrasts_cluster_bootstrap.csv")
    a = ct[(ct.metric == "auroc") & (ct.reference == "source_only")]
    protos = [p for p in ["pooled_erm", "group_dro", "iw_source", "iw_pooled", "group_balanced"] if p in set(a.protocol)]
    combos = [(m, l) for m in ["tfidf_lr", "banglabert", "xlmr"] for l in ("r6", "aligned") if not a[(a.model == m) & (a.label == l)].empty]
    fig, axs = plt.subplots(1, len(combos), figsize=(7.16, 2.5), sharey=True, squeeze=False)
    axs = axs[0]
    cols = {"pooled_erm": "#0072B2", "group_dro": "#D55E00", "iw_source": "#009E73", "iw_pooled": "#CC79A7", "group_balanced": "#E69F00"}
    mk = {"pooled_erm": "o", "group_dro": "s", "iw_source": "^", "iw_pooled": "v", "group_balanced": "D"}
    seen = set()
    for ax, (m, l) in zip(axs, combos):
        g = a[(a.model == m) & (a.label == l)]
        for k, pr in enumerate(protos):
            for j, tg in enumerate(TGTS):
                v = g[(g.protocol == pr) & (g.target == tg)]
                if v.empty:
                    continue
                v = v.iloc[0]; y = j + (k - (len(protos) - 1) / 2) * 0.15
                ax.errorbar(v["diff"], y, xerr=[[v["diff"] - v.ci_low], [v.ci_high - v["diff"]]], fmt=mk[pr], color=cols[pr], ms=3, lw=0.8,
                            mfc=cols[pr] if v.p_holm < 0.05 else "white", label=None if pr in seen else PNAME[pr])
                seen.add(pr)
        ax.axvline(0, color="0.5", lw=0.6, ls="--")
        ref = g.reference.iloc[0]
        ax.set_title(MNAME[m] + ", " + LAB_TEX[l] + ("" if ref == "source_only" else "\nvs pooled ERM"), fontsize=7)
        ax.set_xlabel(r"$\Delta$AUROC", fontsize=7); ax.tick_params(labelsize=6)
    axs[0].set_yticks(range(len(TGTS))); axs[0].set_yticklabels([TSHORT[t] for t in TGTS]); axs[0].invert_yaxis()
    hs, ls_ = [], []
    for ax in axs:
        h_, l_ = ax.get_legend_handles_labels(); hs += h_; ls_ += l_
    fig.legend(hs, ls_, loc="upper center", ncol=len(hs), frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.9)); save(fig, "fig7_benchmark")
    A = (ct.metric == "auroc") & ct.p_holm.notna() & (ct.reference == "source_only") & ~ct.protocol.isin(["source_mlls", "source_oracle", "in_domain"])  # training protocols vs source-only (prior corrections leave AUROC unchanged); Holm family = all paired contrasts
    X += nm("ContrastsTotal", str(int(A.sum())))
    X += nm("ContrastsSigPos", str(int((A & (ct.p_holm < 0.05) & (ct["diff"] > 0)).sum())))
    X += nm("ContrastsSigNeg", str(int((A & (ct.p_holm < 0.05) & (ct["diff"] < 0)).sum())))

# ---- E15 synthetic table
if (R / "E15/synthetic_recovery_summary.csv").exists():
    sy = pd.read_csv(R / "E15/synthetic_recovery_summary.csv")
    base = sy[(sy.delta == 0) & (sy.n_anchor == 1083)]
    rows = []
    for om in sorted(base.omega.unique()):
        for lam in sorted(base["lambda"].unique()):
            r = {r"$\omega$": f"{om:.1f}", r"$\lambda$": f"{lam:.1f}"}
            for met, mname in [("auroc", "AUROC"), ("nll", "NLL")]:
                for pr, pn in [("r6", "$Y$"), ("hard", r"$Y^\star$"), ("soft", "$q$")]:
                    v = base[(base.omega == om) & (base["lambda"] == lam) & (base.metric == met) & (base.proxy == pr)].bias.iloc[0]
                    r[mname + " " + pn] = f"{v:+.3f}"
            rows.append(r)
    tex_table(pd.DataFrame(rows), "tab_synthetic", r"Synthetic check with known ground truth: bias of each target-side label as an estimate of the source-construct risk $R^\star$ (1,083 anchors, no injected shift, 200 replications). $\omega$: share of the model's signal that runs through the rubric level ($\omega=1$ satisfies Proposition~\ref{prop:soft}). $\lambda$: logit offset of the anchor link (Assumption~\ref{as:rep} holds at $\lambda=0$).", "tab:synthetic", "rr" + "r" * 6, wide=True)

    def sv(om, lam, met, pr, n=1083, d=0.0, col="bias"):
        return sy[(sy.omega == om) & (sy["lambda"] == lam) & (sy.metric == met) & (sy.proxy == pr) & (sy.n_anchor == n) & (sy.delta == d)][col].iloc[0]
    X += nm("SynSoftAucBiasOmegaOne", f"{sv(1.0, 0.0, 'auroc', 'soft'):+.3f}") + nm("SynHardAucBiasOmegaOne", f"{sv(1.0, 0.0, 'auroc', 'hard'):+.2f}")
    X += nm("SynHardAucBiasOmegaHalf", f"{sv(0.5, 0.0, 'auroc', 'hard'):+.3f}") + nm("SynSoftAucBiasOmegaHalf", f"{sv(0.5, 0.0, 'auroc', 'soft'):+.2f}")
    X += nm("SynHardAucBiasOmegaZero", f"{sv(0.0, 0.0, 'auroc', 'hard'):+.2f}") + nm("SynSoftAucBiasOmegaZero", f"{sv(0.0, 0.0, 'auroc', 'soft'):+.2f}")
    X += nm("SynSoftNllBiasLamOne", f"{sv(1.0, 1.0, 'nll', 'soft'):+.2f}") + nm("SynHardNllBiasLamOne", f"{sv(1.0, 1.0, 'nll', 'hard'):+.2f}")
    X += nm("SynSoftCovSmall", f"{sv(1.0, 0.0, 'nll', 'soft', n=100, col='coverage'):.2f}") + nm("SynSoftCovLarge", f"{sv(1.0, 0.0, 'nll', 'soft', n=1083, col='coverage'):.2f}")
    X += nm("SynTrueAucDeltaZero", f"{sv(0.5, 0.0, 'auroc', 'r6', col='true_mean'):.2f}") + nm("SynTrueAucDeltaHalf", f"{sv(0.5, 0.0, 'auroc', 'r6', d=0.5, col='true_mean'):.2f}")

# ---- E18 anchor proxy table
if (R / "E18/anchor_proxy_check_summary.csv").exists():
    ap = pd.read_csv(R / "E18/anchor_proxy_check_summary.csv"); tb = pd.read_csv(R / "E18/score_beyond_level_test.csv")
    keep = ["source_only", "pooled_erm", "group_dro"]
    rows = []
    for (m, pr, tl), g in ap.groupby(["model", "protocol", "train_label"]):
        if pr not in keep:
            continue
        g = g.set_index("proxy"); tt = tb[(tb.model == m) & (tb.protocol == pr) & (tb.train_label == tl)]
        rows.append({"Model": MNAME[m], "Protocol": PNAME[pr], "Trained on": LAB_TEX[tl],
                     "AUROC vs $Y_A$": f"{g.loc['truth_YA', 'auroc']:.3f}",
                     r"$\Delta$ $Y$": f"{g.loc['r6', 'auroc_bias_vs_truth']:+.3f}",
                     r"$\Delta$ $Y^\star$": f"{g.loc['hard_aligned', 'auroc_bias_vs_truth']:+.3f}",
                     r"$\Delta$ $q$": f"{g.loc['soft_crossfit', 'auroc_bias_vs_truth']:+.3f}",
                     r"NLL $\Delta$ $Y^\star$": f"{g.loc['hard_aligned', 'nll_bias_vs_truth']:+.3f}",
                     r"NLL $\Delta$ $q$": f"{g.loc['soft_crossfit', 'nll_bias_vs_truth']:+.3f}",
                     r"$\beta$": f"{tt.beta_logit_p_given_L.mean():.2f}"})
    t = pd.DataFrame(rows)
    tex_table(t, "tab_anchor_proxy", r"Proxy check on BanglaClick news anchors, where the source label $Y_A$ is observed. Models were retrained leave-one-environment-out and none saw these titles. $\Delta$: metric against the proxy minus metric against $Y_A$ (AUROC unless marked NLL; mean over seeds). $q$ is cross-fitted over two folds of anchor classes. $\beta$: coefficient of the model logit in a logistic regression of $Y_A$ on rubric-level indicators and the logit, clustered by anchor class; $\beta\neq0$ violates the condition of Proposition~\ref{prop:soft}.", "tab:anchorproxy", "lll" + "r" * (t.shape[1] - 3), wide=True)
    sub = ap[ap.protocol.isin(keep)]
    for pr, tag in [("r6", "Rsix"), ("hard_aligned", "Hard"), ("soft_crossfit", "Soft")]:
        b = sub[sub.proxy == pr].auroc_bias_vs_truth
        X += nm(f"ApBias{tag}Min", f"{b.min():+.2f}") + nm(f"ApBias{tag}Max", f"{b.max():+.2f}")
        bn = sub[sub.proxy == pr].nll_bias_vs_truth
        X += nm(f"ApNllBias{tag}Min", f"{bn.min():+.2f}") + nm(f"ApNllBias{tag}Max", f"{bn.max():+.2f}")
    tv = sub[sub.proxy == "truth_YA"].auroc
    X += nm("ApTruthMin", f"{tv.min():.2f}") + nm("ApTruthMax", f"{tv.max():.2f}")
    tb2 = tb[tb.protocol.isin(keep)]
    X += nm("ApBetaMin", f"{tb2.beta_logit_p_given_L.min():.2f}") + nm("ApBetaMax", f"{tb2.beta_logit_p_given_L.max():.2f}")
    X += nm("ApBetaPmax", f"{tb2.p_value.max():.3f}")

# ---- anchor representativeness with models that never saw the anchors (E10T)
if (R / "E10/anchor_representativeness_dedup_models.csv").exists():
    rp = pd.read_csv(R / "E10/anchor_representativeness_dedup_models.csv")
    rp = rp[rp.n > 0]
    w = rp.groupby(["level", "anchor"]).apply(lambda d: (d.mean_p * d.n).sum() / d.n.sum(), include_groups=False).unstack("anchor")
    dif = w[True] - w[False]
    X += nm("RepDedupMin", f"{dif.drop(2).min():.2f}") + nm("RepDedupMax", f"{dif.drop(2).max():.2f}") + nm("RepDedupLevelTwo", f"{dif.loc[2]:+.2f}")

with open(PAPER / "numbers.tex", "a", encoding="utf-8") as fh:
    fh.write(X)
print("benchmark/synthetic/proxy assets written;", X.count("newcommand"), "macros")


# ---------------------------------------------------------------- E19: score-conditional linking (tables, figure, macros)
if (R / "E19/application_summary.csv").exists():
    Y = ""
    v1 = pd.read_csv(R / "E19/v1_summary.csv"); v2 = pd.read_csv(R / "E19/v2_summary.csv"); v3 = pd.read_csv(R / "E19/v3_summary.csv")
    ap = pd.read_csv(R / "E19/application_summary.csv"); con = pd.read_csv(R / "E19/application_contrasts.csv")
    PROT = {"pooled_erm": "pooled ERM", "group_dro": "Group DRO"}
    LABN = {"r6": "$Y$", "aligned": "$Y^\\star$"}

    def scn(s):
        if s == "tfidf_lr":
            return "TF-IDF LR", "source only (blind)", "--"
        if "|" not in s:
            return MNAME.get(s, s), "source only (blind)", "--"
        m, p, l = s.split("|")
        return MNAME.get(m, m), PROT.get(p, p), LABN.get(l, l)
    rows = []
    for s, g in v1.groupby("scorer", sort=False):
        b = g.set_index("proxy")
        m, p, l = scn(s)
        rows.append({"Scorer": m, "Protocol": p, "Trained on": l,
                     **{k: f"{b.loc[c, 'auroc_bias_mean']:+.3f}" for k, c in (("$\\Delta$ $Y$", "r6"), ("$\\Delta$ $Y^\\star$", "hard_aligned"),
                                                                               ("$\\Delta$ $q$", "soft_isotonic"), ("$\\Delta$ SCL-lin", "scl_gamma1"), ("$\\Delta$ SCL", "scl_flex"))},
                     "NLL $\\Delta$ SCL": f"{b.loc['scl_flex', 'nll_bias_mean']:+.3f}"})
    t = pd.DataFrame(rows).sort_values(["Scorer", "Protocol", "Trained on"])
    tex_table(t, "tab_scl_validation", "Cross-fitted validation of the target-side labels on news anchors, where $Y_A$ is observed. Each label is fitted on four fifths of the anchor publisher clusters and applied to the held-out fifth (20 fold draws, mean over seeds). $\\Delta$: AUROC (or NLL) of the scorer against the label minus against $Y_A$. SCL: score-conditional link with a linear-spline score function ($\\gamma=1$); SCL-lin: linear in the logit. No scorer saw an anchor title.",
              "tab:sclval", colfmt="lllrrrrrr", wide=True)
    sv1 = v1[v1.proxy == "scl_flex"]; sl1 = v1[v1.proxy == "scl_gamma1"]
    Y += nm("SclLinVOneAbsMax", f"{sl1.auroc_bias_mean.abs().max():.3f}")
    nl = v1.pivot_table(index="scorer", columns="proxy", values="label_nll_vs_YA")
    Y += nm("SclFlexNllWins", str(int((nl.scl_flex <= nl.scl_gamma1).sum()))) + nm("SclVOneNScorersB", str(len(nl)))
    gp = pd.read_csv(R / "E19/selection_gap_calibration.csv")
    Y += nm("GapObsMin", f"{gp.observed_gap_sd.min():.2f}") + nm("GapObsMax", f"{gp.observed_gap_sd.max():.2f}")
    Y += nm("GapKHalf", f"{gp['induced_gap_kappa0.5'].mean():.2f}") + nm("GapKTwo", f"{gp['induced_gap_kappa2'].mean():.2f}")
    Y += nm("GapKOne", f"{gp['induced_gap_kappa1'].mean():.2f}")
    sf = v2[(v2.proxy == "scl_flex") & (v2.delta_label == 0) & (v2.kappa_score <= 1)]
    Y += nm("SclVTwoAucMaxRealistic", f"{sf.auroc_bias_mean.abs().max():.3f}")
    s2l = v2[(v2.proxy == "scl_gamma1") & (v2.delta_label == 0) & (v2.kappa_score == 0.5)]
    h2r = v2[(v2.proxy == "hard_aligned") & (v2.delta_label == 0) & (v2.kappa_score == 0.5)]
    Y += nm("SclLinVTwoRealMax", f"{s2l.auroc_bias_mean.abs().max():.3f}")
    Y += nm("HardVTwoRealMin", f"{h2r.auroc_bias_mean.min():.2f}") + nm("HardVTwoRealMax", f"{h2r.auroc_bias_mean.max():.2f}")
    s2lk = v2[(v2.proxy == "scl_gamma1") & (v2.delta_label == 0) & (v2.kappa_score == 2)]
    Y += nm("SclLinVTwoKTwoMax", f"{s2lk.auroc_bias_mean.abs().max():.3f}")
    Y += nm("SclVOneAbsMax", f"{sv1.auroc_bias_mean.abs().max():.3f}") + nm("SclVOneNllAbsMax", f"{sv1.nll_bias_mean.abs().max():.3f}")
    Y += nm("SclVOneHardMin", f"{v1[v1.proxy == 'hard_aligned'].auroc_bias_mean.min():.2f}") + nm("SclVOneHardMax", f"{v1[v1.proxy == 'hard_aligned'].auroc_bias_mean.max():.2f}")
    Y += nm("SclVOneSoftMin", f"{v1[v1.proxy == 'soft_isotonic'].auroc_bias_mean.min():.2f}") + nm("SclVOneSoftMax", f"{v1[v1.proxy == 'soft_isotonic'].auroc_bias_mean.max():.2f}")
    Y += nm("SclVOneNScorers", str(v1.scorer.nunique()))
    s2 = v2[v2.proxy == "scl_flex"]; h2 = v2[v2.proxy == "soft_isotonic"]
    Y += nm("SclVTwoAucMaxNull", f"{s2[s2.delta_label == 0].auroc_bias_mean.abs().max():.3f}")
    Y += nm("SclVTwoNllMaxNull", f"{s2[s2.delta_label == 0].nll_bias_mean.abs().max():.3f}")
    Y += nm("SclVTwoSoftNllKTwo", f"{h2[(h2.delta_label == 0) & (h2.kappa_score == 2)].nll_bias_mean.mean():.2f}")
    Y += nm("SclVTwoAucMaxAlt", f"{s2[s2.delta_label == 1].auroc_bias_mean.abs().max():.3f}")
    Y += nm("SclVTwoNllAltMin", f"{s2[s2.delta_label == 1].nll_bias_mean.min():.2f}") + nm("SclVTwoNllAltMax", f"{s2[s2.delta_label == 1].nll_bias_mean.max():.2f}")
    Y += nm("SclVTwoPrevAltMax", f"{s2[s2.delta_label == 1].prev_bias_mean.max():.2f}")
    h3 = v3[(v3.proxy == "hard_aligned")]
    s3all = v3[v3.proxy == "scl_flex"].groupby("n_anchors").auroc_rmse.max()
    for n, k in ((50, "Fifty"), (100, "Hundred"), (200, "TwoHundred"), (400, "FourHundred")):
        Y += nm(f"SclBudgetRmse{k}", f"{s3all.loc[n]:.3f}")
    Y += nm("SclBudgetHardMin", f"{h3.auroc_bias_mean.min():.2f}") + nm("SclBudgetHardMax", f"{h3.auroc_bias_mean.max():.2f}")
    # application table (blind TF-IDF scorer: the only scorer observed on every cell and on the source test set)
    a = ap[ap.scorer == "tfidf_lr"].set_index(["env", "label"])
    rows = []
    for e in ENVS:
        g0, g1 = a.loc[(e, "scl_g0")], a.loc[(e, "scl_g1")]
        ident = e == "banglaclick:news"
        rows.append({"Cell": ENAME[e], "$Y$": f3(a.loc[(e, 'r6'), 'auroc']), "$Y^\\star$": f3(a.loc[(e, 'hard_aligned'), 'auroc']),
                     "SCL, $\\gamma\\in[0,1]$": f"[{g0.auroc:.3f}, {g1.auroc:.3f}]",
                     "SCL, $\\gamma=1$ (95\\% CI)": f"{g1.auroc:.3f} ({g1.auroc_lo:.3f}, {g1.auroc_hi:.3f})" + ("" if ident else "$^\\dagger$"),
                     "$\\hat\\pi^\\star$": f"{g1.prevalence:.2f}", "$\\rho^\\star$ (NLL)": f"{g1.nll_residual_after_prior:+.3f}"})
    tex_table(pd.DataFrame(rows), "tab_scl_app", "AUROC of the anchor-blind TF-IDF source model on regime-B cells against three target-side labels: published ($Y$), hard aligned ($Y^\\star$) and the score-conditional link over the sensitivity range $\\gamma\\in[0,1]$. Source test AUROC " + f3(a.src_auroc.iloc[0]) + ". $\\hat\\pi^\\star$: implied source-construct prevalence and $\\rho^\\star$: prior-corrected NLL gap to the source, both at $\\gamma=1$. Intervals: 300-draw bootstrap over anchor publisher clusters and target clusters, seed-averaged (3 seeds); YouTube has 10 clusters and Facebook 26. $^\\dagger$Transported cell: $\\gamma=1$ assumes that the news within-level score function carries over; it is an endpoint of the sensitivity range, not an estimate.",
              "tab:sclapp", colfmt="lrrrrrr", wide=True)
    n1 = a.loc[("banglaclick:news", "scl_g1")]
    Y += nm("SclNewsAuc", f3(n1.auroc)) + nm("SclNewsAucLo", f3(n1.auroc_lo)) + nm("SclNewsAucHi", f3(n1.auroc_hi))
    Y += nm("SclNewsHard", f3(a.loc[("banglaclick:news", "hard_aligned"), "auroc"])) + nm("SclNewsPub", f3(a.loc[("banglaclick:news", "r6"), "auroc"]))
    Y += nm("SclSrcAuc", f3(n1.src_auroc))
    Y += nm("SclNewsResid", f"{n1.nll_residual_after_prior:+.3f}") + nm("SclNewsPrev", f"{n1.prevalence:.2f}")
    Y += nm("SclBeta", f"{n1.beta_hat:.2f}") + nm("SclNewsAucLin", f3(a.loc[("banglaclick:news", "scl_lin_g1"), "auroc"]))
    for e, k in (("banglaclick:facebook", "Fb"), ("banglaclick:twitter", "Tw"), ("banglaclick:youtube", "Yt")):
        Y += nm(f"Scl{k}Lo", f3(a.loc[(e, "scl_g0"), "auroc"])) + nm(f"Scl{k}Hi", f3(a.loc[(e, "scl_g1"), "auroc"]))
        Y += nm(f"Scl{k}Hard", f3(a.loc[(e, "hard_aligned"), "auroc"]))
    cn = con[(con.scorer == "tfidf_lr") & (con.env == "banglaclick:news") & (con.metric == "auroc") & (con.contrast == "scl_g1 - hard_aligned")].iloc[0]
    Y += nm("SclNewsMinusHard", f"{cn['mean']:.3f}") + nm("SclNewsMinusHardLo", f"{cn.lo:.3f}") + nm("SclNewsMinusHardHi", f"{cn.hi:.3f}")
    # figure: estimated AUROC over gamma by cell
    fig, ax = plt.subplots(figsize=(W1, 2.4))
    cols = {"banglaclick:news": C_SRC, "banglaclick:facebook": "#1F5F8B", "banglaclick:twitter": "#6A8F3A", "banglaclick:youtube": "#B4532A"}
    gs = [0, 0.25, 0.5, 0.75, 1]
    for e in ENVS:
        m = [a.loc[(e, f"scl_g{g:g}"), "auroc"] for g in gs]
        lo = [a.loc[(e, f"scl_g{g:g}"), "auroc_lo"] for g in gs]; hi = [a.loc[(e, f"scl_g{g:g}"), "auroc_hi"] for g in gs]
        ax.fill_between(gs, lo, hi, color=cols[e], alpha=0.12, lw=0)
        ax.plot(gs, m, "-o", color=cols[e], ms=2.8, lw=1.0, label=ENAME[e].replace(" (B)", ""))
        ax.plot([-0.08], [a.loc[(e, "hard_aligned"), "auroc"]], marker="s", color=cols[e], ms=3.2, ls="none", clip_on=False)
    ax.axhline(a.src_auroc.iloc[0], ls="--", color=C_GREY, lw=0.8)
    ax.text(0.02, a.src_auroc.iloc[0] + 0.006, "source test", fontsize=6.5, color=C_GREY)
    ax.text(-0.08, 0.525, r"$Y^\star$", fontsize=6.5, ha="center", color=C_GREY)
    ax.set_xlabel(r"$\gamma$: share of news score function transported")
    ax.set_ylabel(r"estimated AUROC against $Y_A$")
    ax.set_xlim(-0.14, 1.04); ax.set_ylim(0.5, 0.88); ax.set_xticks(gs)
    ax.legend(frameon=False, ncol=2, loc="lower right", handlelength=1.4)
    save(fig, "fig8_scl_sensitivity")
    with open(PAPER / "numbers.tex", "a", encoding="utf-8") as fh:
        fh.write(Y)
    print("E19 assets written;", Y.count("newcommand"), "macros")

# ---------------------------------------------------------------- independent re-derivation (scripts/scope_independent_check.py)
if (R / "INDEPENDENT_CHECK/independent_check.json").exists():
    ic = json.loads((R / "INDEPENDENT_CHECK/independent_check.json").read_text(encoding="utf-8"))["overlap"]["strict_nfc_whitespace"]
    Z = nm("IndConflictsOne", str(ic["cut1"]["conflicts"])) + nm("IndAnchors", f"{ic['shared_title_classes']:,}".replace(",", "{,}"))
    Z += nm("IndDisagreeOne", f"{100 * ic['cut1']['disagreement']:.1f}")
    with open(PAPER / "numbers.tex", "a", encoding="utf-8") as fh:
        fh.write(Z)


# ================================================================ E20 partial identification, E21 anchor design, E22 regime cut-points
FAM = {"tfidf_lr": "Tfidf", "banglabert": "Bbert", "xlmr": "Xlmr"}
CK = {"banglaclick:news": "News", "banglaclick:facebook": "Fb", "banglaclick:youtube": "Yt", "banglaclick:twitter": "Tw"}
W = ""
if (R / "E20/identified_sets_summary.csv").exists():
    ids_all = pd.read_csv(R / "E20/identified_sets_summary.csv")
    ids = ids_all[ids_all.scorer == "tfidf_lr"].set_index("env")  # primary scorer: source test split as reference
    mc = pd.read_csv(R / "E20/m_check.csv"); vs = pd.read_csv(R / "E20/validation_summary.csv")
    rx_all = pd.read_csv(R / "E20/relaxation_summary.csv"); bd_all = pd.read_csv(R / "E20/breakdown_vs_source_auroc.csv")
    rx = rx_all[rx_all.scorer == "tfidf_lr"]; bd = bd_all[bd_all.scorer == "tfidf_lr"]
    rows = []
    for e in ENVS:
        s = ids.loc[e]
        rows.append({"Cell": ENAME[e], "AL": f"[{s.auroc_AL_lo:.2f}, {s.auroc_AL_hi:.2f}]",
                     "AL+M": f"[{s.auroc_ALM_lo:.3f}, {s.auroc_ALM_hi:.3f}]", "AL+M, 95\\% CS": f"[{s.auroc_ALM_cs_lo:.3f}, {s.auroc_ALM_cs_hi:.3f}]",
                     "Const.\\ link": f"{s.auroc_const:.3f}",
                     "NLL, AL+M": f"[{s.nll_ALM_lo:.3f}, {s.nll_ALM_hi:.3f}]", "Brier, AL+M": f"[{s.brier_ALM_lo:.3f}, {s.brier_ALM_hi:.3f}]",
                     "$\\pi^\\star$ [95\\% CS]": f"{s.prevalence_AL:.2f} [{s.prevalence_AL_lo:.2f}, {s.prevalence_AL_hi:.2f}]"})
    src = ids.src_auroc.iloc[0]
    tex_table(pd.DataFrame(rows), "tab_ident_sets",
              "Identified sets for the source-construct risk $R^\\star_e$ of the anchor-blind TF-IDF source model on each regime-B cell, without target labels (Proposition~\\ref{prop:ident}). The first four numeric columns are AUROC. AL: level-link transport only; AL+M: adds within-level monotonicity in the score. Point sets are seed-averaged (3 seeds); CS: 95\\% confidence set from 300 bootstrap draws over anchor publisher clusters and target clusters (outer quantiles of the endpoints). \\emph{Const.\\ link}: labels $q_i=\\hat g(L_i)$, which attain the AL+M lower AUROC bound and the AL+M upper NLL and Brier bounds. Source test AUROC " + f3(src) + ", NLL " + f3(ids.src_nll.iloc[0]) + ". $\\pi^\\star$: source-construct prevalence, identified under AL.",
              "tab:identsets", "lrrrrrrr", wide=True)
    W += nm("IdSrcAuc", f3(src))
    for e, k in CK.items():
        s = ids.loc[e]
        W += nm(f"IdAl{k}Lo", f"{s.auroc_AL_lo:.2f}") + nm(f"IdAl{k}Hi", f"{s.auroc_AL_hi:.2f}")
        W += nm(f"IdAlm{k}Lo", f3(s.auroc_ALM_lo)) + nm(f"IdAlm{k}Hi", f3(s.auroc_ALM_hi)) + nm(f"IdAlmCs{k}Lo", f3(s.auroc_ALM_cs_lo))
        W += nm(f"IdRelaxOne{k}", f3(rx[(rx.env == e) & (rx.metric == "auroc") & (rx.delta == 1)].lo_ALM.iloc[0]))
        W += nm(f"IdPrev{k}", f"{s.prevalence_AL:.2f}")
    W += nm("IdAlLoMin", f"{ids.auroc_AL_lo.min():.2f}") + nm("IdAlHiMax", f"{ids.auroc_AL_hi.max():.2f}")
    W += nm("IdAlmLoMin", f"{ids.auroc_ALM_lo.min():.2f}") + nm("IdAlmLoMax", f"{ids.auroc_ALM_lo.max():.2f}")
    W += nm("IdAlmCsLoMin", f"{ids.auroc_ALM_cs_lo.min():.2f}") + nm("IdAlmCsLoMax", f"{ids.auroc_ALM_cs_lo.max():.2f}")
    W += nm("IdNllAlmHiMin", f"{ids.nll_ALM_hi.min():.2f}") + nm("IdNllAlmHiMax", f"{ids.nll_ALM_hi.max():.2f}")
    W += nm("IdSrcNll", f3(ids.src_nll.iloc[0]))
    W += nm("IdContainsSrc", str(int((bd.side == "contains").sum()))) + nm("IdContainsSrcN", str(len(bd)))
    r1 = rx[(rx.metric == "auroc") & (rx.delta == 1)]
    W += nm("IdRelaxOneMin", f3(r1.lo_ALM.min())) + nm("IdRelaxOneMax", f3(r1.lo_ALM.max()))
    rh = rx[(rx.metric == "auroc") & (rx.delta == 0.5)]
    W += nm("IdRelaxHalfMin", f3(rh.lo_ALM.min())) + nm("IdRelaxHalfMax", f3(rh.lo_ALM.max()))
    mcs = mc.groupby("scorer")[["within_level_auroc", "wl_auroc_lo", "wl_auroc_hi", "beta_hat", "beta_lo", "beta_hi"]].mean()
    for m, k in FAM.items():
        W += nm(f"MWl{k}", f"{mcs.loc[m, 'within_level_auroc']:.2f}") + nm(f"MWl{k}Lo", f"{mcs.loc[m, 'wl_auroc_lo']:.2f}") + nm(f"MWl{k}Hi", f"{mcs.loc[m, 'wl_auroc_hi']:.2f}")
        W += nm(f"MBeta{k}", f"{mcs.loc[m, 'beta_hat']:.2f}") + nm(f"MBeta{k}Lo", f"{mcs.loc[m, 'beta_lo']:.2f}") + nm(f"MBeta{k}Hi", f"{mcs.loc[m, 'beta_hi']:.2f}")
    W += nm("IdAnchors", f"{int(mc.n_anchors.iloc[0]):,}".replace(",", "{,}")) + nm("IdAnchorClusters", str(int(mc.n_clusters.iloc[0])))
    W += nm("IdValFolds", str(int(vs.n_folds.iloc[0]))) + nm("IdValCovMin", f"{vs.coverage_ALM.min():.2f}")
    W += nm("IdValCovHits", str(int(round(vs.coverage_ALM.min() * vs.n_folds.iloc[0]))))
    W += nm("IdValAucCovMin", f"{vs[vs.metric == 'auroc'].coverage_ALM.min():.2f}") + nm("IdValLossCovMin", f"{vs[vs.metric != 'auroc'].coverage_ALM.min():.2f}")
    W += nm("IdValLossCovMax", f"{vs[vs.metric != 'auroc'].coverage_ALM.max():.2f}") + nm("IdValNllMissMax", f"{vs[vs.metric == 'nll'].mean_miss_ALM.max():.3f}")
    va = vs[vs.metric == "auroc"].set_index("scorer")
    W += nm("IdValWidthAlAuc", f"{va.width_AL.mean():.2f}") + nm("IdValWidthAlmAuc", f"{va.width_ALM.mean():.2f}")
    W += nm("IdValConstGapMin", f"{va.truth_minus_const.min():.2f}") + nm("IdValConstGapMax", f"{va.truth_minus_const.max():.2f}")
    # transformer scorers: reference is the same model's source validation split (srcval_), not the test split
    tfm = [m for m in ("banglabert", "xlmr") if (ids_all.scorer == m).any()]
    if tfm:
        sc = ["tfidf_lr"] + tfm; rows = []
        for e in ENVS:
            r = {"Cell": ENAME[e]}
            for m in sc:
                q = ids_all[(ids_all.scorer == m) & (ids_all.env == e)].iloc[0]
                r[MNAME[m]] = f"[{q.auroc_ALM_lo:.3f}, {q.auroc_ALM_hi:.3f}] ({q.auroc_ALM_cs_lo:.3f})"
            rows.append(r)
        rows.append({"Cell": "Source held-out AUROC", **{MNAME[m]: f3(ids_all[ids_all.scorer == m].src_auroc.iloc[0]) for m in sc}})
        tex_table(pd.DataFrame(rows), "tab_ident_scorers",
                  "AL+M identified sets for source-construct AUROC by scorer (seed-averaged; in parentheses, lower end of the 95\\% confidence set). Last row: the same scorer's AUROC on held-out BanglaBait data, the source test split for TF-IDF and the cluster-grouped validation split for the transformers.",
                  "tab:identscorers", "l" + "r" * len(sc))
        for m in tfm:
            q = ids_all[ids_all.scorer == m]; b = bd_all[bd_all.scorer == m]; k = FAM[m]
            W += nm(f"IdAlmLo{k}Min", f3(q.auroc_ALM_lo.min())) + nm(f"IdAlmLo{k}Max", f3(q.auroc_ALM_lo.max()))
            W += nm(f"IdAlmCsLo{k}Min", f3(q.auroc_ALM_cs_lo.min())) + nm(f"IdSrcAuc{k}", f3(q.src_auroc.iloc[0]))
            W += nm(f"IdContains{k}", str(int((b.side == "contains").sum()))) + nm(f"IdContains{k}N", str(len(b)))
            for e, kk in CK.items():
                W += nm(f"IdAlm{kk}Lo{k}", f3(q[q.env == e].auroc_ALM_lo.iloc[0]))
        qa = ids_all[ids_all.scorer.isin(tfm)]
        W += nm("IdAlmLoTfMin", f3(qa.auroc_ALM_lo.min())) + nm("IdAlmLoTfMax", f3(qa.auroc_ALM_lo.max()))
        W += nm("IdAlmCsLoTfMin", f3(qa.auroc_ALM_cs_lo.min())) + nm("IdAlmCsLoTfMax", f3(qa.auroc_ALM_cs_lo.max()))
        bt = bd_all[bd_all.scorer.isin(tfm)]
        W += nm("IdContainsTf", str(int((bt.side == "contains").sum()))) + nm("IdContainsTfN", str(len(bt)))
        W += nm("IdSidesTf", ", ".join(f"{k_}: {v_}" for k_, v_ in bt.side.value_counts().items()))
    # figure: identified sets (a) and relaxation of the link (b)
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.3), gridspec_kw={"width_ratios": [1.25, 1]})
    a = axs[0]
    for j, e in enumerate(ENVS):
        s = ids.loc[e]
        a.plot([s.auroc_AL_lo, s.auroc_AL_hi], [j, j], color="#C9C9C9", lw=7, solid_capstyle="butt", label="AL" if j == 0 else None)
        a.plot([s.auroc_ALM_lo, s.auroc_ALM_hi], [j, j], color=C_AL, lw=3.2, solid_capstyle="butt", label="AL + M" if j == 0 else None)
        a.plot([s.auroc_ALM_cs_lo, s.auroc_ALM_lo], [j, j], color=C_AL, lw=0.8)
        a.plot([s.auroc_ALM_cs_lo] * 2, [j - 0.18, j + 0.18], color=C_AL, lw=0.8, label="AL + M, 95% CS lower end" if j == 0 else None)
    a.axvline(src, color=C_R6, ls="--", lw=0.9); a.text(src - 0.008, -0.62, "source test", color=C_R6, fontsize=6.5, ha="right")
    a.axvline(0.5, color=C_GREY, ls=":", lw=0.6)
    a.set_yticks(range(len(ENVS))); a.set_yticklabels([ENAME[e].replace(" (B)", "") for e in ENVS]); a.invert_yaxis()
    a.set_ylim(len(ENVS) + 0.25, -0.9)
    a.set_xlim(0.1, 1.0); a.set_xlabel(r"source-construct AUROC (against $Y_A$)")
    a.set_title("(a) Identified sets, TF-IDF source model", loc="left")
    a.legend(frameon=False, loc="lower left", fontsize=6.3, ncol=3, handlelength=1.6, columnspacing=1.0, borderaxespad=0.1)
    a = axs[1]
    colsE = {"banglaclick:news": C_SRC, "banglaclick:facebook": "#1F5F8B", "banglaclick:twitter": "#6A8F3A", "banglaclick:youtube": "#B4532A"}
    for e in ENVS:
        g = rx[(rx.env == e) & (rx.metric == "auroc")].sort_values("delta")
        a.plot(g.delta, g.lo_ALM, "-o", ms=2.6, lw=1.0, color=colsE[e], label=ENAME[e].replace(" (B)", ""))
    a.axhline(0.5, color=C_GREY, ls=":", lw=0.6)
    a.set_xlabel(r"link misspecification $\delta$ (logit scale)"); a.set_ylabel("AL + M lower bound on AUROC")
    a.set_title(r"(b) Relaxed link: $|\mathrm{logit}\,h-\mathrm{logit}\,g|\leq\delta$", loc="left")
    a.legend(frameon=False, fontsize=6.3)
    fig.tight_layout(); save(fig, "fig9_identified_sets")

if (R / "E21/design_summary.csv").exists():
    ds = pd.read_csv(R / "E21/design_summary.csv")
    DES = {"srs": "SRS", "level_prop": "level-stratified", "level_x_score_neyman": "level $\\times$ score, Neyman"}
    EST = {"sample_only": "sample only (HT)", "diff": "difference est.", "scl_refit": "SCL refit", "model_only": "link only (no labels)"}
    rows = []
    h = ds[ds.n == 100]
    for d_, dn in DES.items():
        for est, en in EST.items():
            if est == "model_only" and d_ != "srs":
                continue  # the label-free plug-in does not depend on the design
            r = {"Design": dn if est != "model_only" else "(any)", "Estimator": en}
            for m in ("tfidf_lr", "banglabert", "xlmr"):
                v = h[(h.design == d_) & (h.estimator == est) & (h.scorer == m)].iloc[0]
                mn = MNAME[m].replace(" LR", "")
                r[mn + " RMSE"] = f3(v.auroc_rmse)
                r[mn + " cov."] ="--" if pd.isna(v.auroc_ci_coverage) else f"{v.auroc_ci_coverage:.2f}"
            rows.append(r)
    tex_table(pd.DataFrame(rows), "tab_design",
              "Anchor-audit designs on news anchors ($n=100$ labelled anchors per draw, " + str(int(ds.draws.iloc[0])) + " draws; Section~\\ref{sec:anchoraudit}). RMSE of the estimated source-construct AUROC against its value on the held-out half of the anchor clusters, and coverage (cov.) of the nominal 95\\% design-based interval. \\emph{Link only} uses the transported link and no new labels: an in-distribution reference with no design-based guarantee under shift. SCL refit: the score-conditional link refitted on the sample alone.",
              "tab:design", "ll" + "r" * 6, wide=True)
    for d_, k in (("srs", "Srs"), ("level_x_score_neyman", "Ney"), ("level_prop", "Lev")):
        red = 1 - h[(h.design == d_) & (h.estimator == "diff")].set_index("scorer").auroc_rmse / h[(h.design == d_) & (h.estimator == "sample_only")].set_index("scorer").auroc_rmse
        W += nm(f"DesRed{k}Min", f"{100 * red.min():.0f}") + nm(f"DesRed{k}Max", f"{100 * red.max():.0f}")
    dh = h[h.estimator == "diff"]; sh = h[h.estimator == "sample_only"]; mh = h[h.estimator == "model_only"]
    W += nm("DesDiffRmseMin", f"{dh.auroc_rmse.min():.3f}") + nm("DesDiffRmseMax", f"{dh.auroc_rmse.max():.3f}")
    W += nm("DesSampleRmseMin", f"{sh.auroc_rmse.min():.3f}") + nm("DesSampleRmseMax", f"{sh.auroc_rmse.max():.3f}")
    W += nm("DesModelRmseMin", f"{mh.auroc_rmse.min():.3f}") + nm("DesModelRmseMax", f"{mh.auroc_rmse.max():.3f}")
    cv = ds[(ds.estimator == "diff") & (ds.n >= 100)].auroc_ci_coverage
    W += nm("DesCovMin", f"{cv.min():.2f}") + nm("DesCovMax", f"{cv.max():.2f}")
    c25 = ds[(ds.estimator == "diff") & (ds.n == 25)].auroc_ci_coverage
    W += nm("DesCovSmallMin", f"{c25.min():.2f}") + nm("DesCovSmallMax", f"{c25.max():.2f}")
    W += nm("DesDraws", str(int(ds.draws.iloc[0])))
    # figure: RMSE and coverage against n (mean over the three scorers; band = scorer range)
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.2))
    sty = [("srs", "sample_only", C_GREY, "--", "SRS, sample only"), ("level_x_score_neyman", "sample_only", C_R6, "--", "Neyman, sample only"),
           ("srs", "diff", "#6A8F3A", "-", "SRS, difference"), ("level_x_score_neyman", "diff", C_AL, "-", "Neyman, difference"),
           ("srs", "model_only", C_SRC, ":", "link only (no labels)")]
    for d_, est, c, ls, lab in sty:
        q = ds[(ds.design == d_) & (ds.estimator == est)].groupby("n")
        for ax, col in zip(axs, ("auroc_rmse", "auroc_ci_coverage")):
            if est == "model_only" and col == "auroc_ci_coverage":
                continue
            mu, lo, hi = q[col].mean(), q[col].min(), q[col].max()
            ax.fill_between(mu.index, lo, hi, color=c, alpha=0.12, lw=0)
            ax.plot(mu.index, mu, ls=ls, marker="o", ms=2.4, lw=1.0, color=c, label=lab)
    axs[0].set_xlabel("labelled anchors $n$"); axs[0].set_ylabel("AUROC RMSE"); axs[0].set_title("(a) Accuracy", loc="left")
    axs[1].axhline(0.95, color=C_GREY, lw=0.6, ls=":"); axs[1].set_ylim(0.6, 1.0)
    axs[1].set_xlabel("labelled anchors $n$"); axs[1].set_ylabel("coverage of 95% interval"); axs[1].set_title("(b) Interval coverage", loc="left")
    axs[0].legend(frameon=False, fontsize=6.2)
    fig.tight_layout(); save(fig, "fig10_design")

if (R / "E22/cut_points.csv").exists():
    cp = pd.read_csv(R / "E22/cut_points.csv").assign(model="tfidf_lr")
    cp.loc[~cp.has_C, "tau_C"] = np.nan
    if (R / "E22/cut_points_transformer.csv").exists():
        cp = pd.concat([cp, pd.read_csv(R / "E22/cut_points_transformer.csv")], ignore_index=True)
    hc = cp[cp.has_C]
    W += nm("CutOrderShare", f"{100 * ((hc.theta_1 < hc.tau_C) & (hc.tau_C < hc.tau_A)).mean():.0f}") + nm("CutFits", str(len(hc)))
    for m, k in FAM.items():
        d = cp[cp.model == m]
        if d.empty:
            continue
        W += nm(f"CutGapA{k}", f"{(d.tau_A - d.theta_2).mean():+.2f}")
        W += nm(f"CutGapA{k}Min", f"{(d.tau_A - d.theta_2).min():+.2f}") + nm(f"CutGapA{k}Max", f"{(d.tau_A - d.theta_2).max():+.2f}")
        W += nm(f"CutABelowTwo{k}", str(int((d.tau_A < d.theta_2).sum()))) + nm(f"CutN{k}", str(len(d)))
        W += nm(f"CutTauC{k}Min", f"{d.tau_C.min():+.2f}") + nm(f"CutTauC{k}Max", f"{d.tau_C.max():+.2f}")
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    marks = [("theta_1", r"$\theta_1$ (B: $L\geq1$)", C_R6, "o"), ("tau_C", r"$\tau_C$ (BaitBuster)", "#6A8F3A", "D"),
             ("tau_A", r"$\tau_A$ (BanglaBait)", C_SRC, "s"), ("theta_2", r"$\theta_2$ (B: $L\geq2$)", C_AL, "o")]
    fams = [m for m in FAM if m in set(cp.model)]
    for j, m in enumerate(fams):
        d = cp[cp.model == m]
        for k, (col, lab, c, mk) in enumerate(marks):
            v = d[col].dropna()
            ax.plot([v.min(), v.max()], [j, j], color=c, lw=0.8, alpha=0.7)
            ax.plot(v.mean(), j, marker=mk, color=c, ms=4.5, ls="none", label=lab if j == 0 else None, mfc=c if col != "tau_C" else "white")
    ax.set_yticks(range(len(fams))); ax.set_yticklabels([MNAME[m] for m in fams]); ax.invert_yaxis()
    ax.set_xlabel("latent clickbait intensity (logit scale of each model)")
    ax.set_ylim(len(fams) - 0.4, -1.3)
    ax.legend(frameon=False, fontsize=6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.02), handletextpad=0.3, columnspacing=0.8)
    save(fig, "fig11_cutpoints")

if (R / "E17/protocol_contrasts_cluster_bootstrap.csv").exists():
    ct = pd.read_csv(R / "E17/protocol_contrasts_cluster_bootstrap.csv")
    lc = ct[(ct.protocol == "latent_cuts") & (ct.reference == "pooled_erm")]
    if len(lc):
        def cell(v):
            return f"{v['diff']:+.3f}" + ("$^{*}$" if v.p_holm < 0.05 else "")
        rows = []
        for m in ("tfidf_lr", "banglabert", "xlmr"):
            for lab in ("r6", "aligned"):
                for met, mt in (("auroc", "AUROC"), ("nll", "NLL")):
                    r = {"Model": MNAME[m], "Label": LAB_TEX[lab], "Metric": mt}
                    for tg in TGTS:
                        v = lc[(lc.model == m) & (lc.label == lab) & (lc.metric == met) & (lc.target == tg)]
                        r[TSHORT[tg]] = cell(v.iloc[0]) if len(v) else "--"
                    rows.append(r)
        tex_table(pd.DataFrame(rows), "tab_latent_vs_pooled",
                  "Regime cut-point model minus pooled ERM (the nested equal-cut model) under leave-one-environment-out training: paired differences over three seeds with publisher/channel-cluster bootstrap. $^{*}$Holm-adjusted $p<0.05$ within each metric over all paired contrasts. Positive AUROC and negative NLL differences favour the cut-point model. Clusters: Facebook 26, YT-B 10, YT-C 13.",
                  "tab:latent", "lll" + "r" * len(TGTS), wide=True)
        for m, k in FAM.items():
            for lab, lk in (("r6", "Rsix"), ("aligned", "Al")):
                for tg in TGTS:
                    for met, mk in (("auroc", "Auc"), ("nll", "Nll")):
                        v = lc[(lc.model == m) & (lc.label == lab) & (lc.metric == met) & (lc.target == tg)]
                        if len(v):
                            W += nm(f"Lat{k}{lk}{TMAC[tg]}{mk}", f"{v['diff'].iloc[0]:+.3f}")
        sig = lc[lc.p_holm < 0.05]
        good = ((sig.metric == "auroc") & (sig["diff"] > 0)) | ((sig.metric != "auroc") & (sig["diff"] < 0))
        W += nm("LatSigPos", str(int(good.sum()))) + nm("LatSigNeg", str(int((~good).sum()))) + nm("LatTotal", str(len(lc)))
    tfc = ct[(ct.model == "tfidf_lr") & (ct.reference == "source_only") & (ct.metric == "auroc")].set_index(["protocol", "label", "target"])
    for pr, pk in (("pooled_erm", "Pool"), ("group_dro", "Dro"), ("iw_pooled", "Iwp")):
        for lab, lk in (("r6", "Rsix"), ("aligned", "Al")):
            for tg, tk in (("banglaclick:facebook", "Fb"), ("banglaclick:twitter", "Tw")):
                W += nm(f"TfSrc{pk}{lk}{tk}", f"{abs(tfc.loc[(pr, lab, tg), 'diff']):.2f}")
    nl_ = bm[bm.model == "tfidf_lr"].pivot_table(index=["label", "target"], columns="protocol", values="nll").loc["r6"]
    gn = (nl_.source_only - nl_[["pooled_erm", "group_dro"]].T).T.loc[["banglaclick:news", "banglaclick:twitter", "banglaclick:youtube"]]
    W += nm("TfPoolNllGainMin", f"{gn.values.min():.2f}") + nm("TfPoolNllGainMax", f"{gn.values.max():.2f}")
    sd_ = pd.read_csv(R / "E4/decomposition_summary.csv"); sd_ = sd_[(sd_.aligned_label == "q_al") & sd_.env.isin(["banglaclick:news", "banglaclick:facebook", "banglaclick:twitter"])]
    W += nm("SoftConstructPos", str(int((sd_.nll_construct > 0).sum()))) + nm("SoftConstructN", str(len(sd_)))
    W += nm("SoftConstructMin", f"{sd_.nll_construct.min():+.3f}")
    ls_ = ct[(ct.protocol == "latent_cuts") & (ct.reference == "source_only") & (ct.metric == "auroc")]
    W += nm("LatSrcSigPos", str(int(((ls_.p_holm < 0.05) & (ls_["diff"] > 0)).sum()))) + nm("LatSrcSigNeg", str(int(((ls_.p_holm < 0.05) & (ls_["diff"] < 0)).sum())))
    W += nm("LatSrcTotal", str(len(ls_)))
    so = ct[(ct.reference == "source_only") & ct.protocol.isin(["pooled_erm", "group_dro"]) & ct.model.isin(["banglabert", "xlmr"]) & (ct.metric == "auroc")]
    W += nm("TrSrcSigAuc", str(int((so.p_holm < 0.05).sum()))) + nm("TrSrcTotalAuc", str(len(so)))
    for m, k in (("banglabert", "Bbert"), ("xlmr", "Xlmr")):
        for pr, pk in (("pooled_erm", "Pool"), ("group_dro", "Dro")):
            for lab, lk in (("r6", "Rsix"), ("aligned", "Al")):
                v = so[(so.model == m) & (so.protocol == pr) & (so.label == lab) & (so.target == "banglaclick:twitter")]
                if len(v):
                    W += nm(f"TrSrc{pk}{lk}Tw{k}", f"{v['diff'].iloc[0]:+.3f}")
    vv = so[(so.model == "xlmr") & (so.protocol == "pooled_erm") & (so.label == "r6") & (so.target == "banglaclick:facebook")].iloc[0]
    W += nm("TrSrcPoolRsixFbXlmr", f"{vv['diff']:+.3f}") + nm("TrSrcPoolRsixFbXlmrP", f"{vv.p_holm:.2f}")

with open(PAPER / "numbers.tex", "a", encoding="utf-8") as fh:
    fh.write(W)
print("E20-E22 assets written;", W.count("newcommand"), "macros")


# ---------------------------------------------------------------- S5: reviewer-round analyses (E4T, E21S, E23, E24)
W = ""
CELLS5 = ["banglaclick:news", "banglaclick:facebook", "banglaclick:twitter", "banglaclick:youtube"]
TK = {"banglaclick:news": "News", "banglaclick:news_nonanchor": "NewsNa", "banglaclick:facebook": "Fb", "banglaclick:twitter": "Tw",
      "banglaclick:youtube": "Yt", "baitbuster:youtube": "YtC", "banglabait:news": "Src"}
MK = {"ebm": "Ebm", "tfidf_lr": "Tf", "banglabert": "Bbert", "xlmr": "Xlmr"}
CUTNAME = {"level>=1": r"$\Lv\geq1$ (published)", "level>=2": r"$\Lv\geq2$", "level>=3": r"$\Lv\geq3$", "trigger>=1": r"trigger $\geq1$",
           "trigger>=2": r"trigger $\geq2$", "quality_fails>=1": r"FAST failures $\geq1$", "quality_fails>=2": r"FAST failures $\geq2$",
           "quality_fails>=3": r"FAST failures $\geq3$", "trigger>=1 & level>=1": r"trigger $\geq1$ and $\Lv\geq1$",
           "trigger>=1 | level>=2": r"trigger $\geq1$ or $\Lv\geq2$", "level>=1 & not(attribution only)": r"$\Lv\geq1$ excl.\ attribution-only"}
E24D = R / "E24"
if (E24D / "binarisations_on_anchors.csv").exists():
    b = pd.read_csv(E24D / "binarisations_on_anchors.csv")
    cf = json.loads((E24D / "binarisations_crossfit.json").read_text())
    t = pd.DataFrame({"Binarisation of the rubric": b.cut.map(CUTNAME), r"$\kappa$": b.kappa.map(f3), "PABAK": b.pabak.map(f3),
                      r"$\kappa_{\max}$": b.kappa_max.map(f3), "Disagr.": b.disagreement.map(f3), "Prev.": b.prevalence.map(f3),
                      "Selected": b.selection_share.map(lambda v: f"{100 * v:.0f}\\%")})
    tex_table(t, "tab_binarisations", r"Agreement of rubric binarisations with the source label $\YA$ on the anchors. Selected: share of cross-fitted publisher folds (5 folds $\times$ 20 repetitions) in which the cut has the highest training-fold $\kappa$. Source prevalence on the anchors " + f3(b.prev_source.iloc[0]) + ".",
              "tab:binarisations", "lrrrrrr")
    bi = b.set_index("cut")
    for c, k in (("level>=1", "LOne"), ("level>=2", "LTwo"), ("trigger>=1", "TrOne"), ("quality_fails>=2", "QTwo")):
        W += nm(f"Bin{k}Kappa", f"{bi.loc[c, 'kappa']:.2f}") + nm(f"Bin{k}Pabak", f"{bi.loc[c, 'pabak']:.2f}") + nm(f"Bin{k}Prev", f"{bi.loc[c, 'prevalence']:.2f}")
        W += nm(f"Bin{k}Kmax", f"{bi.loc[c, 'kappa_max']:.2f}") + nm(f"Bin{k}Sel", f"{100 * bi.loc[c, 'selection_share']:.0f}")
    W += nm("BinCfKappa", f"{cf['crossfit_kappa_mean']:.2f}") + nm("BinCfLo", f"{cf['crossfit_kappa_lo']:.2f}") + nm("BinCfHi", f"{cf['crossfit_kappa_hi']:.2f}")
    W += nm("BinFolds", str(cf["folds"])) + nm("BinReps", str(cf["reps"]))
    # frozen AUROC by binarisation
    ab = pd.read_csv(E24D / "frozen_auroc_by_binarisation.csv")
    ms = pd.read_csv(E24D / "frozen_metric_suite.csv")
    src = ms[(ms.env == "banglabait:news")].set_index("model").auroc
    rows = []
    for e in CELLS5:
        for m in MODELS:
            g_ = ab[(ab.env == e) & (ab.model == m)].set_index("cut")
            prev = [f3(g_.loc[c, "prevalence"]) for c in ("level>=1", "level>=2", "trigger>=1")] if m == MODELS[0] else ["", "", ""]
            rows.append([ENAME[e] if m == MODELS[0] else "", MNAME[m], f3(src[m])] + [f3(g_.loc[c, "auroc"]) for c in ("level>=1", "level>=2", "trigger>=1")] + prev)
            for c, ck in (("level>=1", "LOne"), ("level>=2", "LTwo"), ("trigger>=1", "TrOne")):
                W += nm(f"Cut{ck}{TK[e]}{MK[m]}", f3(g_.loc[c, "auroc"]))
    for m in MODELS:
        W += nm(f"SrcAuc{MK[m]}", f3(src[m]))
    t = pd.DataFrame(rows, columns=["Cell", "Model", "Source", r"$\Lv\geq1$", r"$\Lv\geq2$", r"trig.$\geq1$", r"$\pi_{\Lv\geq1}$", r"$\pi_{\Lv\geq2}$", r"$\pi_{\mathrm{trig}}$"])
    tex_table(t, "tab_auroc_by_cut", r"AUROC of frozen source models (mean over five seeds) on each BanglaClick cell under three binarisations of the same rubric, with the source (BanglaBait test) AUROC. Right: prevalence of each binarisation. No model is retrained; only the target label changes.",
              "tab:aurocbycut", "llrrrrrrr", wide=True)
    # component-based link transport
    cl = pd.read_csv(E24D / "component_link_transport.csv").set_index("env")
    rows = []
    for e in ["banglaclick:news", "banglaclick:news_nonanchor", "banglaclick:facebook", "banglaclick:twitter", "banglaclick:youtube"]:
        r_ = cl.loc[e]
        rows.append([ENAME[e]] + [f"{r_[f'g_e_{l}']:.2f}" for l in range(4)] + [f"{r_.delta_implied_all_levels:.2f} [{r_.delta_implied_all_levels_lo:.2f}, {r_.delta_implied_all_levels_hi:.2f}]"])
        W += nm(f"CompDelta{TK[e]}", f"{r_.delta_implied_all_levels:.2f}") + nm(f"CompDelta{TK[e]}Lo", f"{r_.delta_implied_all_levels_lo:.2f}") + nm(f"CompDelta{TK[e]}Hi", f"{r_.delta_implied_all_levels_hi:.2f}")
    rows.append(["Anchors (fitted link)"] + [f"{cl.iloc[0][f'g_news_{l}']:.2f}" for l in range(4)] + ["--"])
    t = pd.DataFrame(rows, columns=["Cell", r"$\hat g_e(0)$", r"$\hat g_e(1)$", r"$\hat g_e(2)$", r"$\hat g_e(3)$", r"$\hat\delta_e$ [95\% CI]"])
    tex_table(t, "tab_complink", r"Component-based link transport (Eq.~\eqref{eq:compdelta}). $\hat g_e(l)$ averages the anchor-fitted $\hat m(l,c)=\hat P(\YA=1\mid\Lv=l,C=c)$ over the rubric components $c$ observed in cell $e$; $\hat\delta_e$ is the largest logit distance from the anchor link. Intervals: 500 bootstrap draws over anchor publishers. $\hat\delta_e$ (maximum over all four levels) reflects only shifts in the observed components; it neither bounds nor captures shifts that the components do not explain.",
              "tab:complink", "lrrrrr")
    mx = pd.read_csv(E24D / "component_mix_by_level.csv").set_index(["env", "level"])
    W += nm("MixYtOneTrig", f"{mx.loc[('banglaclick:youtube', 1), 'trigger_ge1']:.2f}") + nm("MixAncOneTrig", f"{mx.loc[('anchors', 1), 'trigger_ge1']:.2f}")
    W += nm("MixYtOneTone", f"{mx.loc[('banglaclick:youtube', 1), 'fail_t']:.2f}") + nm("MixAncOneTone", f"{mx.loc[('anchors', 1), 'fail_t']:.2f}")
    # YouTube without the dominant channel
    yt = pd.read_csv(E24D / "youtube_without_top_channel.csv")
    rows = []
    for _, r_ in yt.iterrows():
        rows.append(["All (10 ch.)" if r_.subset == "all" else "Excl.\\ top (9 ch.)", MNAME[r_.model], str(int(r_.n)), f3(r_.auroc_r6), f3(r_.auroc_aligned),
                     f"{r_.nll_construct:+.3f}", f"{r_.nll_prior:+.3f}", f"{r_.nll_residual:+.3f}"])
    t = pd.DataFrame(rows, columns=["Subset", "Model", "$n$", r"AUROC $Y$", r"AUROC $Y^{*}$", r"$\phi_A$", r"$\phi_P$", "Residual"])
    tex_table(t, "tab_yt_top", r"YouTube (B) with and without its largest channel. Decomposition of the log-loss gap under $Y^{*}=1[\Lv\geq2]$ (mean over five seeds).",
              "tab:yttop", "llrrrrrr")
    wo = yt[yt.subset == "without_top_channel"]; al = yt[yt.subset == "all"]
    W += nm("YtTopN", str(int(al.n.iloc[0] - wo.n.iloc[0]))) + nm("YtRestN", str(int(wo.n.iloc[0])))
    W += nm("YtTopConstrPos", str(int((wo.nll_construct > 0).sum()))) + nm("YtResAllMin", f"{al.nll_residual.min():.3f}") + nm("YtResAllMax", f"{al.nll_residual.max():.3f}")
    W += nm("YtResWoMin", f"{wo.nll_residual.min():.3f}") + nm("YtResWoMax", f"{wo.nll_residual.max():.3f}")
    W += nm("YtPrevAlAll", f"{al.prev_aligned.iloc[0]:.2f}") + nm("YtPrevAlWo", f"{wo.prev_aligned.iloc[0]:.2f}")
    # full metric suite (frozen models)
    rows = []
    for e in ["banglabait:news"] + CELLS5 + ["baitbuster:youtube"]:
        for lab in ("y_r6", "y_al"):
            g_ = ms[(ms.env == e) & (ms.label == lab)]
            if not len(g_):
                continue
            for m in MODELS:
                r_ = g_[g_.model == m].iloc[0]
                first = m == MODELS[0]
                rows.append([ENAME[e] if first and lab == "y_r6" else "", ("$Y$" if lab == "y_r6" else "$Y^{*}$") if first else "", MNAME[m],
                             f3(r_.prevalence) if first else "", f3(r_.auroc), f3(r_.auprc), f3(r_.nll), f3(r_.brier), f3(r_.ece),
                             f3(r_.balanced_accuracy), f3(r_.macro_f1)])
                for met, mk2 in (("auprc", "Auprc"), ("ece", "Ece"), ("balanced_accuracy", "Bacc"), ("macro_f1", "Mfone"), ("auroc", "Auc")):
                    W += nm(f"Ms{TK[e]}{'Al' if lab == 'y_al' else 'Rsix'}{MK[m]}{mk2}", f3(r_[met]))
    t = pd.DataFrame(rows, columns=["Cell", "Label", "Model", r"$\pi$", "AUROC", "AUPRC", "NLL", "Brier", "ECE", "BAcc", "Macro-F1"])
    tex_table(t, "tab_metric_suite", r"Full metric suite for the frozen source models (mean over five seeds). $Y$: published label; $Y^{*}=1[\Lv\geq2]$. Threshold metrics at 0.5. ECE: 15 equal-width bins.",
              "tab:metricsuite", "lllrrrrrrrr", wide=True)
    if (E24D / "protocol_contrasts_jackknife.csv").exists():
        jk = pd.read_csv(E24D / "protocol_contrasts_jackknife.csv"); bt_ = pd.read_csv(R / "E17/protocol_contrasts_cluster_bootstrap.csv")
        kk = ["model", "label", "target", "protocol", "reference", "metric"]
        jm = jk.merge(bt_[kk + ["p_holm"]], on=kk, how="left")
        for met, mk3 in (("auroc", "Auc"), ("nll", "Nll")):
            g_ = jm[jm.metric == met]
            W += nm(f"Jk{mk3}N", str(len(g_))) + nm(f"Jk{mk3}Boot", str(int((g_.p_holm < 0.05).sum()))) + nm(f"Jk{mk3}Jack", str(int((g_.p_holm_jack < 0.05).sum())))
            W += nm(f"Jk{mk3}Both", str(int(((g_.p_holm < 0.05) & (g_.p_holm_jack < 0.05)).sum())))
        ja = jm[jm.metric == "auroc"]
        lost = ja[(ja.p_holm < 0.05) & (ja.p_holm_jack >= 0.05)]
        both = ja[(ja.p_holm < 0.05) & (ja.p_holm_jack < 0.05)]
        W += nm("JkAucLost", str(len(lost))) + nm("JkLostFb", str(int((lost.target == "banglaclick:facebook").sum())))
        W += nm("JkBothTw", str(int((both.target == "banglaclick:twitter").sum())))
        rows = []
        for _, r_ in ja[(ja.p_holm < 0.05) | (ja.p_holm_jack < 0.05)].iterrows():
            rows.append([MNAME[r_.model], "$Y$" if r_.label == "r6" else "$Y^{*}$", ENAME[r_.target], r_.protocol.replace("_", " "), r_.reference.replace("_", " "),
                         f"{r_['diff']:+.3f}", str(int(r_.clusters)), f"{r_.p_holm:.3f}", f"{r_.p_holm_jack:.3f}"])
        t = pd.DataFrame(rows, columns=["Model", "Label", "Target", "Protocol", "Reference", r"$\Delta$AUROC", "$G$", r"$p_{\mathrm{boot}}$", r"$p_{\mathrm{jack}}$"])
        tex_table(t, "tab_jackknife", r"AUROC contrasts significant under either test. $p_{\mathrm{boot}}$: pairs cluster bootstrap, Holm over all paired contrasts of the benchmark; $p_{\mathrm{jack}}$: leave-one-cluster-out jackknife $t$-test with $G-1$ degrees of freedom, Holm over the same contrasts. $G$: clusters.",
                  "tab:jackknife", "llllllrrr", wide=True)
        for (m_, lab_, tg_, pr_, rf_), k_ in ((("banglabert", "r6", "banglaclick:facebook", "latent_cuts", "pooled_erm"), "LatFbBbert"),
                                              (("xlmr", "r6", "banglaclick:facebook", "latent_cuts", "pooled_erm"), "LatFbXlmr"),
                                              (("tfidf_lr", "r6", "banglaclick:facebook", "latent_cuts", "pooled_erm"), "LatFbTf"),
                                              (("tfidf_lr", "r6", "banglaclick:facebook", "pooled_erm", "source_only"), "PoolFbTf")):
            v = ja[(ja.model == m_) & (ja.label == lab_) & (ja.target == tg_) & (ja.protocol == pr_) & (ja.reference == rf_)].iloc[0]
            W += nm(f"Jk{k_}P", f"{v.p_holm_jack:.2f}") + nm(f"Jk{k_}Pb", f"{v.p_holm:.3f}")
    if (E24D / "scl_news_nonanchor.csv").exists():
        sc = pd.read_csv(E24D / "scl_news_nonanchor.csv")
        rows = []
        for s_, g_ in sc.groupby("scorer", sort=False):
            g_ = g_.set_index("label"); k_ = MK[s_]; r1 = g_.loc["scl_g1"]
            rows.append([MNAME[s_], f3(g_.loc["r6", "auroc"]), f3(g_.loc["hard_aligned", "auroc"]), f"{f3(r1.auroc)} [{f3(r1.auroc_lo)}, {f3(r1.auroc_hi)}]", f3(r1.src_auroc)])
            W += nm(f"SclNa{k_}", f3(r1.auroc)) + nm(f"SclNa{k_}Lo", f3(r1.auroc_lo)) + nm(f"SclNa{k_}Hi", f3(r1.auroc_hi)) + nm(f"SclNaSrc{k_}", f3(r1.src_auroc))
            W += nm(f"SclNaPub{k_}", f3(g_.loc["r6", "auroc"])) + nm(f"SclNaDrop{k_}", f"{r1.src_auroc - r1.auroc_lo:.2f}")
            W += nm(f"SclNa{k_}GZero", f3(g_.loc["scl_g0", "auroc"])) + nm(f"SclNa{k_}GHalf", f3(g_.loc["scl_g0.5", "auroc"])) + nm(f"SclNa{k_}GThreeQ", f3(g_.loc["scl_g0.75", "auroc"]))
            rows[-1][3:3] = [f3(g_.loc["scl_g0", "auroc"]), f3(g_.loc["scl_g0.5", "auroc"])]
        t = pd.DataFrame(rows, columns=["Scorer", "Published $Y$", r"$1[\Lv\geq2]$", r"$\gamma=0$", r"$\gamma=0.5$", r"$\gamma=1$ [95\% CI]", "Source held-out"])
        tex_table(t, "tab_scl_nonanchor", r"Source-construct AUROC on BanglaClick news items that are not anchors (anchors only fit the link), for the three anchor-blind source-only scorers (mean over three seeds; intervals: 300 bootstrap draws over anchor publishers and target clusters). $\gamma$ scales the anchor-fitted score slope of the link: $\gamma=1$ is the estimate under Assumption~\ref{as:crep}, $\gamma=0$ the level-only link (the lower endpoint of Proposition~\ref{prop:ident}(b)). Source held-out: TF-IDF on the BanglaBait test set, transformers on their cluster-grouped BanglaBait validation split.",
                  "tab:sclnonanchor", "lrrrrrr")

# decomposition under both aligned labels, with the interaction term
d2 = pd.read_csv(R / "E4/decomposition_summary.csv"); d2 = d2[d2.aligned_label == "y_al"].assign(lab="L2")
if (R / "E4T/decomposition_summary.csv").exists():
    dt = pd.read_csv(R / "E4T/decomposition_summary.csv"); dt = dt[dt.aligned_label == "y_al"].assign(lab="Tr")
    dd = pd.concat([d2, dt]); dd["inter"] = (dd.nll_R00 - dd.nll_RA) - (dd.nll_RP - dd.nll_RAP)
    # skill-scale residual: subtract the entropy of each side's prevalence, so that R_AP (target prevalence) and R_S (source
    # prevalence) are compared as excess log loss over a constant predictor at the true base rate
    Hb = lambda p: float(-(p * np.log(p) + (1 - p) * np.log(1 - p)))
    abx = pd.read_csv(R / "E24/frozen_auroc_by_binarisation.csv"); msx = pd.read_csv(R / "E24/frozen_metric_suite.csv")
    piS_ = float(msx[msx.env == "banglabait:news"].prevalence.iloc[0])
    def _skill(r_):
        # prevalence of the hard aligned label in this cell (prev_aligned = mean of y_al; cell-specific, incl. non-anchor news)
        return (r_.nll_RAP - Hb(float(r_.prev_aligned))) - (r_.nll_RS - Hb(piS_))
    dd["skill_res"] = dd.apply(_skill, axis=1)
    rows = []
    for e in CELLS5 + ["banglaclick:news_nonanchor"]:
        for m in MODELS:
            a_ = dd[(dd.env == e) & (dd.model == m) & (dd.lab == "L2")]; b_ = dd[(dd.env == e) & (dd.model == m) & (dd.lab == "Tr")]
            if not len(a_) or not len(b_):
                continue
            a_, b_ = a_.iloc[0], b_.iloc[0]
            fs = lambda v: "--" if pd.isna(v) else f"{v:+.3f}"
            rows.append([ENAME[e] if m == MODELS[0] else "", MNAME[m], f"{a_.nll_gap:.3f}"] + [f"{x:+.3f}" for x in (a_.nll_construct, a_.nll_prior, a_.inter, a_.nll_residual)]
                        + [fs(a_.skill_res)] + [f"{x:+.3f}" for x in (b_.nll_construct, b_.nll_prior, b_.inter, b_.nll_residual)] + [fs(b_.skill_res)])
            for lab_, r_ in (("L", a_), ("T", b_)):
                W += nm(f"Dec{lab_}{TK[e]}{MK[m]}Con", f"{r_.nll_construct:+.3f}") + nm(f"Dec{lab_}{TK[e]}{MK[m]}Pri", f"{r_.nll_prior:+.3f}")
                W += nm(f"Dec{lab_}{TK[e]}{MK[m]}Int", f"{r_.inter:+.3f}") + nm(f"Dec{lab_}{TK[e]}{MK[m]}Res", f"{r_.nll_residual:+.3f}")
    t = pd.DataFrame(rows, columns=["Cell", "Model", "Gap", r"$\phi_A$", r"$\phi_P$", "$I$", "Res.", r"Res.$_H$", r"$\phi_A$", r"$\phi_P$", "$I$", "Res.", r"Res.$_H$"])
    tex_table(t, "tab_decomp_both", r"Shapley decomposition of the log-loss gap $R_{00}-R_S$ under two proxies $Y^{*}$ of the source construct: $1[\Lv\geq2]$ (columns 4--8) and $1[\text{trigger}\geq1]$ (columns 9--13). $\phi_A$: label definition; $\phi_P$: prevalence (corrected to the true target prevalence of $Y^{*}$); $I=(R_{00}-R_{A})-(R_{P}-R_{AP})$: interaction, split equally by the Shapley rule; Res.: residual $R_{AP}-R_S$ (text shift plus proxy error); Res.$_H$: the residual on a skill scale, $[R_{AP}-H(\pi^{*}_e)]-[R_S-H(\pi_S)]$, which removes the difference in label entropy between target and source. Frozen models; on news they were trained on most anchor titles (Section~\ref{sec:data}). Mean over five seeds.",
              "tab:decompboth", "llr" + "r" * 10, wide=True)
    sk = dd[dd.env.isin(CELLS5)]
    for lab_, lk_ in (("L2", "L"), ("Tr", "T")):
        for e in CELLS5 + ["banglaclick:news_nonanchor"]:
            if e.endswith("nonanchor"):  # text-based models only (the cue model is not discussed for this subset)
                v = dd[(dd.lab == lab_) & (dd.env == e) & dd.model.isin(["tfidf_lr", "banglabert", "xlmr"])].skill_res
            else:
                v = sk[(sk.lab == lab_) & (sk.env == e)].skill_res
            W += nm(f"Skill{lk_}{TK[e]}Min", f"{v.min():+.3f}") + nm(f"Skill{lk_}{TK[e]}Max", f"{v.max():+.3f}")
    tr = dd[(dd.lab == "Tr") & dd.env.isin(CELLS5)]; l2 = dd[(dd.lab == "L2") & dd.env.isin(CELLS5)]
    W += nm("DecTrConPos", str(int((tr.nll_construct > 0).sum()))) + nm("DecTrN", str(len(tr)))
    W += nm("DecLTwoConPos", str(int((l2.nll_construct > 0).sum()))) + nm("DecLTwoN", str(len(l2)))
    W += nm("DecIntAbsMax", f"{dd[dd.env.isin(CELLS5)].inter.abs().max():.2f}")
    W += nm("DecIntMin", f"{dd[dd.env.isin(CELLS5)].inter.min():.2f}")
    jt = dd[dd.env.isin(CELLS5)].assign(js=lambda x: (x.nll_R00 - x.nll_RAP) / x.nll_gap)
    W += nm("DecJointShareLTwoNewsFbTwMin", f"{100 * jt[(jt.lab == 'L2') & (jt.env != 'banglaclick:youtube') & (jt.model != 'ebm')].js.min():.0f}")
    yy = tr[tr.env == "banglaclick:youtube"]
    W += nm("DecTrYtConMin", f"{yy.nll_construct.min():+.3f}") + nm("DecTrYtConMax", f"{yy.nll_construct.max():+.3f}")
    W += nm("DecTrYtResMin", f"{yy.nll_residual.min():.3f}") + nm("DecTrYtResMax", f"{yy.nll_residual.max():.3f}")

# ablation of the regime cut-point model (TF-IDF, LOEO)
if (R / "E23/ablation_summary.csv").exists():
    ab_ = pd.concat([pd.read_csv(R / "E23/ablation_summary.csv"), pd.read_csv(R / "E22/latent_cuts_summary.csv").drop(columns=["construct"]),
                     pd.read_csv(R / "E10/loeo_tfidf_summary.csv")], ignore_index=True)
    ab_ = ab_[ab_.model == "tfidf_lr"]
    PR = [("latent_cuts", "Full"), ("regime_bias", r"$-$cuts"), ("latent_noanchor", r"$-$anchors"), ("same_regime", r"$-$other reg."), ("pooled_erm", "Pooled ERM"), ("source_only", "Source")]
    PK = {"latent_cuts": "Full", "regime_bias": "NoCut", "latent_noanchor": "NoAnc", "same_regime": "Same", "pooled_erm": "Pool", "source_only": "Src"}
    abc = pd.read_csv(R / "E23/ablation_contrasts_vs_latent_cuts.csv")
    rows = []
    for lab in ("r6", "aligned"):
        for tg in CELLS5 + ["baitbuster:youtube"]:
            for met in ("auroc", "nll"):
                v = []
                for pr, _ in PR:
                    x = ab_[(ab_.label == lab) & (ab_.target == tg) & (ab_.protocol == pr)]
                    if not len(x):
                        v.append("--"); continue
                    s = f3(x[met].iloc[0])
                    c_ = abc[(abc.label == lab) & (abc.target == tg) & (abc.protocol == pr) & (abc.metric == met)]
                    if pr in ("regime_bias", "latent_noanchor", "same_regime") and len(c_) and c_.p_holm.iloc[0] < 0.05:
                        s += "$^{*}$"
                    v.append(s)
                    W += nm(f"Abl{'Rsix' if lab == 'r6' else 'Al'}{TK[tg]}{PK[pr]}{'Auc' if met == 'auroc' else 'Nll'}", f3(x[met].iloc[0]))
                rows.append([("$Y$" if lab == "r6" else "$Y^{*}$") if tg == CELLS5[0] and met == "auroc" else "", ENAME[tg] if met == "auroc" else "", "AUROC" if met == "auroc" else "NLL"] + v)
    t = pd.DataFrame(rows, columns=["Label", "Target", "Metric"] + [p for _, p in PR])
    tex_table(t, "tab_ablation", r"Ablation of the regime cut-point model (TF-IDF, leave-one-environment-out, mean over three seeds). $-$cuts: per-regime intercepts with binary labels; $-$anchors: anchor items removed from training; $-$other reg.: training only on environments of the target's regime. $^{*}$Holm-adjusted $p<0.05$ against the full model (paired publisher/channel-cluster bootstrap; family: the three ablations within a metric).",
              "tab:ablation", "lll" + "r" * len(PR), wide=True)
    sig = abc[abc.p_holm < 0.05]
    W += nm("AblSigN", str(len(sig))) + nm("AblFamN", str(int(abc.p_holm.notna().sum())))

# misspecified working link in the design audit
if (R / "E21S/design_misspecified_link.csv").exists():
    ds = pd.read_csv(R / "E21S/design_misspecified_link.csv"); ds = ds[ds.n == 100]
    rows = []
    for h_ in sorted(ds.h_shift.unique()):
        for dz in ("srs", "level_x_score_neyman"):
            g_ = ds[(ds.h_shift == h_) & (ds.design == dz)].set_index("estimator")
            rows.append([f"{h_:+.1f}", "SRS" if dz == "srs" else "Neyman", f"{g_.loc['diff', 'auroc_bias']:+.3f}", f3(g_.loc["diff", "auroc_rmse"]), f3(g_.loc["diff", "auroc_cover"]),
                         f3(g_.loc["sample_only", "auroc_rmse"]), f"{g_.loc['model_only', 'auroc_bias']:+.3f}", f"{g_.loc['model_only', 'nll_bias']:+.3f}"])
    t = pd.DataFrame(rows, columns=["Logit shift", "Design", "Diff. bias", "Diff. RMSE", "Diff. cover.", "Sample RMSE", "Plug-in AUROC bias", "Plug-in NLL bias"])
    tex_table(t, "tab_design_mis", r"Anchor audit with a deliberately mis-transported working link: the link fitted on the other half of the publishers is shifted on the logit scale before use. $n=100$ labelled anchors, 200 draws per scorer, mean over the anchor-blind scorers. Diff.: difference estimator (AUROC bias, RMSE and coverage of the nominal 95\% interval). Plug-in: label-free estimate from the shifted link.",
              "tab:designmis", "llrrrrrr")
    dd_ = ds[ds.estimator == "diff"]; mo = ds[ds.estimator == "model_only"]
    W += nm("DesMisDiffBiasMax", f"{dd_.auroc_bias.abs().max():.3f}") + nm("DesMisCovMin", f"{dd_.auroc_cover.min():.2f}") + nm("DesMisCovMax", f"{dd_.auroc_cover.max():.2f}")
    W += nm("DesMisPlugNllBiasMax", f"{mo.nll_bias.abs().max():.3f}") + nm("DesMisPlugAucBiasMax", f"{mo.auroc_bias.abs().max():.3f}")
    W += nm("DesMisDiffRmseMin", f"{dd_.auroc_rmse.min():.3f}") + nm("DesMisDiffRmseMax", f"{dd_.auroc_rmse.max():.3f}")

# E26: placebo null for the joint label-and-base-rate component
if (R / "E26/decomposition_placebo_summary.csv").exists():
    pl = pd.read_csv(R / "E26/decomposition_placebo_summary.csv")
    CELLN = {"banglaclick:news": "News", "banglaclick:facebook": "Facebook", "banglaclick:twitter": "Twitter", "banglaclick:youtube": "YouTube"}
    CK = {"banglaclick:news": "News", "banglaclick:facebook": "Fb", "banglaclick:twitter": "Tw", "banglaclick:youtube": "Yt"}
    rows = []
    for (env_, m_), g_ in pl.groupby(["env", "model"], sort=False):
        g_ = g_.set_index("proxy")
        rows.append([CELLN[env_], MNAME[m_], f3(g_.loc["level2", "nll_gap"])] +
                    [f"{f3(g_.loc[px, 'nll_joint'])} / {f3(g_.loc[px, 'nll_joint_placebo_mean'])} / {g_.loc[px, 'nll_excess_share']:.2f}" for px in ("level2", "trigger")])
    order = {"News": 0, "Facebook": 1, "Twitter": 2, "YouTube": 3}
    rows.sort(key=lambda r: (order[r[0]], r[1]))
    t = pd.DataFrame(rows, columns=["Cell", "Model", "Gap", r"$\Lv\geq2$: $J$ / placebo / share", r"Trigger: $J$ / placebo / share"])
    tex_table(t, "tab_placebo", r"Placebo null for the joint label-and-base-rate component $J=R_{00}-R_{AP}$ (log loss, nats; mean over seeds). Placebo: the aligned label permuted within the cell (same prevalence, no item information), 200 permutations. Share: $(J-\bar J_{\text{placebo}})/(R_{00}-R_S)$, the part of the gap attributable to the information in the aligned label.",
              "tab:placebo", "llrll")
    tx = pl[pl.model.isin(["tfidf_lr", "banglabert", "xlmr"])]
    for px_, pk_ in (("level2", "L"), ("trigger", "T")):
        for env_, ck_ in CK.items():
            v_ = tx[(tx.proxy == px_) & (tx.env == env_)]
            W += nm(f"Plc{pk_}{ck_}ShareMin", f"{v_.nll_excess_share.min():.2f}") + nm(f"Plc{pk_}{ck_}ShareMax", f"{v_.nll_excess_share.max():.2f}")
            W += nm(f"Plc{pk_}{ck_}PlShareMin", f"{v_.nll_placebo_share.min():.2f}") + nm(f"Plc{pk_}{ck_}PlShareMax", f"{v_.nll_placebo_share.max():.2f}")
    W += nm("PlcPermMax", f"{tx.nll_p_perm.max():.3f}") + nm("PlcNPerm", "200")

# E25: media attachments of BanglaClick social-media items (text-free counts)
if (R / "E25/banglaclick_media_type_counts.csv").exists():
    md_ = pd.read_csv(R / "E25/banglaclick_media_type_counts.csv")
    soc_ = md_[md_.platform == "Social Media"]
    W += nm("NMediaSocial", f"{int(soc_[soc_.media_type != '<empty>'].n.sum()):,}") + nm("NSocialItems", f"{int(soc_.n.sum()):,}")

# E30: identified sets with the news link estimated under conditional MAR (g_tilde) next to the anchor link (g_hat)
if (R / "E30/identified_sets_summary.csv").exists():
    s30 = pd.read_csv(R / "E30/identified_sets_summary.csv"); lk30 = pd.read_csv(R / "E30/news_link_by_level.csv")
    df30 = pd.read_csv(R / "E30/alm_lower_difference.csv").set_index(["scorer", "env"])
    jk30 = pd.read_csv(R / "E30/jackknife_few_cluster.csv"); jk30 = jk30[jk30.quantity == "auroc_ALM_lo"].set_index(["scorer", "env", "link"])
    SC3 = ["tfidf_lr", "banglabert", "xlmr"]; CEL = ["banglaclick:news", "banglaclick:facebook", "banglaclick:twitter", "banglaclick:youtube"]
    rows = []
    for sc_ in SC3:
        for e_ in CEL:
            a_ = s30[(s30.scorer == sc_) & (s30.env == e_)].set_index("link")
            h_, t_ = a_.loc["g_hat_anchor"], a_.loc["g_tilde_scl"]; d_ = df30.loc[(sc_, e_)]
            jk_ = (f"[{f3(jk30.loc[(sc_, e_, 'g_tilde_scl'), 'jk_lo'])}, {f3(jk30.loc[(sc_, e_, 'g_tilde_scl'), 'jk_hi'])}]"
                   if (sc_, e_, "g_tilde_scl") in jk30.index else "--")
            rows.append([MNAME[sc_], ENAME[e_], f"{f3(h_.auroc_ALM_lo)} ({f3(h_.auroc_ALM_cs_lo)})", f"{f3(t_.auroc_ALM_lo)} ({f3(t_.auroc_ALM_cs_lo)})",
                         f"{d_['mean']:+.3f} [{d_.lo:+.3f}, {d_.hi:+.3f}]", jk_, f3(t_.src_auroc)])
    t = pd.DataFrame(rows, columns=["Scorer", "Cell", r"AL+M lower, $\hat g$ (CS)", r"AL+M lower, $\tilde g$ (CS)", r"$\tilde g-\hat g$ [95\%]", r"Jackknife, $\tilde g$", "Source"])
    tex_table(t, "tab_ident_gtilde", r"AL+M lower end of the identified set of source-construct AUROC with the news level link estimated from the anchors alone ($\hat g$, Assumptions~\ref{as:rep}+\ref{as:link}) or from all news items under conditional MAR ($\tilde g$, Assumptions~\ref{as:crep}+\ref{as:link}). In parentheses: lower end of the 95\% confidence set; bracketed: paired 95\% interval of the difference (2,000 cluster-bootstrap draws, news publishers resampled jointly for anchors and non-anchor items). Jackknife: delete-one-cluster 95\% $t_{G-1}$ interval of the $\tilde g$ lower end, link held fixed, for cells with at most 26 clusters. Source: held-out source AUROC.",
              "tab:identgtilde", "llrrrrr", wide=True)
    lrows = []
    for sc_ in SC3:
        for lk_, nm_ in (("g_hat_anchor", r"$\hat g$ (anchors)"), ("g_tilde_scl", r"$\tilde g$ (all news)")):
            v_ = lk30[(lk30.scorer == sc_) & (lk30.link == lk_)].iloc[0]
            lrows.append([MNAME[sc_], nm_] + [f3(v_[f"g{l}"]) for l in range(4)])
    tex_table(pd.DataFrame(lrows, columns=["Scorer", "Link", r"$\Lv=0$", r"$\Lv=1$", r"$\Lv=2$", r"$\Lv=3$"]), "tab_news_link",
              r"News level link $P(\YA=1\mid\Lv=l)$ estimated from the anchors and from all news items under conditional MAR ($\tilde g_l$: mean over news items at level $l$ of the observed $\YA$ on anchors and of the score-conditional link elsewhere). $\hat g$ does not depend on the scorer; $\tilde g$ does, and agreement across scorers is necessary for Assumption~\ref{as:crep}.",
              "tab:newslink", "llrrrr")
    gh_ = lk30[lk30.link == "g_hat_anchor"].set_index("scorer")[[f"g{l}" for l in range(4)]]
    gt_ = lk30[lk30.link == "g_tilde_scl"].set_index("scorer")[[f"g{l}" for l in range(4)]]
    drop_ = (gh_ - gt_).to_numpy()
    W += nm("GtDropMin", f"{drop_.min():.2f}") + nm("GtDropMax", f"{drop_.max():.2f}") + nm("GtSpreadMax", f"{(gt_.max() - gt_.min()).max():.3f}")
    W += nm("GtAlmDiffMin", f"{df30['mean'].min():+.3f}") + nm("GtAlmDiffMax", f"{df30['mean'].max():+.3f}")
    jt_ = jk30.xs("g_tilde_scl", level="link").reset_index()
    for e_, k_ in (("banglaclick:facebook", "Fb"), ("banglaclick:youtube", "Yt")):
        v_ = jt_[jt_.env == e_]
        W += nm(f"GtJk{k_}LoMin", f3(v_.jk_lo.min())) + nm(f"GtJk{k_}LoMax", f3(v_.jk_lo.max()))
    tt_ = s30[s30.link == "g_tilde_scl"]
    W += nm("GtAlmLoMin", f3(tt_.auroc_ALM_lo.min())) + nm("GtAlmLoMax", f3(tt_.auroc_ALM_lo.max()))

# E31: few-cluster inference -- jackknife size and delete-one-cluster intervals for headline quantities
if (R / "E31/jackknife_intervals.csv").exists():
    sz = pd.read_csv(R / "E31/jackknife_size.csv")
    rows = []
    for (G_, dom_), g_ in sz.groupby(["G", "dominant_cluster"]):
        g_ = g_.set_index("method")
        rows.append([str(G_), "yes" if dom_ else "no"] + [f"{g_.loc[m_, 'size_at_5pct']:.3f} ({g_.loc[m_, 'mc_se']:.3f})" for m_ in ("pairs_bootstrap", "jackknife_t")])
    tex_table(pd.DataFrame(rows, columns=["Clusters", "One cluster = half", "Pairs bootstrap", r"Jackknife $t_{G-1}$"]), "tab_jk_size",
              r"Empirical size (Monte Carlo SE) of nominal 5\% tests of an AUROC contrast between two exchangeable scorers (true contrast 0) in simulated clustered data with cluster-varying signal; 500 replications, 399 bootstrap draws. With one dominant cluster, that cluster holds half of the items, as the largest BanglaClick YouTube channel.",
              "tab:jksize", "llrr")
    jk = pd.read_csv(R / "E31/jackknife_intervals.csv")
    ci4 = {"hard_aligned": pd.read_csv(R / "E4/decomposition_cluster_bootstrap_ci.csv").set_index(["env", "model"]),
           "trigger": pd.read_csv(R / "E4T/decomposition_cluster_bootstrap_ci.csv").set_index(["env", "model"])}
    ci19 = pd.read_csv(R / "E19/application_summary.csv").set_index(["scorer", "env", "label"])
    QN = {"nll_construct": r"$\phi_A$", "nll_residual": r"$\rho$", "auroc": "AUROC"}
    PX = {"hard_aligned": r"$\Lv\geq2$", "trigger": "trigger", "scl_g1": r"SCL $\gamma=1$"}
    rows = []
    for e_ in ("banglaclick:facebook", "banglaclick:youtube"):
        for px_ in ("hard_aligned", "trigger"):
            for m_ in ("tfidf_lr", "banglabert", "xlmr"):
                for q_ in ("nll_construct", "nll_residual"):
                    v_ = jk[(jk.analysis == "decomposition_E4") & (jk.env == e_) & (jk.model == m_) & (jk.label == px_) & (jk.quantity == q_)]
                    if not len(v_):
                        continue
                    v_ = v_.iloc[0]; b_ = ci4[px_].loc[(e_, m_)]
                    rows.append([ENAME[e_], PX[px_], MNAME[m_], QN[q_], f3(v_.estimate), f"[{f3(b_[q_ + '_lo'])}, {f3(b_[q_ + '_hi'])}]",
                                 f"[{f3(v_.jk_lo)}, {f3(v_.jk_hi)}]", f"{f3(v_.loco_min)}--{f3(v_.loco_max)}"])
        for m_ in ("tfidf_lr", "banglabert", "xlmr"):
            v_ = jk[(jk.analysis == "scl_application_E19") & (jk.env == e_) & (jk.model == m_) & (jk.label == "scl_g1") & (jk.quantity == "auroc")].iloc[0]
            b_ = ci19.loc[(m_, e_, "scl_g1")]
            rows.append([ENAME[e_], PX["scl_g1"], MNAME[m_], QN["auroc"], f3(v_.estimate), f"[{f3(b_.auroc_lo)}, {f3(b_.auroc_hi)}]",
                         f"[{f3(v_.jk_lo)}, {f3(v_.jk_hi)}]", f"{f3(v_.loco_min)}--{f3(v_.loco_max)}"])
    tex_table(pd.DataFrame(rows, columns=["Cell", "Proxy", "Model", "Quantity", "Estimate", "Pairs 95\\%", "Jackknife 95\\%", "LOCO range"]), "tab_jk_cells",
              r"Headline quantities in the cells with few clusters (Facebook 26 pages, YouTube 10 channels): decomposition components (log loss, nats; frozen source models, mean over seeds) and source-construct AUROC under the score-conditional link. Pairs: percentile cluster bootstrap as in the main tables; jackknife: delete-one-cluster $t_{G-1}$ interval; LOCO range: minimum and maximum of the estimate with one cluster removed. For SCL the link is refitted after each deletion.",
              "tab:jkcells", "llllrrrr", wide=True)
    dz = jk[(jk.analysis == "decomposition_E4") & jk.quantity.isin(["nll_construct", "nll_residual"]) & jk.model.isin(["tfidf_lr", "banglabert", "xlmr"])]
    yt_ = dz[dz.env == "banglaclick:youtube"]; fb_ = dz[dz.env == "banglaclick:facebook"]
    inc = lambda d: int(((d.jk_lo <= 0) & (d.jk_hi >= 0)).sum())
    W += nm("JkYtZeroN", str(inc(yt_))) + nm("JkYtN", str(len(yt_))) + nm("JkFbZeroN", str(inc(fb_))) + nm("JkFbN", str(len(fb_)))
    fbc_ = fb_[(fb_.quantity == "nll_construct") & (fb_.label == "hard_aligned")]
    W += nm("JkFbConLoMin", f3(fbc_.jk_lo.min()))
    ebm_ = jk[(jk.analysis == "decomposition_E4") & (jk.env == "banglaclick:youtube") & (jk.model == "ebm") & (jk.label == "trigger") & (jk.quantity == "nll_residual")].iloc[0]
    W += nm("JkYtEbmTrResLo", f3(ebm_.jk_lo)) + nm("JkYtEbmTrResHi", f3(ebm_.jk_hi))
    for G_, k_ in ((10, "Ten"), (13, "Thirteen"), (26, "TwentySix")):
        for dom_, d_ in ((False, ""), (True, "Dom")):
            g_ = sz[(sz.G == G_) & (sz.dominant_cluster == dom_)].set_index("method").size_at_5pct
            W += nm(f"JkSize{k_}{d_}", f"{g_['jackknife_t']:.3f}") + nm(f"PairsAucSize{k_}{d_}", f"{g_['pairs_bootstrap']:.3f}")

# E32: prior-corrected pooled ERM (deployable MLLS and oracle-prevalence references)
if (R / "E32/prior_corrected_summary.csv").exists():
    s32 = pd.read_csv(R / "E32/prior_corrected_summary.csv"); c32 = pd.read_csv(R / "E32/contrasts_vs_prior_corrected_pooled.csv")
    TG = {"banglaclick:news": "News (B)", "banglaclick:facebook": "Facebook (B)", "banglaclick:twitter": "Twitter (B)",
          "banglaclick:youtube": "YouTube (B)", "baitbuster:youtube": "YouTube (C)"}
    rows = []
    for m_ in ("tfidf_lr", "banglabert", "xlmr"):
        for lab_, ln_ in (("r6", "published"), ("aligned", "aligned")):
            for t_, tn_ in TG.items():
                v_ = s32[(s32.model == m_) & (s32.label == lab_) & (s32.target == t_)].set_index("protocol")
                rows.append([MNAME[m_], ln_, tn_, f"{v_.loc['pooled_erm', 'prevalence']:.2f}", f"{v_.loc['pooled_erm_mlls', 'pi_mlls']:.2f}"]
                            + [f3(v_.loc[p_, "nll"]) for p_ in ("pooled_erm", "pooled_erm_mlls", "pooled_erm_oracle", "latent_cuts", "group_dro")])
    tex_table(pd.DataFrame(rows, columns=["Model", "Label", "Target", r"$\pi_e$", r"$\hat\pi_e^{\text{MLLS}}$", "Pooled", "+MLLS", "+oracle $\pi_e$", "Cut-point", "Group DRO"]),
              "tab_prior_pooled", r"Prior-corrected pooled ERM on the LOEO targets: log loss (nats, mean over seeds). $\pi_e$: true target prevalence; $\hat\pi_e^{\text{MLLS}}$: MLLS estimate from unlabelled target scores, used by the deployable correction (+MLLS). +oracle $\pi_e$ moves the pooled model to the true target prevalence and therefore uses target labels; it is the best that any prior correction can do, not a deployable baseline. AUROC is unchanged by prior correction.",
              "tab:priorpooled", "lllrrrrrrr", wide=True)
    def cnt(ref, pr, m):
        y_ = c32[(c32.reference == ref) & (c32.protocol == pr) & (c32.metric == m)]
        good = (y_["diff"] > 0) if m == "auroc" else (y_["diff"] < 0)
        return len(y_), int(((y_.p_holm < .05) & good).sum()), int(((y_.p_holm < .05) & ~good).sum())
    for ref_, rk_ in (("pooled_erm_mlls", "Mlls"), ("pooled_erm_oracle", "Orc")):
        for pr_, pk_ in (("latent_cuts", "Lat"), ("group_dro", "Dro"), ("pooled_erm", "Pool")):
            for m_, mk_ in (("nll", "Nll"), ("auroc", "Auc")):
                n_, b_, w_ = cnt(ref_, pr_, m_)
                W += nm(f"Pc{rk_}{pk_}{mk_}N", str(n_)) + nm(f"Pc{rk_}{pk_}{mk_}Better", str(b_)) + nm(f"Pc{rk_}{pk_}{mk_}Worse", str(w_))
    ml_ = s32[s32.protocol == "pooled_erm_mlls"]
    W += nm("PcMllsWrongDir", str(int((np.sign(ml_.pi_mlls - ml_.pi_train) != np.sign(ml_.prevalence - ml_.pi_train)).sum()))) + nm("PcMllsCells", str(len(ml_)))
    tw_ = s32[(s32.target == "banglaclick:twitter") & (s32.label == "aligned") & (s32.model == "banglabert")].set_index("protocol")
    W += nm("PcTwAlBbertPool", f3(tw_.loc["pooled_erm", "nll"])) + nm("PcTwAlBbertMlls", f3(tw_.loc["pooled_erm_mlls", "nll"]))
    nw_ = s32[(s32.target == "banglaclick:news") & (s32.label == "r6")]
    W += nm("PcNewsRsixPiTrue", f"{nw_.prevalence.mean():.2f}") + nm("PcNewsRsixPiMllsMin", f"{nw_[nw_.protocol == 'pooled_erm_mlls'].pi_mlls.min():.2f}") + nm("PcNewsRsixPiMllsMax", f"{nw_[nw_.protocol == 'pooled_erm_mlls'].pi_mlls.max():.2f}")

with open(PAPER / "numbers.tex", "a", encoding="utf-8") as fh:
    fh.write(W)
print("S5 assets written;", W.count("newcommand"), "macros")
