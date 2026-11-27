"""E24: reviewer-requested robustness analyses (round 1 of the internal review; see paper reviews/round-1-meta.md).

Every analysis reuses existing frozen predictions, anchors and rubric components; no model is retrained.

  binarisations      M-06  which binary cut of the BanglaClick rubric best matches the source label on anchors?
                           Level, trigger, quality-failure and combined cuts; kappa, PABAK, kappa_max given the
                           marginals; cut selected by kappa with 5-fold publisher cross-fitting (20 repeats).
  component_link     M-02  partial test of Assumption 2 (level-link transport). A model of Y_A given level AND rubric
                           components is fitted on anchors and pushed through each platform's component mix, giving
                           g_e(l) and the implied logit misspecification delta_e = max_l |logit g_e(l) - logit g(l)|.
                           Valid only if Y_A given (level, components) transports, a weaker assumption than AL.
  youtube_no_top     M-10  BanglaClick YouTube with its dominant channel removed.
  metric_suite       M-14  AUPRC, balanced accuracy, macro-F1, ECE, Brier of the frozen models per cell and label.
  jackknife          M-08  publisher-cluster jackknife t-test (G-1 df) for every protocol contrast vs source-only, as
                           a check on the pairs cluster bootstrap that over-rejects at few clusters.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from . import metrics as M

FAILS = ("f", "a", "s", "t")  # factual, attribution, specificity, tone (BanglaClick FAST criteria)


def _fail_cols(df, col="fast_fail"):
    s = df[col].fillna("").astype(str)
    return pd.DataFrame({f"fail_{c}": s.str.split(",").apply(lambda v, c=c: float(c in v)) for c in FAILS}, index=df.index)


def kappa_stats(y, x):
    y = np.asarray(y, float); x = np.asarray(x, float); po = np.mean(y == x)
    py, px = y.mean(), x.mean(); pe = py * px + (1 - py) * (1 - px)
    k = (po - pe) / (1 - pe) if pe < 1 else np.nan
    pmax = min(py, px) + min(1 - py, 1 - px); kmax = (pmax - pe) / (1 - pe) if pe < 1 else np.nan
    return {"kappa": k, "pabak": 2 * po - 1, "kappa_max": kmax, "disagreement": 1 - po, "prevalence": px, "prev_source": py}


def candidate_cuts(df, level="bc_level"):
    L = df[level].astype(int).to_numpy(); T = df.trigger_level.fillna(0).astype(int).to_numpy()
    F = df.fast_level.fillna(0).astype(int).to_numpy(); fc = _fail_cols(df)
    c = {f"level>={k}": L >= k for k in (1, 2, 3)}
    c.update({f"trigger>={k}": T >= k for k in (1, 2)})
    c.update({f"quality_fails>={k}": F >= k for k in (1, 2, 3)})
    c["trigger>=1 & level>=1"] = (T >= 1) & (L >= 1)
    c["trigger>=1 | level>=2"] = (T >= 1) | (L >= 2)
    c["level>=1 & not(attribution only)"] = (L >= 1) & ~((fc.fail_a.to_numpy() == 1) & (fc[["fail_f", "fail_s", "fail_t"]].sum(1).to_numpy() == 0) & (T == 0))
    return {k: v.astype(float) for k, v in c.items()}


def binarisations(a, rng, reps=20, folds=5):
    y = a.bb_label.to_numpy().astype(float); cand = candidate_cuts(a)
    full = pd.DataFrame([{"cut": k, **kappa_stats(y, v)} for k, v in cand.items()])
    cl = a.bc_cluster.unique(); sel, cf = [], []
    for r in range(reps):
        fmap = dict(zip(cl, rng.permutation(len(cl)) % folds)); fd = a.bc_cluster.map(fmap).to_numpy()
        yy, xx = [], []
        for f in range(folds):
            tr, te = fd != f, fd == f
            best = max(cand, key=lambda k: kappa_stats(y[tr], cand[k][tr])["kappa"])
            sel.append(best); yy.append(y[te]); xx.append(cand[best][te])
        cf.append(kappa_stats(np.concatenate(yy), np.concatenate(xx))["kappa"])
    sel = pd.Series(sel).value_counts(normalize=True).rename("selection_share").reset_index().rename(columns={"index": "cut"})
    if "cut" not in sel.columns:
        sel.columns = ["cut", "selection_share"]
    return full.merge(sel, on="cut", how="left").fillna({"selection_share": 0.0}), {
        "crossfit_kappa_mean": float(np.mean(cf)), "crossfit_kappa_lo": float(np.percentile(cf, 2.5)),
        "crossfit_kappa_hi": float(np.percentile(cf, 97.5)), "reps": reps, "folds": folds}


def _design(df, level):
    X = pd.get_dummies(df[level].astype(int), prefix="L").reindex(columns=[f"L_{k}" for k in (1, 2, 3)], fill_value=0).astype(float)
    X = pd.concat([X, _fail_cols(df)], axis=1)
    X = pd.concat([X, pd.get_dummies(df.trigger_level.fillna(0).astype(int).clip(0, 2), prefix="T").reindex(columns=["T_1", "T_2"], fill_value=0).astype(float)], axis=1)
    return X.to_numpy()


def component_link(a, cells, rng, boot=500):
    """a: anchors (bb_label, bc_level, components, bc_cluster); cells: {env: frame with level and components}."""
    def once(an):
        m = LogisticRegression(C=1e4, max_iter=2000).fit(_design(an, "bc_level"), an.bb_label.astype(int))
        g = an.groupby(an.bc_level.astype(int)).bb_label.mean().reindex(range(4))
        out = {}
        for e, c in cells.items():
            ph = m.predict_proba(_design(c, "level"))[:, 1]
            ge = pd.Series(ph).groupby(c.level.astype(int).to_numpy()).mean().reindex(range(4))
            d = (logit(ge.clip(1e-3, 1 - 1e-3)) - logit(g.clip(1e-3, 1 - 1e-3))).abs()
            out[e] = {**{f"g_e_{l}": ge[l] for l in range(4)}, **{f"g_news_{l}": g[l] for l in range(4)},
                      **{f"dlogit_{l}": d[l] for l in range(4)}, "delta_implied": float(d[[0, 1, 2]].max()),
                      "delta_implied_all_levels": float(d.max())}
        return out
    pt = once(a); cl = a.bc_cluster.unique(); gi = a.groupby("bc_cluster").indices; bs = []
    for b in range(boot):
        pick = rng.choice(cl, len(cl)); an = a.iloc[np.concatenate([gi[k] for k in pick])]
        if an.bb_label.nunique() < 2 or an.bc_level.nunique() < 4:
            continue
        for e, v in once(an).items():
            bs.append({"env": e, "draw": b, **v})
    bs = pd.DataFrame(bs)
    rows = []
    for e, v in pt.items():
        r = {"env": e, **v}
        g = bs[bs.env == e]
        for k in ("delta_implied", "delta_implied_all_levels", "dlogit_0", "dlogit_1", "dlogit_2", "dlogit_3"):
            r[k + "_lo"], r[k + "_hi"] = float(np.nanpercentile(g[k], 2.5)), float(np.nanpercentile(g[k], 97.5))
        rows.append(r)
    return pd.DataFrame(rows)


def component_mix(cells):
    rows = []
    for e, c in cells.items():
        fc = _fail_cols(c)
        for l, g in c.groupby(c.level.astype(int)):
            f = fc.loc[g.index]
            rows.append({"env": e, "level": l, "n": len(g), **{k: float(f[k].mean()) for k in f.columns},
                         "trigger_ge1": float((g.trigger_level.fillna(0) >= 1).mean())})
    return pd.DataFrame(rows)


def jackknife_contrast(y, clusters, pa, pb, metric):
    """Leave-one-cluster-out jackknife of metric(pb) - metric(pa); t-test with G-1 df."""
    f = {"auroc": M.auroc, "nll": M.nll}[metric]
    uc, inv = np.unique(clusters, return_inverse=True); G = len(uc)
    d0 = f(y, pb) - f(y, pa); reps = []
    for g in range(G):
        k = inv != g
        if len(np.unique(y[k])) < 2:
            continue
        reps.append(f(y[k], pb[k]) - f(y[k], pa[k]))
    reps = np.array(reps); Gj = len(reps)
    se = np.sqrt((Gj - 1) / Gj * np.sum((reps - reps.mean()) ** 2)) if Gj > 2 else np.nan
    t = d0 / se if se and se > 0 else np.nan
    return d0, se, (2 * stats.t.sf(abs(t), Gj - 1) if np.isfinite(t) else np.nan), G
