"""E20: sharp partial-identification bounds for source-construct risk on regime-B cells (scope.bounds).

Replaces the gamma sensitivity range of E19 for cells without anchors by identified sets under stated,
nested assumptions (A0 none; AL level-link transport; AL+M within-level monotonicity), with
  M-CHECK  evidence for M on the news anchors (within-level score signal about Y_A, cluster bootstrap);
  VALID    5-fold publisher-cluster cross-fit on anchors: does the AL / ALM set computed from the training
           folds' link contain the held-out anchors' true risk? (a direct test of AL transport across clusters);
  APP      identified sets per cell, seed-averaged, with a joint cluster bootstrap (anchor clusters for the link,
           target clusters for the cell); conservative 95% confidence sets for the identified set take the
           2.5th percentile of the lower and the 97.5th of the upper endpoint;
  BREAK    relaxation of AL by delta on the logit of g(l) and the breakdown delta* at which the ALM set first
           reaches the source AUROC.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import bounds as B, construct, linking, e19
from . import metrics as M

METRICS = ("auroc", "nll", "brier")
DELTAS = (0.0, 0.25, 0.5, 1.0)


def _g(d):
    return construct.linking_function(e19._lvframe(d))


def m_check(d, rng, reps=200):
    """Within-level evidence for M: logistic slope of Y_A on the logit score with level intercepts, and the
    within-level AUROC of the score for Y_A pooled over levels (0.5 = no signal), cluster bootstrap CIs."""
    def stats(x):
        b = linking.fit(x.L, x.z, x.ya, 1.0)["beta_hat"]
        num = den = 0.0
        for l in range(4):
            s = x[x.L == l]
            if s.ya.nunique() == 2:
                w = s.ya.sum() * (1 - s.ya).sum(); num += M.auroc(s.ya.to_numpy(), s.z.to_numpy()) * w; den += w
        return b, (num / den if den else np.nan)
    b0, a0 = stats(d); gi = d.groupby("cluster_id").indices; cl = np.array(list(gi))
    bs = np.array([stats(d.iloc[np.concatenate([gi[c] for c in rng.choice(cl, len(cl))])]) for _ in range(reps)])
    return {"beta_hat": b0, "beta_lo": np.nanpercentile(bs[:, 0], 2.5), "beta_hi": np.nanpercentile(bs[:, 0], 97.5),
            "within_level_auroc": a0, "wl_auroc_lo": np.nanpercentile(bs[:, 1], 2.5), "wl_auroc_hi": np.nanpercentile(bs[:, 1], 97.5),
            "n_anchors": len(d), "n_clusters": len(cl)}


def validate(d, rng, reps=20, folds=5):
    rows = []; cl = d.cluster_id.unique()
    for r in range(reps):
        fd = d.cluster_id.map(dict(zip(cl, rng.permutation(len(cl)) % folds))).to_numpy()
        for k in range(folds):
            tr, te = d[fd != k], d[fd == k]
            if te.ya.nunique() < 2:
                continue
            s = B.identified_sets(te.L.to_numpy(), te.p.to_numpy(), _g(tr))
            for m, f in (("auroc", M.auroc), ("nll", M.nll), ("brier", M.brier)):
                t = f(te.ya.to_numpy(), te.p.to_numpy())
                rows.append({"rep": r, "fold": k, "metric": m, "truth": t, "n": len(te),
                             "in_AL": s[f"{m}_AL_lo"] <= t <= s[f"{m}_AL_hi"], "in_ALM": s[f"{m}_ALM_lo"] <= t <= s[f"{m}_ALM_hi"],
                             "width_AL": s[f"{m}_AL_hi"] - s[f"{m}_AL_lo"], "width_ALM": s[f"{m}_ALM_hi"] - s[f"{m}_ALM_lo"],
                             "dist_ALM": max(s[f"{m}_ALM_lo"] - t, t - s[f"{m}_ALM_hi"], 0.0),
                             "truth_minus_const": t - s[f"{m}_const"]})
    return pd.DataFrame(rows)


def _cells(sc, pool_i, lv, cells):
    out = {}
    for e in cells:
        c = sc[sc.env == e].copy()
        c["L"] = lv.loc[c.row_id].to_numpy().astype(int); c["cluster"] = pool_i.loc[c.row_id, "cluster_id"].to_numpy()
        out[e] = c.reset_index(drop=True)
    return out


def application(scores, pool, anc, lv, src_y, rng, boot):
    pool_i = pool.set_index("row_id"); seeds = sorted(scores.seed.unique()); per = {}
    for s in seeds:
        sc = scores[scores.seed == s]; src = sc[sc.env == "banglabait:news"]
        per[s] = (e19.anchor_items(sc, pool, anc, lv), _cells(sc, pool_i, lv, e19.B_CELLS),
                  {m: f(src_y.loc[src.row_id].to_numpy(), src.p.to_numpy()) for m, f in (("auroc", M.auroc), ("nll", M.nll), ("brier", M.brier))})

    def evaluate(sel):
        out = []
        for s, (d, cells, rs) in sel.items():
            g = _g(d)
            for e, c in cells.items():
                out.append({"seed": s, "env": e, **B.identified_sets(c.L.to_numpy(), c.p.to_numpy(), g),
                            **{f"src_{m}": rs[m] for m in rs}, **{f"g{l}": g[l] for l in range(4)}})
        return pd.DataFrame(out)

    point = evaluate(per); boots = []
    d0, c0, _ = per[seeds[0]]
    acl = d0.cluster_id.unique()
    for b in range(boot):
        apick = rng.choice(acl, len(acl)); cpick = {e: rng.choice(c0[e].cluster.unique(), c0[e].cluster.nunique()) for e in c0}
        sel = {}
        for s, (d, cells, rs) in per.items():
            gi = d.groupby("cluster_id").indices
            cs = {e: c.iloc[np.concatenate([c.groupby("cluster").indices[k] for k in cpick[e]])] for e, c in cells.items()}
            sel[s] = (d.iloc[np.concatenate([gi[k] for k in apick])], cs, rs)
        r = evaluate(sel).groupby("env").mean(numeric_only=True).reset_index(); r["draw"] = b; boots.append(r)

    # relaxation and breakdown (point estimates, per seed, then averaged)
    rel = []
    for s, (d, cells, rs) in per.items():
        g = _g(d)
        for e, c in cells.items():
            L, p = c.L.to_numpy(), c.p.to_numpy()
            for dl in DELTAS:
                for m in METRICS:
                    rel.append({"seed": s, "env": e, "delta": dl, "metric": m, **B.relaxed_bounds(L, p, g, dl, m)})
            sa = B.identified_sets(L, p, g)
            side = "below" if sa["auroc_ALM_hi"] < rs["auroc"] else ("above" if sa["auroc_ALM_lo"] > rs["auroc"] else None)
            rel.append({"seed": s, "env": e, "delta": np.nan, "metric": "auroc_breakdown_vs_source", "side": side or "contains",
                        "delta_star": B.breakdown_delta(L, p, g, "auroc", rs["auroc"], side) if side else 0.0})
    return point, pd.concat(boots, ignore_index=True), pd.DataFrame(rel)


def run(scorers, pool, anc, lv, src_y, rng, boot=300):
    mc, va, pts, bts, rels = [], [], [], [], []
    for name, s in scorers.items():
        full = s.env.eq("banglabait:news").any() and all(s.env.eq(e).any() for e in e19.B_CELLS)
        for sd in sorted(s.seed.unique()):
            d = e19.anchor_items(s[s.seed == sd], pool, anc, lv)
            mc.append({"scorer": name, "seed": sd, **m_check(d, rng)})
            v = validate(d, rng); v["scorer"], v["seed"] = name, sd; va.append(v)
        if full and src_y is not None:
            pt, bt, rl = application(s, pool, anc, lv, src_y, rng, boot)
            for f in (pt, bt, rl):
                f["scorer"] = name
            pts.append(pt); bts.append(bt); rels.append(rl)
    return pd.DataFrame(mc), pd.concat(va), pd.concat(pts), pd.concat(bts), pd.concat(rels)
