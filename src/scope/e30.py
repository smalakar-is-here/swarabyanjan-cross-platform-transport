"""E30: identified sets with the news level link estimated under conditional MAR (review item R1-W2).

E20 feeds the identified sets with g_hat(l), the level frequencies of Y_A among the anchors. That estimates
P(Y_A = 1 | L = l, anchor) and equals the news link g_news(l) = P(Y_A = 1 | L = l, news) only under anchor
representativeness (Assumption 1), which the paper reports as violated (anchors have higher scores within
levels). Under Assumption 3 (Y_A independent of anchor membership given (L, f)), the news link is identified by
    g_tilde(l) = mean over all news items i with L_i = l of q_i,
    q_i = Y_A,i on anchors, h_hat(l, f_i) elsewhere   (h_hat: the flexible score-conditional link of E19).
The sets then rest on Assumptions 2 + 3 (+ M) instead of 1 + 2 (+ M). g_tilde is estimated with each scorer's
score; if Assumption 3 holds for every scorer, the three estimates target the same g_news, so their spread
across scorers is a free diagnostic.

Uncertainty. (i) Joint cluster bootstrap, 2,000 draws: news publisher clusters are resampled once and shared by
the anchors and the non-anchor news items (both enter g_tilde), and every target cell resamples its own
clusters; seed-averaged; conservative confidence set [2.5th pct of the lower end, 97.5th pct of the upper end].
(ii) For target cells with at most 26 clusters, the delete-one-cluster jackknife of each endpoint with the link
held fixed (scope.jackknife), because E13 shows the pairs bootstrap over-rejects there.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import bounds as B, construct, linking, e19, jackknife
from . import metrics as M

NEWS = "banglaclick:news"
FEW = 26
ENDPOINTS = ("auroc_AL_lo", "auroc_AL_hi", "auroc_ALM_lo", "auroc_ALM_hi", "auroc_const", "nll_ALM_lo", "nll_ALM_hi",
             "brier_ALM_lo", "brier_ALM_hi", "prevalence_AL")


def g_hat(d):
    return construct.linking_function(e19._lvframe(d))


def g_tilde(d, news):
    lk = linking.fit(d.L, d.z, d.ya, 1.0, flex=True)
    q = linking.cell_labels(lk, news.L.to_numpy(), news.z.to_numpy(), news.ya_obs.to_numpy())
    L = news.L.to_numpy()
    return np.array([q[L == l].mean() if np.any(L == l) else np.nan for l in range(4)])


def prepare(scores, pool, anc, lv, src_y):
    pool_i = pool.set_index("row_id"); aset = set(anc.normalized_title_sha256); a_idx = anc.set_index("normalized_title_sha256")
    per = {}
    for s in sorted(scores.seed.unique()):
        sc = scores[scores.seed == s]; cells = {}
        for e in e19.B_CELLS:
            c = sc[sc.env == e].copy()
            c["L"] = lv.loc[c.row_id].to_numpy().astype(int); c["z"] = linking.logit(c.p)
            c["cluster"] = pool_i.loc[c.row_id, "cluster_id"].to_numpy()
            h = pool_i.loc[c.row_id, "normalized_title_sha256"]
            c["ya_obs"] = np.where(h.isin(aset), a_idx.reindex(h).bb_label.to_numpy(), np.nan).astype(float)
            cells[e] = c.reset_index(drop=True)
        src = sc[sc.env == "banglabait:news"]
        rs = {m: f(src_y.loc[src.row_id].to_numpy(), src.p.to_numpy()) for m, f in (("auroc", M.auroc), ("nll", M.nll), ("brier", M.brier))}
        per[s] = (e19.anchor_items(sc, pool, anc, lv), cells, rs)
    return per


def links(d, cells):
    return {"g_hat_anchor": g_hat(d), "g_tilde_scl": g_tilde(d, cells[NEWS])}


def evaluate(per, fixed_links=None, only=None):
    out = []
    for s, (d, cells, rs) in per.items():
        lk = fixed_links[s] if fixed_links else links(d, cells)
        for e, c in cells.items():
            if only and e != only:
                continue
            for name, g in lk.items():
                out.append({"seed": s, "env": e, "link": name, **B.identified_sets(c.L.to_numpy(), c.p.to_numpy(), g),
                            **{f"src_{m}": v for m, v in rs.items()}, **{f"g{l}": g[l] for l in range(4)}})
    return pd.DataFrame(out)


def run(scorers, pool, anc, lv, src_y, rng, boot=2000):
    pts, bts, jks = [], [], []
    for name, scores in scorers.items():
        per = prepare(scores, pool, anc, lv, src_y)
        pt = evaluate(per); pt["scorer"] = name; pts.append(pt)
        s0 = next(iter(per)); d0, c0, _ = per[s0]
        news_cl = c0[NEWS].cluster.unique()
        other = {e: c0[e].cluster.unique() for e in c0 if e != NEWS}
        for b in range(boot):
            npick = rng.choice(news_cl, len(news_cl)); opick = {e: rng.choice(v, len(v)) for e, v in other.items()}
            sel = {}
            for s, (d, cells, rs) in per.items():
                gi = d.groupby("cluster_id").indices
                dd = d.iloc[np.concatenate([gi[k] for k in npick if k in gi])]
                cs = {}
                for e, c in cells.items():
                    ci = c.groupby("cluster").indices
                    cs[e] = c.iloc[np.concatenate([ci[k] for k in (npick if e == NEWS else opick[e])])]
                sel[s] = (dd, cs, rs)
            r = evaluate(sel).groupby(["env", "link"]).mean(numeric_only=True).reset_index(); r["draw"] = b; r["scorer"] = name
            bts.append(r)
        # delete-one-cluster jackknife for few-cluster target cells, links fixed at the full-sample estimates
        fixed = {s: links(d, cells) for s, (d, cells, rs) in per.items()}
        for e, cl in other.items():
            if len(cl) > FEW:
                continue
            full = evaluate(per, fixed, e).groupby("link").mean(numeric_only=True)
            loo = {k: [] for k in full.index}
            for k in cl:
                red = {s: (d, {**cells, e: cells[e][cells[e].cluster != k]}, rs) for s, (d, cells, rs) in per.items()}
                r = evaluate(red, fixed, e).groupby("link").mean(numeric_only=True)
                for lk in r.index:
                    loo[lk].append(r.loc[lk].to_dict())
            for lk in full.index:
                for row in jackknife.summarize(full.loc[lk].to_dict(), loo[lk], len(cl), list(ENDPOINTS)):
                    jks.append({"scorer": name, "env": e, "link": lk, **row})
    return pd.concat(pts, ignore_index=True), pd.concat(bts, ignore_index=True), pd.DataFrame(jks)
