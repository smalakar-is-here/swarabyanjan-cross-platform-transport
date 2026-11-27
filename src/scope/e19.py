"""E19: score-conditional anchor linking (scope.linking) on real data, with validation where truth is known.

Scorer. An anchor-blind source-only model: trained on BanglaBait train with every title of the
BanglaClick-news cell removed (the only target cell that shares titles with the source), so it has seen
no anchor and no regime-B or regime-C item. TF-IDF here (identical to the E10 source-only model for the
news target; checked); transformer scorers are added from the GPU units' xcell_ outputs when present.

Parts.
  V1  Cross-fitted validation on anchors (Y_A observed): fit each proxy on 4/5 of anchor publisher
      clusters, label the held-out fifth, compare its estimated risk with the true risk. 20 fold draws.
  V2  Selection stress test on anchors with known truth: a pseudo-anchor subset is drawn with probability
      increasing in the model score (the documented Assumption-1 violation, under which Assumption 1'
      holds) or additionally in Y_A (violates Assumption 1'); proxies are fitted on the subset and scored
      on the rest. 200 draws per setting.
  APP Source-construct risk of the scorer on every regime-B cell over gamma in {0, .25, .5, .75, 1}, with a
      joint cluster bootstrap (anchor publisher clusters for the link, target clusters for the cell).
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import GroupShuffleSplit
from . import paths, data, construct, linking, priors
from . import metrics as M

GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0]
B_CELLS = ["banglaclick:news", "banglaclick:facebook", "banglaclick:youtube", "banglaclick:twitter"]


def blind_tfidf(pool, src_test, seeds):
    """Anchor-blind source-only TF-IDF scorer; returns long frame row_id, env, seed, p, z_raw."""
    from .train_cpu import featurizer, fit_lr, fit_temperature
    news = pool[pool.env == "banglaclick:news"]
    train = pool[pool.env == "banglabait:news"]
    train = train[~train.normalized_title_sha256.isin(news.normalized_title_sha256)]
    score = pd.concat([pool[pool.env != "banglabait:news"][["row_id", "env", "title"]],
                       src_test[["row_id", "env", "title"]]], ignore_index=True)
    out = []
    for seed in seeds:
        fit_i, val_i = next(GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed).split(train, groups=train.cluster_id))
        tr, va = train.iloc[fit_i], train.iloc[val_i]
        feats = featurizer(); Xtr = feats.fit_transform(tr.title)
        clf = fit_lr(Xtr, tr.y_r6.to_numpy(), seed=seed)
        T = fit_temperature(clf.decision_function(feats.transform(va.title)), va.y_r6.to_numpy())
        p = expit(clf.decision_function(feats.transform(score.title)) / T)
        out.append(pd.DataFrame({"row_id": score.row_id.to_numpy(), "env": score.env.to_numpy(), "seed": seed, "p": p,
                                 "pi_train": float(tr.y_r6.mean())}))
    return pd.concat(out, ignore_index=True)


def proxies_fit(tr_L, tr_z, tr_ya, tr_level_frame):
    """Fit every proxy on a training subset of anchors. tr_level_frame has bc_level, bb_label (for g, t*)."""
    g = construct.linking_function(tr_level_frame)
    t_star = max((1, 2, 3), key=lambda t: M.kappa(tr_level_frame.bb_label, (tr_level_frame.bc_level >= t).astype(int)))
    full = linking.fit(tr_L, tr_z, tr_ya, gamma=1.0)
    level_only = linking.fit(tr_L, tr_z, tr_ya, gamma=0.0, beta=full["score"])
    flex = linking.fit(tr_L, tr_z, tr_ya, gamma=1.0, flex=True)
    return {"g": g, "t_star": t_star, "scl": full, "scl0": level_only, "sclf": flex}


def proxies_label(fitted, L, z):
    return {"r6": (L >= 1).astype(float), "hard_aligned": (L >= fitted["t_star"]).astype(float),
            "soft_isotonic": fitted["g"][L], "scl_gamma0": linking.predict(fitted["scl0"], L, z),
            "scl_gamma1": linking.predict(fitted["scl"], L, z), "scl_flex": linking.predict(fitted["sclf"], L, z)}


def anchor_items(scores, pool, anc, lv):
    """Anchor items of the news cell for one scorer seed: L, z, Y_A, publisher cluster."""
    a = anc.set_index("normalized_title_sha256")
    d = scores[scores.env == "banglaclick:news"].merge(pool[["row_id", "normalized_title_sha256", "cluster_id"]], on="row_id")
    d = d[d.normalized_title_sha256.isin(a.index)].copy()
    d["L"] = lv.loc[d.row_id].to_numpy().astype(int); d["ya"] = a.loc[d.normalized_title_sha256, "bb_label"].to_numpy().astype(float)
    d["z"] = linking.logit(d.p)
    return d.reset_index(drop=True)


def observed_gap(scores, pool, anc, lv):
    """Mean over levels of (anchor - non-anchor) mean logit within level, in SDs of the news-cell logit."""
    nw = scores[scores.env == "banglaclick:news"].merge(pool[["row_id", "normalized_title_sha256"]], on="row_id")
    nw["L"] = lv.loc[nw.row_id].to_numpy(); nw["z"] = linking.logit(nw.p); nw["a"] = nw.normalized_title_sha256.isin(set(anc.normalized_title_sha256))
    return float(np.mean([(nw[(nw.L == l) & nw.a].z.mean() - nw[(nw.L == l) & ~nw.a].z.mean()) / nw.z.std() for l in range(4)]))


def selection_gap(d, kappa, rng, reps=200, target_share=0.6):
    zs = ((d.z - d.z.mean()) / d.z.std()).to_numpy(); c = np.log(target_share / (1 - target_share))
    g = [zs[s].mean() - zs[~s].mean() for s in (rng.random(len(d)) < expit(kappa * zs + c) for _ in range(reps))]
    return float(np.mean(g))


def _lvframe(d):
    return pd.DataFrame({"bc_level": d.L.to_numpy(), "bb_label": d.ya.to_numpy().astype(int)})


def v1_crossfit(d, reps, rng, folds=5):
    rows = []
    cl = d.cluster_id.unique()
    for r in range(reps):
        f = dict(zip(cl, rng.permutation(len(cl)) % folds)); fd = d.cluster_id.map(f).to_numpy()
        lab = {k: np.empty(len(d)) for k in ("r6", "hard_aligned", "soft_isotonic", "scl_gamma0", "scl_gamma1", "scl_flex")}
        for k in range(folds):
            tr, te = fd != k, fd == k
            fit = proxies_fit(d.L.to_numpy()[tr], d.z.to_numpy()[tr], d.ya.to_numpy()[tr], _lvframe(d[tr]))
            for name, q in proxies_label(fit, d.L.to_numpy()[te], d.z.to_numpy()[te]).items():
                lab[name][te] = q
        truth = linking.risks(d.ya.to_numpy(), d.p.to_numpy())
        for name, q in lab.items():
            est = linking.risks(q, d.p.to_numpy())
            rows.append({"rep": r, "proxy": name, **{f"{m}_est": est[m] for m in est}, **{f"{m}_true": truth[m] for m in truth},
                         **{f"{m}_bias": est[m] - truth[m] for m in est},
                         "label_nll_vs_YA": M.nll(d.ya.to_numpy(), np.clip(q, 1e-6, 1 - 1e-6))})
    return pd.DataFrame(rows)


def v2_selection(d, kappas, deltas, reps, rng, target_share=0.6):
    rows = []
    zs = (d.z - d.z.mean()) / d.z.std(); L = d.L.to_numpy(); z = d.z.to_numpy(); ya = d.ya.to_numpy(); p = d.p.to_numpy()
    for kap in kappas:
        for dl in deltas:
            eta = kap * zs.to_numpy() + dl * (ya - ya.mean())
            c = np.log(target_share / (1 - target_share))
            for r in range(reps):
                sel = rng.random(len(d)) < expit(eta + c)
                if sel.all() or (~sel).sum() < 50:
                    continue
                fit = proxies_fit(L[sel], z[sel], ya[sel], _lvframe(d[sel]))
                truth = linking.risks(ya[~sel], p[~sel])
                for name, q in proxies_label(fit, L[~sel], z[~sel]).items():
                    est = linking.risks(q, p[~sel])
                    rows.append({"kappa_score": kap, "delta_label": dl, "rep": r, "proxy": name, "n_fit": int(sel.sum()),
                                 **{f"{m}_bias": est[m] - truth[m] for m in est},
                                 "prev_bias": float(np.mean(q) - np.mean(ya[~sel]))})
    return pd.DataFrame(rows)


def v3_anchor_budget(d, sizes, reps, rng):
    """How many anchors does SCL need? Subsample n anchor publisher-cluster-wise, fit on the subsample, estimate
    the risk of the full anchor set (truth known); report bias and SD of the estimate. Hard aligned for comparison."""
    rows = []; cl = d.cluster_id.unique(); gi = d.groupby("cluster_id").indices
    truth = linking.risks(d.ya.to_numpy(), d.p.to_numpy())
    for n in sizes:
        for r in range(reps):
            pick = rng.permutation(cl); idx = []
            for c in pick:
                idx.extend(gi[c])
                if len(idx) >= n:
                    break
            s = d.iloc[np.array(idx)]
            if s.ya.nunique() < 2:
                continue
            fit = proxies_fit(s.L.to_numpy(), s.z.to_numpy(), s.ya.to_numpy(), _lvframe(s))
            for name, q in proxies_label(fit, d.L.to_numpy(), d.z.to_numpy()).items():
                if name not in ("scl_gamma1", "scl_flex", "hard_aligned"):
                    continue
                est = linking.risks(q, d.p.to_numpy())
                rows.append({"n_anchors": n, "rep": r, "proxy": name, **{f"{m}_bias": est[m] - truth[m] for m in est}})
    return pd.DataFrame(rows)


def application(scores, pool, anc, lv, src_y, rng, boot):
    """Per seed and cell: risks under r6, hard aligned, and SCL over gamma; prior-corrected variants; source risk.
    Bootstrap: resample anchor publisher clusters (refit link) and target clusters (resample cell), seed-averaged."""
    pool_i = pool.set_index("row_id"); aset = set(anc.normalized_title_sha256)
    a_idx = anc.set_index("normalized_title_sha256")
    seeds = sorted(scores.seed.unique())
    per = {}
    for s in seeds:
        sc = scores[scores.seed == s]
        d = anchor_items(sc, pool, anc, lv)
        cells = {}
        for e in B_CELLS:
            c = sc[sc.env == e].copy()
            c["L"] = lv.loc[c.row_id].to_numpy().astype(int); c["z"] = linking.logit(c.p)
            c["cluster"] = pool_i.loc[c.row_id, "cluster_id"].to_numpy()
            h = pool_i.loc[c.row_id, "normalized_title_sha256"]
            c["ya_obs"] = np.where(h.isin(aset), a_idx.reindex(h).bb_label.to_numpy(), np.nan).astype(float)
            cells[e] = c.reset_index(drop=True)
        src = sc[sc.env == "banglabait:news"]
        per[s] = (d, cells, linking.risks(src_y.loc[src.row_id].to_numpy(), src.p.to_numpy()), float(src.pi_train.iloc[0]))

    def evaluate(per_sel):
        out = []
        for s, (d, cells, rs, pi_s) in per_sel.items():
            full = linking.fit(d.L, d.z, d.ya, 1.0, flex=True)  # primary: flexible score function (see run.E19 manifest)
            links = {gm: linking.fit(d.L, d.z, d.ya, gm, beta=full["score"]) for gm in GAMMAS}
            lin = linking.fit(d.L, d.z, d.ya, 1.0)
            for e, c in cells.items():
                L, z, p = c.L.to_numpy(), c.z.to_numpy(), c.p.to_numpy()
                yo = c.ya_obs.to_numpy() if e == "banglaclick:news" else None
                labs = {"r6": (L >= 1).astype(float), "hard_aligned": (L >= 2).astype(float)}
                for gm, lk in links.items():
                    labs[f"scl_g{gm:g}"] = linking.cell_labels(lk, L, z, yo)
                labs["scl_lin_g1"] = linking.cell_labels(lin, L, z, yo)
                for name, q in labs.items():
                    pi_t = float(np.mean(q)); pc = priors.prior_adjust(p, pi_s, pi_t)
                    r = linking.risks(q, p); rp = linking.risks(q, pc)
                    out.append({"seed": s, "env": e, "label": name, "prevalence": pi_t, "auroc": r["auroc"], "nll": r["nll"],
                                "brier": r["brier"], "nll_prior_corrected": rp["nll"], "brier_prior_corrected": rp["brier"],
                                "nll_gap": r["nll"] - rs["nll"], "nll_residual_after_prior": rp["nll"] - rs["nll"],
                                "auroc_gap": r["auroc"] - rs["auroc"], "src_auroc": rs["auroc"], "src_nll": rs["nll"],
                                "beta_hat": full["beta_hat"]})
        return pd.DataFrame(out)

    point = evaluate(per)
    boots = []
    for b in range(boot):
        sel = {}
        acl = per[seeds[0]][0].cluster_id.unique(); apick = rng.choice(acl, len(acl), replace=True)
        cpick = {e: rng.choice(per[seeds[0]][1][e].cluster.unique(), per[seeds[0]][1][e].cluster.nunique(), replace=True) for e in B_CELLS}
        for s, (d, cells, rs, pi_s) in per.items():
            gi = d.groupby("cluster_id").indices; ai = np.concatenate([gi[k] for k in apick])
            cs = {}
            for e, c in cells.items():
                ci = c.groupby("cluster").indices; cs[e] = c.iloc[np.concatenate([ci[k] for k in cpick[e]])]
            sel[s] = (d.iloc[ai], cs, rs, pi_s)
        r = evaluate(sel).groupby(["env", "label"]).mean(numeric_only=True).reset_index(); r["draw"] = b; boots.append(r)
    return point, pd.concat(boots, ignore_index=True)


def run(pool, src_test, anc, lv, src_y, seeds, seed_rng, out_write, boot=300, extra_scorers=None):
    t0 = time.time(); rng = np.random.default_rng(seed_rng)
    sc = blind_tfidf(pool, src_test, seeds); sc["scorer"] = "tfidf_lr"
    scorers = {"tfidf_lr": sc}
    for name, frame in (extra_scorers or {}).items():
        scorers[name] = frame
    v1s, v2s, v3s, pts, bts, gaps = [], [], [], [], [], []
    for name, s in scorers.items():
        for sd in sorted(s.seed.unique()):
            d = anchor_items(s[s.seed == sd], pool, anc, lv)
            v = v1_crossfit(d, 20, rng); v["scorer"], v["seed"] = name, sd; v1s.append(v)
            v = v2_selection(d, [0.0, 0.5, 1.0, 2.0], [0.0, 1.0], 200, rng); v["scorer"], v["seed"] = name, sd; v2s.append(v)
            v = v3_anchor_budget(d, [50, 100, 200, 400, 800], 200, rng); v["scorer"], v["seed"] = name, sd; v3s.append(v)
            gaps.append({"scorer": name, "seed": sd, "observed_gap_sd": observed_gap(s[s.seed == sd], pool, anc, lv),
                         **{f"induced_gap_kappa{k:g}": selection_gap(d, k, rng) for k in (0.5, 1.0, 2.0)}})
        if src_y is not None and s.env.eq("banglabait:news").any():
            pt, bt = application(s, pool, anc, lv, src_y, rng, boot)
            pt["scorer"], bt["scorer"] = name, name; pts.append(pt); bts.append(bt)
    return sc, pd.concat(v1s), pd.concat(v2s), pd.concat(v3s), pd.concat(pts), pd.concat(bts), pd.DataFrame(gaps)
