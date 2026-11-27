"""E13: reliability of cluster-level inference at the observed numbers of clusters.

Data-generating process (calibrated to the observed cells, clearly labelled as simulation):
G clusters with unequal sizes (lognormal, matching the mean cluster size of the cell), a binary
cue with cluster-varying prevalence, and a binary label from a logistic model with a cluster
random intercept (SD sigma_u) and a true cue effect. The estimand is the risk difference
(linear-probability slope), the R6 H3 scale. Methods compared:
  pairs   percentile cluster (pairs) bootstrap, the R6 method
  cr1     cluster-robust t with G-1 df
  wcr     wild cluster restricted bootstrap-t, Rademacher weights (wild_cluster_rd_pvalue)
Metric: empirical rejection rate of a TRUE null (size) at nominal 5%, and coverage for pairs.
"""
from __future__ import annotations
import numpy as np
from scipy import stats
from .transport import wild_cluster_rd_pvalue


def simulate_cell(G, mean_size, sigma_u, beta, rng):
    sizes = np.maximum(3, rng.lognormal(np.log(mean_size) - 0.5, 1.0, G).astype(int))
    cl = np.repeat(np.arange(G), sizes)
    u = rng.normal(0, sigma_u, G)[cl]
    cue_p = rng.beta(2, 5, G)[cl]
    x = (rng.random(len(cl)) < cue_p).astype(float)
    eta = -0.5 + beta * x + u
    y = (rng.random(len(cl)) < 1 / (1 + np.exp(-eta))).astype(float)
    return y, x, cl


def cr1_p(y, x, cl):
    G = cl.max() + 1; X = np.column_stack([np.ones_like(x), x])
    XtX_inv = np.linalg.inv(X.T @ X); b = XtX_inv @ X.T @ y; u = y - X @ b
    sc = np.zeros((G, 2)); np.add.at(sc, cl, X * u[:, None])
    n = len(y); V = XtX_inv @ (sc.T @ sc) @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - 2))
    t = b[1] / np.sqrt(V[1, 1])
    return float(2 * stats.t.sf(abs(t), G - 1))


def pairs_reject(y, x, cl, draws, rng):
    G = cl.max() + 1; groups = [np.flatnonzero(cl == g) for g in range(G)]
    est = []
    for _ in range(draws):
        idx = np.concatenate([groups[i] for i in rng.integers(0, G, G)])
        xx = x[idx]
        if xx.min() == xx.max():
            continue  # R6 'invalid draw'; here skipped to measure the best case
        est.append(np.polyfit(xx, y[idx], 1)[0])
    if len(est) < 0.5 * draws:
        return np.nan
    lo, hi = np.percentile(est, [2.5, 97.5])
    return float(not (lo <= 0 <= hi))


def run(G_list, reps, seed, mean_size=40, sigma_u=0.8, boot=399):
    rng = np.random.default_rng(seed)
    rows = []
    for G in G_list:
        rej = {"pairs": [], "cr1": [], "wcr": []}
        for _ in range(reps):
            y, x, cl = simulate_cell(G, mean_size, sigma_u, 0.0, rng)
            if x.min() == x.max() or y.min() == y.max():
                continue
            rej["cr1"].append(cr1_p(y, x, cl) < 0.05)
            rej["wcr"].append(wild_cluster_rd_pvalue(y, x, cl, boot, rng)[1] < 0.05)
            rej["pairs"].append(pairs_reject(y, x, cl, boot, rng))
        for m, r in rej.items():
            r = np.array(r, float); r = r[~np.isnan(r)]
            rows.append({"G": G, "method": m, "reps_valid": int(len(r)), "size_at_5pct": float(r.mean()),
                         "mc_se": float(np.sqrt(r.mean() * (1 - r.mean()) / len(r)))})
    return rows
