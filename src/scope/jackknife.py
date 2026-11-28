"""Delete-one-cluster jackknife for cells with few clusters (E30, E31), and its size at G = 10, 13, 26.

For a statistic theta computed on a cell with G clusters, theta_(-k) is the statistic with cluster k removed.
  SE_jk^2 = (G - 1)/G * sum_k (theta_(-k) - mean_k theta_(-k))^2     (Efron and Stein 1981; the cluster version
  is the CR3 variance for smooth functionals, MacKinnon, Nielsen and Webb 2023, J. Econometrics 232(2))
  interval: theta_hat +- t_{G-1, 0.975} * SE_jk.
The leave-one-cluster-out range [min_k theta_(-k), max_k theta_(-k)] is reported beside it as a plain
influence diagnostic. Neither is assumed exact; jackknife_size() measures the size of the jackknife t-test
for an AUROC contrast at the cluster counts of the real cells, next to the pairs cluster bootstrap.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from . import metrics as M


def summarize(full: dict, loo: list[dict], G: int, keys=None) -> list[dict]:
    keys = keys or [k for k, v in full.items() if isinstance(v, (int, float, np.floating)) and np.isfinite(v)]
    L = pd.DataFrame(loo)
    rows = []
    tq = stats.t.ppf(0.975, G - 1)
    for k in keys:
        v = L[k].to_numpy(float); v = v[np.isfinite(v)]
        if len(v) < 2:
            continue
        se = float(np.sqrt((len(v) - 1) / len(v) * np.sum((v - v.mean()) ** 2)))
        rows.append({"quantity": k, "estimate": float(full[k]), "jk_se": se, "jk_lo": float(full[k] - tq * se),
                     "jk_hi": float(full[k] + tq * se), "loco_min": float(v.min()), "loco_max": float(v.max()),
                     "G": G, "n_valid_deletions": int(len(v)), "t_df": G - 1})
    return rows


def _cell(G, rng, dominant, mean_size=40, sigma_u=0.8, sigma_a=0.5):
    """Two exchangeable scorers of one binary label with cluster-varying signal, so the true AUROC contrast is 0."""
    sizes = np.maximum(3, rng.lognormal(np.log(mean_size) - 0.5, 1.0, G).astype(int))
    if dominant:  # one cluster holds about half the items (as the largest YouTube channel)
        sizes[0] = sizes[1:].sum()
    cl = np.repeat(np.arange(G), sizes); n = len(cl)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-0.5 + rng.normal(0, sigma_u, G)[cl])))).astype(float)
    shared = rng.normal(0, 0.5, G)[cl]
    s1 = (1 + rng.normal(0, sigma_a, G))[cl] * y + shared + rng.normal(size=n)
    s2 = (1 + rng.normal(0, sigma_a, G))[cl] * y + shared + rng.normal(size=n)
    return y, s1, s2, cl


def _contrast(y, s1, s2):
    return M.auroc(y, s1) - M.auroc(y, s2)


def jackknife_size(G_list=(10, 13, 26), reps=500, boot=399, seed=0):
    rng = np.random.default_rng(seed); rows = []
    for G in G_list:
        for dominant in (False, True):
            rej = {"jackknife_t": [], "pairs_bootstrap": []}
            for _ in range(reps):
                y, s1, s2, cl = _cell(G, rng, dominant)
                if y.min() == y.max():
                    continue
                th = _contrast(y, s1, s2)
                loo = [_contrast(y[cl != k], s1[cl != k], s2[cl != k]) for k in range(G)]
                loo = np.array([v for v in loo if np.isfinite(v)])
                se = np.sqrt((len(loo) - 1) / len(loo) * np.sum((loo - loo.mean()) ** 2))
                rej["jackknife_t"].append(float(abs(th / se) > stats.t.ppf(0.975, G - 1)) if se > 0 else np.nan)
                groups = [np.flatnonzero(cl == k) for k in range(G)]; bs = []
                for _ in range(boot):
                    idx = np.concatenate([groups[i] for i in rng.integers(0, G, G)])
                    v = _contrast(y[idx], s1[idx], s2[idx])
                    if np.isfinite(v):
                        bs.append(v)
                lo, hi = np.percentile(bs, [2.5, 97.5])
                rej["pairs_bootstrap"].append(float(not (lo <= 0 <= hi)))
            for m, r in rej.items():
                r = np.array(r, float); r = r[np.isfinite(r)]
                rows.append({"G": G, "dominant_cluster": dominant, "method": m, "reps_valid": int(len(r)),
                             "size_at_5pct": float(r.mean()), "mc_se": float(np.sqrt(r.mean() * (1 - r.mean()) / len(r)))})
    return pd.DataFrame(rows)
