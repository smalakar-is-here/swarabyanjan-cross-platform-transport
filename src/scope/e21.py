"""E21: design-based target auditing with a transported link (model-assisted anchor sampling).

Problem. On a target cell U (N items, all with level L_i and frozen score p_i) the source-construct label y_i
is unobserved. A budget of n items can be relabelled under the source construct. Estimate R*_U(f).

Representation. Every metric is a smooth function of population totals that are linear in y:
  log loss / Brier:  R = N^-1 sum_i [b_i + y_i d_i]                        (d, b known from p_i)
  weighted AUROC:    R = (T1 - S^2/2) / (S (N - S)),  T1 = sum_i y_i c_i,  S = sum_i y_i,  c_i = midrank(p_i) - 1/2.
Model-assisted (difference) estimator. With a working link h_i = h(L_i, p_i) fitted on *other* data (anchors of
another cell; here: the other half of the anchor clusters) and a probability sample s with inclusion probabilities
pi_i, each total T_w = sum_i y_i w_i is estimated by
    T_w_hat = sum_{i in U} h_i w_i + sum_{i in s} (y_i - h_i) w_i / pi_i,
which is design-unbiased for every h (Särndal, Swensson & Wretman, 1992, ch. 6). AUROC is the plug-in of the two
estimated totals (design-consistent); its variance by linearisation with u_i = (y_i - h_i)(a c_i + b),
a = dR/dT1, b = dR/dS. No assumption on h, the link or its transport is needed for validity; a better h only
reduces variance. Inference is design-based, so it does not rest on a few-cluster asymptotic.

Designs. SRS; stratified by level (proportional); stratified by level x score tercile with Neyman allocation
n_h ∝ N_h sqrt(mean_h h(1-h) w_h^2) for the AUROC influence weights w (at the plug-in value), >= 2 per stratum.
Estimators compared. sample-only (Horvitz-Thompson / Hajek), model-only plug-in (h, no labels), difference
(model-assisted), and SCL refit on the sample (E19, model-based).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
from . import linking
from . import metrics as M
from .bounds import _d_coef


def _auroc_from_totals(T1, S, N):
    return (T1 - S * S / 2) / (S * (N - S))


def _grad(T1, S, N):
    den = S * (N - S)
    a = 1 / den
    b = (-S * den - (T1 - S * S / 2) * (N - 2 * S)) / den ** 2
    return a, b


def allocate(strata, n, h, w, design):
    labs, inv = np.unique(strata, return_inverse=True); Nh = np.bincount(inv).astype(float)
    if design == "neyman":
        sd = np.sqrt(np.bincount(inv, h * (1 - h) * w * w) / Nh) + 1e-9
        raw = Nh * sd
    else:
        raw = Nh.copy()
    nh = np.maximum(2, np.floor(raw / raw.sum() * n)).astype(int); nh = np.minimum(nh, Nh.astype(int))
    while nh.sum() < n and (nh < Nh).any():  # hand out the remainder by largest fractional need
        gap = raw / raw.sum() * n - nh; gap[nh >= Nh] = -np.inf; nh[np.argmax(gap)] += 1
    while nh.sum() > n:
        k = np.argmax(np.where(nh > 2, nh - raw / raw.sum() * n, -np.inf)); nh[k] -= 1
    return inv, Nh, nh


def draw(inv, Nh, nh, rng):
    s = np.zeros(len(inv), bool); pi = np.zeros(len(inv))
    for k in range(len(Nh)):
        idx = np.flatnonzero(inv == k); s[rng.choice(idx, nh[k], replace=False)] = True; pi[idx] = nh[k] / Nh[k]
    return s, pi


def estimate(y, p, L, z, h, s, pi, inv, Nh, nh):
    """Point estimates and 95% design-based CIs for AUROC and log loss under each estimator."""
    N = len(p); c = rankdata(p) - 0.5; d, b = _d_coef(p, "nll"); out = {}
    ws = 1 / pi[s]
    # difference estimator
    S = h.sum() + np.sum((y[s] - h[s]) * ws); T1 = (h * c).sum() + np.sum((y[s] - h[s]) * c[s] * ws)
    a, bb = _grad(T1, S, N)

    def strat_var(u):  # variance of sum_{s} u_i / pi_i under stratified SRS without replacement
        v = 0.0
        for k in range(len(Nh)):
            m = s & (inv == k)
            if nh[k] > 1:
                v += Nh[k] ** 2 * (1 - nh[k] / Nh[k]) * np.var(u[m], ddof=1) / nh[k]
        return v
    au = _auroc_from_totals(T1, S, N); se = np.sqrt(strat_var((y - h) * (a * c + bb)))
    nl = (b.sum() + (h * d).sum() + np.sum((y[s] - h[s]) * d[s] * ws)) / N; sen = np.sqrt(strat_var((y - h) * d)) / N
    out["diff"] = (au, se, nl, sen)
    # sample-only (Hajek-type plug-in for AUROC; HT for log loss; same linearisation with h = 0)
    S0 = np.sum(y[s] * ws); T0 = np.sum(y[s] * c[s] * ws); a0, b0 = _grad(T0, S0, N) if 0 < S0 < N else (np.nan, np.nan)
    u0 = y * (a0 * c + b0); au0 = _auroc_from_totals(T0, S0, N) if 0 < S0 < N else np.nan
    nl0 = (b.sum() + np.sum(y[s] * d[s] * ws)) / N
    out["sample_only"] = (au0, np.sqrt(strat_var(u0)), nl0, np.sqrt(strat_var(y * d)) / N)
    out["model_only"] = (M.auroc(h, p), np.nan, M.nll(h, p), np.nan)
    if len(np.unique(y[s])) == 2:
        lk = linking.fit(L[s], z[s], y[s], 1.0, flex=True); q = linking.predict(lk, L, z); q[s] = y[s]
        out["scl_refit"] = (M.auroc(q, p), np.nan, M.nll(q, p), np.nan)
    return out


def simulate(d, sizes, reps, rng, h_shift=0.0):
    """Population = anchors of a random half of the publisher clusters; h = flexible SCL link fitted on the other
    half (stands in for a link transported from another cell). Truth = metrics of the population's Y_A."""
    rows = []; cl = d.cluster_id.unique()
    for r in range(reps):
        half = set(rng.permutation(cl)[: len(cl) // 2]); aux = d[~d.cluster_id.isin(half)]; pop = d[d.cluster_id.isin(half)]
        if pop.ya.nunique() < 2 or aux.ya.nunique() < 2:
            continue
        L, z, p, y = pop.L.to_numpy(), pop.z.to_numpy(), pop.p.to_numpy(), pop.ya.to_numpy()
        h = linking.predict(linking.fit(aux.L, aux.z, aux.ya, 1.0, flex=True), L, z)
        if h_shift:  # working link mis-transported by h_shift on the logit scale (design validity check)
            h = 1 / (1 + np.exp(-(np.log(h / (1 - h)) + h_shift)))
        tau_au, tau_nl = M.auroc(y, p), M.nll(y, p)
        c = rankdata(p) - 0.5; S0 = h.sum(); a, bb = _grad((h * c).sum(), S0, len(p)); w_au = a * c + bb
        terc = np.digitize(z, np.quantile(z, [1 / 3, 2 / 3]))
        designs = {"srs": (np.zeros(len(p), int), "prop"), "level_prop": (L, "prop"), "level_x_score_neyman": (L * 3 + terc, "neyman")}
        for n in sizes:
            if n >= len(p) * 0.8:
                continue
            for dn, (st, kind) in designs.items():
                inv, Nh, nh = allocate(st, n, h, w_au, kind)
                s, pi = draw(inv, Nh, nh, rng)
                for est, (au, se, nl, sen) in estimate(y, p, L, z, h, s, pi, inv, Nh, nh).items():
                    rows.append({"rep": r, "n": n, "N": len(p), "design": dn, "estimator": est,
                                 "auroc_err": au - tau_au, "nll_err": nl - tau_nl,
                                 "auroc_cover": (abs(au - tau_au) <= norm.ppf(.975) * se) if np.isfinite(se) else np.nan,
                                 "nll_cover": (abs(nl - tau_nl) <= norm.ppf(.975) * sen) if np.isfinite(sen) else np.nan,
                                 "auroc_se": se})
    return pd.DataFrame(rows)
