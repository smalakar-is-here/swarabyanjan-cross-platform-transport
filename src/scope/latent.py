"""E22: latent-intensity model with regime-specific cut-points (concurrent calibration across label definitions).

Model. One latent clickbait intensity f(x) = w'x shared by every annotation regime; each regime observes it
through its own cut-points (a graded-response / cumulative-logit model, Samejima 1969; McCullagh 1980):
  regime A (BanglaBait, binary):        P(Y_A = 1 | x) = sigma(f(x) - tau_A)
  regime B (BanglaClick, level 0..3):   P(L >= k | x)  = sigma(f(x) - theta_k),  theta_1 < theta_2 < theta_3
  regime C (BaitBuster, binary):        P(Y_C = 1 | x) = sigma(f(x) - tau_C)
Fitted jointly by penalised maximum likelihood (sum of log-likelihoods + ||w||^2 / 2, the liblinear C = 1 scale
of the E10 logistic baseline). Items labelled by two regimes (anchors) enter once per regime, which links the
cut-point scales as common items link test forms in concurrent IRT calibration (Kolen & Brennan, 2014).

Prediction for a target construct t in {A, B>=1 (published), B>=2 (aligned), C}: sigma((f(x) - cut_t) / T), T a
scalar temperature fitted on the cluster-grouped validation split under the same construct. If the target regime
is absent from training (C held out), cut_t is not identified; the source cut tau_A is used and the case is flagged.

Contrast with pooled ERM, which binarises every regime first and fits a single intercept, i.e. forces
tau_A = theta_t = tau_C: the model nests pooled ERM (equal cuts) and removes exactly the constraint that the
label-definition shift violates. It adds 4 parameters.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit

REG = {"A": 0, "B": 1, "C": 2}


def _unpack(v, D):
    w = v[:D]; tA, tC, t1, g2, g3 = v[D:D + 5]
    th = np.array([t1, t1 + np.exp(g2), t1 + np.exp(g2) + np.exp(g3)])
    return w, tA, tC, th


def nll_grad(v, X, reg, y, lam=1.0):
    """Negative penalised log-likelihood and gradient. reg: 0/1/2 per row; y: binary label (A, C) or level (B)."""
    D = X.shape[1]; w, tA, tC, th = _unpack(v, D); f = X @ w
    gf = np.zeros(len(y)); gA = gC = 0.0; gth = np.zeros(3); ll = 0.0
    for r, tau in ((0, tA), (2, tC)):
        m = reg == r
        if m.any():
            e = f[m] - tau; yy = y[m]
            ll += np.sum(yy * log_expit(e) + (1 - yy) * log_expit(-e))
            res = yy - expit(e); gf[m] = res
            if r == 0:
                gA = -res.sum()
            else:
                gC = -res.sum()
    m = reg == 1
    if m.any():
        fb = f[m]; k = y[m].astype(int)
        cut = np.c_[np.full(len(fb), -np.inf), np.tile(th, (len(fb), 1)), np.full(len(fb), np.inf)]
        lo, hi = cut[np.arange(len(fb)), k], cut[np.arange(len(fb)), k + 1]
        a, b = expit(fb - lo), expit(fb - hi)       # P(L>=k), P(L>=k+1)
        P = np.clip(a - b, 1e-12, None); ll += np.sum(np.log(P))
        da, db = a * (1 - a), b * (1 - b)
        gf[m] = (da - db) / P
        for j in range(3):  # theta_{j+1} is the lower cut of level j+1 and the upper cut of level j
            gth[j] = -np.sum((da / P)[k == j + 1]) + np.sum((db / P)[k == j])
    obj = -ll + 0.5 * lam * np.dot(w, w)
    gw = -(X.T @ gf) + lam * w
    # chain rule for the ordered parametrisation theta = t1, t1 + e^g2, t1 + e^g2 + e^g3
    g_t1 = -gth.sum(); g_g2 = -(gth[1] + gth[2]) * np.exp(v[D + 3]); g_g3 = -gth[2] * np.exp(v[D + 4])
    return obj, np.r_[gw, -gA, -gC, g_t1, g_g2, g_g3]  # gA, gC, gth are d(log-lik)/d(cut); obj = -log-lik + penalty


def fit(X, reg, y, lam=1.0, maxiter=500):
    D = X.shape[1]
    v0 = np.r_[np.zeros(D), 0.0, 0.0, -1.0, 0.0, 0.0]
    r = minimize(nll_grad, v0, args=(X, reg, y, lam), jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    w, tA, tC, th = _unpack(r.x, D)
    return {"w": w, "tau_A": tA, "tau_C": tC, "theta": th, "converged": bool(r.success), "has_C": bool((reg == 2).any())}


def cut(model, construct):
    return {"A": model["tau_A"], "B1": model["theta"][0], "B2": model["theta"][1], "B3": model["theta"][2],
            "C": model["tau_C"] if model["has_C"] else model["tau_A"]}[construct]


def score(model, X, construct):
    return X @ model["w"] - cut(model, construct)
