"""E19: score-conditional anchor linking (SCL).

Problem. The level-only link g(l) = P(Y_A=1 | L=l) used by soft linking needs two conditions that both
fail on the real anchors (E10/E18): anchor representativeness given L (Assumption 1) and Y_A independent
of the model score given L (Proposition 1). The documented Assumption-1 violation is a difference in the
*score distribution* within a rubric level between anchors and non-anchors.

Estimand. For a frozen scorer f on regime-B cell e, the source-construct risk
    R*_e(f) = E_e[ l(f(x), Y_A) ],   l(p, y) = y a(p) + (1 - y) b(p)  (log loss, Brier; AUROC as a ratio).

Conditional link. h_e(l, s) = P_e(Y_A = 1 | L = l, f(x) = s). By the tower rule R*_e(f) = E_e[l(f; h_e)]
with no further condition, because f is in the conditioning set (the condition of Proposition 1 holds
trivially). Identification of h_e from anchors needs only
    Assumption 1' (conditional representativeness):  Y_A independent of {anchor membership} given (L, f(x)),
which permits exactly the violation observed (P(f | L) differs between anchors and non-anchors).
For cells other than the anchor cell, using the anchor link additionally assumes that the within-level
score-construct slope transports. We do not assume it: the family
    h_gamma(l, z) = sigmoid(alpha_l(gamma) + gamma * beta_hat * z),   z = logit f(x),
with alpha refitted on anchors at each fixed slope, interpolates between the level-only link (gamma = 0,
soft linking) and the full conditional link (gamma = 1). Reporting R*_e over gamma in [0, 1] is a
sensitivity analysis under "the model's within-level signal about Y_A on cell e is attenuated, not
reversed, relative to news". The estimator must use a scorer that never saw the anchors.
"""
from __future__ import annotations
import numpy as np
from scipy.special import expit
from . import metrics as M

LEVELS = 4


def _design(L):
    X = np.zeros((len(L), LEVELS)); X[np.arange(len(L)), np.asarray(L, int)] = 1.0
    return X


def _irls(X, y, offset, ridge=1e-3, iters=100):
    """Logistic regression by Newton-Raphson with a small ridge on every coefficient."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ w + offset; p = expit(eta)
        g = X.T @ (y - p) - ridge * w
        H = (X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        m = np.max(np.abs(step))
        if m > 2.0:  # damped Newton: bounded steps keep separated subsets from diverging
            step = step * (2.0 / m)
        w = w + step
        if m < 1e-10:
            break
    return w


def _zbasis(z, knots):
    z = np.asarray(z, float)
    return np.column_stack([z] + [np.maximum(z - k, 0.0) for k in knots]) if len(knots) else z[:, None]


def fit(L, z, ya, gamma=1.0, beta=None, flex=False):
    """Fit h_gamma on anchors: logit h = alpha_l + gamma * s(z).

    s(z) = beta z (linear) or, with flex=True, a linear spline in z with hinges at the 25/50/75% anchor quantiles
    (s(z) = beta z + sum_k theta_k (z - k)_+). s is fitted once on the full model (gamma = 1) unless given as `beta`
    (a previous fit's "score" entry, or a float slope); alpha is refitted with gamma * s(z) as an offset.
    Returns dict(alpha[4], beta_hat (linear coefficient), score, gamma)."""
    L = np.asarray(L, int); z = np.asarray(z, float); ya = np.asarray(ya, float)
    if isinstance(beta, (int, float, np.floating)):
        beta = {"coef": np.array([float(beta)]), "knots": np.array([])}
    if beta is None:
        knots = np.quantile(z, [0.25, 0.5, 0.75]) if flex else np.array([])
        Z = _zbasis(z, knots); w = _irls(np.column_stack([_design(L), Z]), ya, np.zeros(len(L)))
        beta = {"coef": w[LEVELS:], "knots": knots}
    s = _zbasis(z, beta["knots"]) @ beta["coef"]
    alpha = _irls(_design(L), ya, gamma * s)
    return {"alpha": alpha, "beta_hat": float(beta["coef"][0]), "score": beta, "gamma": gamma}


def predict(link, L, z):
    s = _zbasis(z, link["score"]["knots"]) @ link["score"]["coef"]
    return expit(link["alpha"][np.asarray(L, int)] + link["gamma"] * s)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def risks(q, p):
    """Source-construct risk estimates of scorer p under soft label q (or true hard labels)."""
    return {"nll": M.nll(q, p), "brier": M.brier(q, p), "auroc": M.auroc(q, p)}


def cell_labels(link, L, z, ya_obs=None):
    """Soft source-construct labels for a cell: the observed Y_A where available (anchors), h elsewhere."""
    q = predict(link, L, z)
    if ya_obs is not None:
        ya_obs = np.asarray(ya_obs, float); m = ~np.isnan(ya_obs); q = q.copy(); q[m] = ya_obs[m]
    return q
