"""E3/E5: label-shift estimation and prior correction from unlabelled target scores.

Methods (all adopted from the literature, named as such):
  prior_adjust  Bayes-rule odds rescaling (Saerens et al., 2002; Elkan, 2001)
  mlls_em       maximum-likelihood label shift by EM on calibrated scores (Saerens et al., 2002;
                calibrated variant: Alexandari et al., 2020)
  bbse          black-box shift estimation from a hard-classifier confusion matrix (Lipton et al., 2018)
Each estimator assumes label shift: P(x | y) is unchanged between source and target. Our E1
result says this assumption fails for the R6 target label, which is what E3 tests.
"""
from __future__ import annotations
import numpy as np

EPS = 1e-9


def prior_adjust(p, pi_s: float, pi_t: float):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    w1, w0 = pi_t / pi_s, (1 - pi_t) / (1 - pi_s)
    return (w1 * p) / (w1 * p + w0 * (1 - p))


def mlls_em(p, pi_s: float, iters: int = 500, tol: float = 1e-8) -> float:
    pi = pi_s
    for _ in range(iters):
        new = float(np.mean(prior_adjust(p, pi_s, pi)))
        if abs(new - pi) < tol:
            return new
        pi = new
    return pi


def bbse(p_target, p_src_holdout, y_src_holdout, t: float = 0.5) -> float:
    yhat_s = np.asarray(p_src_holdout) >= t; y = np.asarray(y_src_holdout).astype(bool)
    # C[i,j] = P(yhat=i, y=j) on source holdout
    C = np.array([[np.mean((yhat_s == i) & (y == j)) for j in (0, 1)] for i in (0, 1)])
    mu = np.array([np.mean(np.asarray(p_target) < t), np.mean(np.asarray(p_target) >= t)])
    try:
        w = np.linalg.solve(C, mu)
    except np.linalg.LinAlgError:
        return float("nan")
    q = w * np.array([1 - y.mean(), y.mean()])  # w_j * pi_s(j) = pi_t(j)
    q = np.clip(q, 0, None)
    return float(q[1] / q.sum()) if q.sum() > 0 else float("nan")


def cc(p_target, t: float = 0.5) -> float:
    return float(np.mean(np.asarray(p_target) >= t))


def pcc(p_target) -> float:
    return float(np.mean(p_target))
