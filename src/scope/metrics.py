"""Metrics that accept hard (0/1) or soft (probabilistic) target labels.

A soft label q in [0,1] is the probability that the item is positive under the reference
construct. With q in {0,1} every function reduces to its usual definition.
"""
from __future__ import annotations
import numpy as np

EPS = 1e-12


def nll(q, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(np.mean(-(q * np.log(p) + (1 - q) * np.log(1 - p))))


def brier(q, p):
    """Expected Brier score E[(Y - p)^2] for Y ~ Bernoulli(q): (p - q)^2 + q(1 - q).
    Equals the usual Brier score for hard labels. Without the q(1 - q) term a soft label would
    understate the risk by the label variance (found by the E15 synthetic check)."""
    q = np.asarray(q, float); p = np.asarray(p, float)
    return float(np.mean((p - q) ** 2 + q * (1 - q)))


def auroc(q, p):
    """Weighted AUROC: each item is a positive with weight q and a negative with weight 1-q (ties count 1/2)."""
    q = np.asarray(q, float); p = np.asarray(p, float)
    _, inv = np.unique(p, return_inverse=True)
    wp = np.bincount(inv, weights=q); wn = np.bincount(inv, weights=1 - q)
    below = np.concatenate([[0.0], np.cumsum(wn)[:-1]])
    den = wp.sum() * wn.sum()
    return float(np.sum(wp * (below + 0.5 * wn)) / den) if den > 0 else float("nan")


def auprc(q, p):
    q = np.asarray(q, float); order = np.argsort(-np.asarray(p, float), kind="mergesort")
    q = q[order]; tp = np.cumsum(q); k = np.arange(1, len(q) + 1)
    prec = tp / k; total = q.sum()
    return float(np.sum(prec * q) / total) if total > 0 else float("nan")


def ece(q, p, bins=15):
    p = np.asarray(p, float); q = np.asarray(q, float)
    edges = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(p[m].mean() - q[m].mean())
    return float(tot)


def calibration_slope_intercept(y, p):
    """Logistic recalibration of hard labels on logit(p): returns (intercept, slope)."""
    from scipy.optimize import minimize
    y = np.asarray(y, float); z = np.log(np.clip(p, EPS, 1 - EPS) / np.clip(1 - p, EPS, 1))

    def f(w):
        s = w[0] + w[1] * z
        return np.mean(np.logaddexp(0, s) - y * s)
    w = minimize(f, x0=[0.0, 1.0], method="BFGS").x
    return float(w[0]), float(w[1])


def threshold_metrics(y, p, t=0.5):
    y = np.asarray(y).astype(int); yhat = (np.asarray(p) >= t).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1p = 2 * prec * rec / (prec + rec) if (tp and prec + rec) else 0.0
    prec0 = tn / (tn + fn) if tn + fn else float("nan")
    rec0 = tn / (tn + fp) if tn + fp else float("nan")
    f1n = 2 * prec0 * rec0 / (prec0 + rec0) if (tn and prec0 + rec0) else 0.0
    bal = float(np.nanmean([rec, rec0])) if (tp + fn and tn + fp) else float("nan")
    return {"accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec, "macro_f1": (f1p + f1n) / 2,
            "balanced_accuracy": bal}


def all_metrics(q, p, hard=None):
    out = {"n": len(q), "prevalence": float(np.mean(q)), "nll": nll(q, p), "brier": brier(q, p),
           "auroc": auroc(q, p), "auprc": auprc(q, p), "ece": ece(q, p)}
    if hard is not None:
        out.update(threshold_metrics(hard, p))
        out["cal_intercept"], out["cal_slope"] = calibration_slope_intercept(hard, p)
    return out


def kappa(a, b):
    a = np.asarray(a); b = np.asarray(b)
    po = np.mean(a == b); pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")
