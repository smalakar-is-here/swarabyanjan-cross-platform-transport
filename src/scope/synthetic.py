"""E15: controlled synthetic recovery with known ground truth.

Question: how well do target-label proxies recover the target risk of a frozen source model under
the *source* construct, R*(f) = E[l(Y_A*, f)], which is unobservable on real data?

Data-generating process (one replication):
  target level   L ~ Categorical(pi_L)                     pi_L = observed BanglaClick level shares
  source label   Y_A* ~ Bernoulli(g*(L))                    g* = isotonic link estimated on real anchors (E1)
  score          z = mu (1 - delta) [omega s(L) + (1 - omega)(2 Y_A* - 1)] + N(0, 1),  p = sigmoid(a + b z)
s(L) is logit g*(L) standardised over pi_L. omega is the share of the score signal that runs
through the rubric level (omega = 1: the score knows Y_A* only through L). (a, b) calibrate p
to Y_A* on an independent sample at delta = 0, so the model is calibrated for the source
construct before the injected shift. delta >= 0 removes discrimination (positive control: R* must rise).
Anchors: n_a items, L ~ pi_L, Y_A ~ Bern(g_anc(L)), logit g_anc = logit g* + lambda.
lambda = 0 is Assumption A1; lambda != 0 violates it.

Proxies for R* (each computed for NLL, Brier, AUROC):
  r6       Y = 1[L >= 1]                 (published binarisation)
  hard     Y = 1[L >= t_hat], t_hat maximises anchor kappa (aligned label of the main text)
  soft     q = g_hat(L)                   (soft linking, g_hat from the anchors)
Reported: bias and RMSE of each proxy against R*, and coverage of a 95% anchor-bootstrap interval
for the soft and hard proxies. The soft proxy conditions on L only, so it misprices any score
information about Y_A* beyond L; the simulation measures how large that is.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.special import expit, logit
from . import metrics as M
from .construct import pava

LOSSES = {"nll": M.nll, "brier": M.brier, "auroc": M.auroc}


def _score(L, ya, sL, mu, omega, delta, rng):
    return mu * (1 - delta) * (omega * sL[L] + (1 - omega) * (2 * ya - 1)) + rng.normal(0, 1, len(L))


def _calibrate(g, pi_L, sL, mu, omega, rng, n=50000):
    L = rng.choice(4, n, p=pi_L); y = (rng.random(n) < g[L]).astype(float); z = _score(L, y, sL, mu, omega, 0.0, rng)
    # exact convex fit (an earlier Nelder-Mead fit failed to converge in one cell of the first run)
    lr = LogisticRegression(C=1e8, solver="lbfgs", max_iter=1000).fit(z[:, None], y)
    return lr.intercept_[0], lr.coef_[0, 0]


def _link(L, ya):
    n = np.bincount(L, minlength=4).astype(float); k = np.bincount(L, weights=ya, minlength=4)
    return pava((k + 0.5) / (n + 1.0), n)


def _tstar(L, ya):
    return max((1, 2, 3), key=lambda t: M.kappa(ya.astype(int), (L >= t).astype(int)))


def _cell(g_true, pi_L, sL, mu, omega, delta, lam, na, reps, boot, seed, n_target):
    rng = np.random.default_rng(seed)
    a, b = _calibrate(g_true, pi_L, sL, mu, omega, rng)
    g_anc = expit(logit(np.clip(g_true, 1e-6, 1 - 1e-6)) + lam)
    rows = []
    for r in range(reps):
        L = rng.choice(4, n_target, p=pi_L); ya = (rng.random(n_target) < g_true[L]).astype(float)
        p = expit(a + b * _score(L, ya, sL, mu, omega, delta, rng))
        La = rng.choice(4, na, p=pi_L); yA = (rng.random(na) < g_anc[La]).astype(float)
        labels = {"r6": (L >= 1).astype(float), "hard": (L >= _tstar(La, yA)).astype(float), "soft": _link(La, yA)[L]}
        bs = {k: [] for k in ("hard", "soft")}
        for _ in range(boot):
            i = rng.integers(0, na, na)
            bh = (L >= _tstar(La[i], yA[i])).astype(float); bq = _link(La[i], yA[i])[L]
            bs["hard"].append({m: f(bh, p) for m, f in LOSSES.items()})
            bs["soft"].append({m: f(bq, p) for m, f in LOSSES.items()})
        row = {"omega": omega, "delta": delta, "lambda": lam, "n_anchor": na, "rep": r}
        for m, f in LOSSES.items():
            truth = f(ya, p); row[f"{m}_true"] = truth
            for k, y in labels.items():
                row[f"{m}_{k}"] = f(y, p)
            for k in ("hard", "soft"):
                v = np.array([d[m] for d in bs[k]]); lo, hi = np.quantile(v, [0.025, 0.975])
                row[f"{m}_{k}_covers"] = bool(lo <= truth <= hi)
        rows.append(row)
    return rows


def run(g_true, pi_L, n_anchor_list, lambdas, deltas, omegas, reps, boot, seed, n_target=2000, mu=1.0, n_jobs=8):
    from joblib import Parallel, delayed
    lg = logit(np.clip(g_true, 1e-6, 1 - 1e-6)); m = float(pi_L @ lg); sd = float(np.sqrt(pi_L @ (lg - m) ** 2))
    sL = (lg - m) / sd
    cells = [(o, d, l, n) for o in omegas for d in deltas for l in lambdas for n in n_anchor_list]
    out = Parallel(n_jobs=n_jobs)(delayed(_cell)(g_true, pi_L, sL, mu, o, d, l, n, reps, boot, seed + 1000 * k, n_target)
                                  for k, (o, d, l, n) in enumerate(cells))
    rows = [r for c in out for r in c]
    df = pd.DataFrame(rows)
    summ = []
    for key, d in df.groupby(["omega", "delta", "lambda", "n_anchor"]):
        for m in LOSSES:
            for k in ("r6", "hard", "soft"):
                e = d[f"{m}_{k}"] - d[f"{m}_true"]
                summ.append({"omega": key[0], "delta": key[1], "lambda": key[2], "n_anchor": key[3], "metric": m, "proxy": k,
                             "true_mean": d[f"{m}_true"].mean(), "bias": e.mean(), "rmse": float(np.sqrt((e ** 2).mean())),
                             "coverage": d[f"{m}_{k}_covers"].mean() if k != "r6" else float("nan"), "reps": len(d)})
    return df, pd.DataFrame(summ)
