"""E11: budgeted target audit -- how many target labels are needed before deployment?

Setting. A source-trained, prior-corrected model f (MLLS prevalence estimate, no target labels)
is to be deployed on an unlabelled target pool of N items. A budget of n target labels can be
bought. The audit estimates the target Brier risk theta = E[(f(x) - y)^2] and the target
prevalence, and decides DEPLOY if the upper confidence bound on theta is below a tolerance
tau (source Brier + delta), otherwise RELABEL.

Estimators (means with a model-based proxy; prediction-powered inference, Angelopoulos et al.,
2023; active inference, Zrnic & Candes, 2024):
  classical   mean of observed losses on a simple random sample (SRS)
  stratified  SRS within score quintiles, proportional allocation
  ppi         proxy mean over all N plus the SRS mean of (loss - proxy); proxy = p(1-p),
              the expected Brier loss if f is calibrated
  active_ppi  inclusion probability proportional to sqrt(proxy) mixed 50/50 with uniform,
              inverse-probability-weighted correction term
All intervals are normal-approximation 95% intervals. Ground truth uses all target labels.
The audit resamples REAL labelled target items; no observation is simulated.
"""
from __future__ import annotations
import numpy as np

Z = 1.959964


def _ci(est, var):
    s = np.sqrt(max(var, 0.0))
    return est, est - Z * s, est + Z * s


def classical(loss, idx):
    l = loss[idx]; n = len(l)
    return _ci(l.mean(), l.var(ddof=1) / n)


def stratified(loss, strata, idx_by_stratum, N):
    est, var = 0.0, 0.0
    for s, idx in idx_by_stratum.items():
        Ns = (strata == s).sum(); l = loss[idx]
        if len(l) == 0:
            continue
        w = Ns / N; est += w * l.mean()
        var += w * w * (l.var(ddof=1) / len(l) if len(l) > 1 else 0.0)
    return _ci(est, var)


def ppi(loss, proxy, idx):
    d = loss[idx] - proxy[idx]; n, N = len(d), len(proxy)
    est = proxy.mean() + d.mean()
    return _ci(est, d.var(ddof=1) / n + proxy.var(ddof=1) / N)


def active_ppi(loss, proxy, n, rng):
    N = len(proxy); u = np.sqrt(np.clip(proxy, 1e-6, None)); u = u / u.sum()
    pi = 0.5 * u + 0.5 / N  # per-draw selection probability (with replacement)
    idx = rng.choice(N, n, replace=True, p=pi)
    z = (loss[idx] - proxy[idx]) / (N * pi[idx])
    est = proxy.mean() + z.mean()
    return _ci(est, z.var(ddof=1) / n)


def run_audit(p, y, tau, budgets, reps, seed, true_theta=None):
    """Return a list of dict rows: one per (budget, method) with error/coverage/decision summary."""
    rng = np.random.default_rng(seed)
    p = np.asarray(p, float); y = np.asarray(y, float)
    loss = (p - y) ** 2; proxy = p * (1 - p)
    N = len(p); theta = loss.mean() if true_theta is None else true_theta
    oracle_deploy = theta <= tau
    q = np.quantile(p, [0.2, 0.4, 0.6, 0.8]); strata = np.digitize(p, q)
    rows = []
    for n in budgets:
        res = {m: [] for m in ("classical", "stratified", "ppi", "active_ppi")}
        for _ in range(reps):
            idx = rng.choice(N, n, replace=False)
            res["classical"].append(classical(loss, idx))
            res["ppi"].append(ppi(loss, proxy, idx))
            ibs = {}
            for s in range(5):
                pool = np.flatnonzero(strata == s); k = max(2, int(round(n * len(pool) / N)))
                ibs[s] = rng.choice(pool, min(k, len(pool)), replace=False)
            res["stratified"].append(stratified(loss, strata, ibs, N))
            res["active_ppi"].append(active_ppi(loss, proxy, n, rng))
        for m, r in res.items():
            r = np.array(r); est, lo, hi = r[:, 0], r[:, 1], r[:, 2]
            deploy = hi <= tau
            regret = np.where(deploy != oracle_deploy, abs(theta - tau), 0.0)
            rows.append({"budget": n, "method": m, "theta_true": theta, "tau": tau,
                         "oracle_decision": "DEPLOY" if oracle_deploy else "RELABEL",
                         "mae": float(np.mean(np.abs(est - theta))), "rmse": float(np.sqrt(np.mean((est - theta) ** 2))),
                         "coverage": float(np.mean((lo <= theta) & (theta <= hi))),
                         "mean_width": float(np.mean(hi - lo)),
                         "decision_accuracy": float(np.mean(deploy == oracle_deploy)),
                         "p_deploy": float(np.mean(deploy)), "mean_regret": float(np.mean(regret))})
    return rows
