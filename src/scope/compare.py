"""E17: paired cluster-bootstrap comparison of training protocols on the LOEO splits.

For one (model, label, target) cell, every protocol's per-item predictions (all seeds) are scored on the
same bootstrap resample of target clusters (publisher/channel), so differences are paired. The metric of
a protocol in a draw is its mean over seeds. Contrast: protocol minus the reference protocol
(source_only, else pooled_erm). Reported: point difference, 95% percentile interval, percentile p-value
2 min(P(d <= 0), P(d >= 0)), bootstrap SE and the Wald p-value diff/SE. Holm adjustment (run.py) uses the
Wald p-value, whose resolution does not depend on the number of draws. Draws where a metric is undefined
(one-class resample) are dropped. Cells with fewer than 40 clusters are flagged: E13 shows the pairs
cluster bootstrap over-rejects with few clusters.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm
from . import metrics as M

METRICS = {"auroc": M.auroc, "nll": M.nll, "brier": M.brier}


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (m - k) * p[i])); adj[i] = run
    return adj


def wald_p(diff: float, d: np.ndarray) -> float:
    """Two-sided p-value of diff / (bootstrap SE). Unlike the percentile p-value, its resolution does not
    depend on the number of draws, which matters once Holm divides the level by about 70 contrasts."""
    if len(d) < 2:
        return np.nan
    se = np.std(d, ddof=1)
    if se == 0:
        return 1.0 if diff == 0 else 0.0
    return float(2 * norm.sf(abs(diff) / se))


def cell(preds: dict, y: np.ndarray, clusters: np.ndarray, ref: str, draws: int, rng) -> pd.DataFrame:
    """preds: protocol -> list of per-seed probability arrays aligned with y."""
    uc, inv = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uc))]
    point = {pr: {m: np.mean([f(y, p) for p in ps]) for m, f in METRICS.items()} for pr, ps in preds.items()}
    boot = {pr: {m: np.empty(draws) for m in METRICS} for pr in preds}
    for b in range(draws):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(uc), len(uc))])
        yb = y[idx]
        for pr, ps in preds.items():
            for m, f in METRICS.items():
                boot[pr][m][b] = np.mean([f(yb, p[idx]) for p in ps])
    rows = []
    for pr in preds:
        if pr == ref:
            continue
        for m in METRICS:
            d = boot[pr][m] - boot[ref][m]
            # A resample can contain one class only (BaitBuster labels are channel labels), where AUROC is
            # undefined. Such draws are dropped; n_valid_draws reports how many remain.
            d = d[np.isfinite(d)]
            if len(d) == 0:
                lo = hi = np.nan
            else:
                lo, hi = np.quantile(d, [0.025, 0.975])
            rows.append({"protocol": pr, "reference": ref, "metric": m, "diff": point[pr][m] - point[ref][m],
                         "ci_low": lo, "ci_high": hi, "p_boot": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))) if len(d) else np.nan,
                         "se_boot": float(np.std(d, ddof=1)) if len(d) > 1 else np.nan,
                         "p_wald": wald_p(point[pr][m] - point[ref][m], d),
                         "n_clusters": len(uc), "few_clusters": len(uc) < 40, "seeds": len(preds[pr]),
                         "n_valid_draws": int(len(d))})
    return pd.DataFrame(rows)
