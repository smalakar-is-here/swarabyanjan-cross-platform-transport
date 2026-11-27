"""E2/E6: cue-label association transport across a crossed platform x regime design.

Cells: (news,A) (news,B) (facebook,B) (youtube,B) (twitter,B) (youtube,C).
For cue c (presence = count > 0) and label definition Y, the cell association is the
log odds ratio  beta_e = log OR(Y, 1[c>0])  (Haldane-corrected), which, unlike the risk
difference, is invariant to label (prior) shift when P(c|Y) is fixed.

Identification. The design is connected but sparse:
  regime B - regime A  is identified only on news (anchor platform),
  regime C - regime B  is identified only on YouTube,
  platform effects     are identified only within regime B.
An additive model beta_e = mu + platform(e) + regime(e) is therefore just-identified for the
regime terms; a platform x regime interaction is NOT identifiable from these cells.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

CELL_DESIGN = {
    "banglabait:news": ("news", "A"), "banglaclick:news": ("news", "B"),
    "banglaclick:facebook": ("facebook", "B"), "banglaclick:youtube": ("youtube", "B"),
    "banglaclick:twitter": ("twitter", "B"), "baitbuster:youtube": ("youtube", "C"),
}


def log_or(y, c) -> float:
    y = np.asarray(y).astype(int); c = np.asarray(c).astype(int)
    a = ((c == 1) & (y == 1)).sum() + .5; b = ((c == 1) & (y == 0)).sum() + .5
    d = ((c == 0) & (y == 1)).sum() + .5; e = ((c == 0) & (y == 0)).sum() + .5
    return float(np.log(a * e / (b * d)))


def risk_diff(y, c) -> float:
    y = np.asarray(y, float); c = np.asarray(c).astype(bool)
    if c.all() or (~c).all():
        return float("nan")
    return float(y[c].mean() - y[~c].mean())


def cluster_boot(items: pd.DataFrame, ycol: str, cue: str, draws: int, rng) -> np.ndarray:
    groups = list(items.groupby("cluster_id").indices.values())
    G = len(groups); out = np.empty((draws, 2))
    y = items[ycol].to_numpy(); c = (items[cue].to_numpy() > 0)
    for b in range(draws):
        idx = np.concatenate([groups[i] for i in rng.integers(0, G, G)])
        out[b] = log_or(y[idx], c[idx]), risk_diff(y[idx], c[idx])
    return out


def wild_cluster_rd_pvalue(y, x, cl, draws: int, rng) -> tuple[float, float]:
    """WCR bootstrap-t (Rademacher, null imposed) for H0: slope=0 in y = a + b x + u, CR1 errors.

    Cameron, Gelbach & Miller (2008); recommended for few clusters (MacKinnon & Webb, 2017).
    Returns (estimate, p-value).
    """
    y = np.asarray(y, float); x = np.asarray(x, float); cl = pd.factorize(np.asarray(cl))[0]
    G = cl.max() + 1; X = np.column_stack([np.ones_like(x), x])
    XtX_inv = np.linalg.inv(X.T @ X)

    def t_stat(yy):
        beta = XtX_inv @ X.T @ yy; u = yy - X @ beta
        S = np.zeros((2, 2))
        sc = np.zeros((G, 2))
        np.add.at(sc, cl, X * u[:, None])
        S = sc.T @ sc
        n, k = len(yy), 2
        V = XtX_inv @ S @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - k))
        return beta[1], beta[1] / np.sqrt(V[1, 1]) if V[1, 1] > 0 else np.nan
    b_hat, t_hat = t_stat(y)
    y0_fit = np.full_like(y, y.mean()); u0 = y - y0_fit  # restricted model: intercept only
    ts = np.empty(draws)
    for d in range(draws):
        w = rng.choice([-1.0, 1.0], G)[cl]
        ts[d] = t_stat(y0_fit + u0 * w)[1]
    p = float(np.mean(np.abs(ts) >= abs(t_hat)))
    return float(b_hat), p


def crossed_effects(cell_beta: dict) -> dict:
    """Just-identified contrasts from cell log-ORs (keys as in CELL_DESIGN)."""
    b = cell_beta
    out = {"regime_B_minus_A_on_news": b["banglaclick:news"] - b["banglabait:news"],
           "regime_C_minus_B_on_youtube": b["baitbuster:youtube"] - b["banglaclick:youtube"]}
    for p in ("facebook", "youtube", "twitter"):
        out[f"platform_{p}_minus_news_in_B"] = b[f"banglaclick:{p}"] - b["banglaclick:news"]
    vals = [b[f"banglaclick:{p}"] for p in ("news", "facebook", "youtube", "twitter")]
    out["platform_sd_in_B"] = float(np.std(vals, ddof=1))
    return out
