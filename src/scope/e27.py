"""E27: analysis of the source-guideline human audit (docs/scope/ANNOTATION_PROTOCOL.md).

Inputs: the private key (scripts/scope_audit_sample.py), the three returned annotator sheets, and anchor-blind
scores. Outputs: annotator agreement, agreement of the new majority label with the BanglaBait label on gold
anchors, and per block x scorer design-based estimates of source-construct AUROC and log loss (difference
estimator, paper Eq. 8; working link = flexible score-conditional link fitted on all news anchors).
Nothing here is run on simulated labels except in tests.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import linking, e21
from . import metrics as M


def fleiss_kappa(labels: np.ndarray) -> float:
    """labels: items x raters, binary (NaN = missing rating; items with < 2 ratings dropped)."""
    L = labels[np.sum(~np.isnan(labels), axis=1) >= 2]
    n = np.sum(~np.isnan(L), axis=1); ones = np.nansum(L, axis=1); zeros = n - ones
    P_i = (ones * (ones - 1) + zeros * (zeros - 1)) / (n * (n - 1))
    p1 = ones.sum() / n.sum(); Pe = p1 ** 2 + (1 - p1) ** 2
    return float((P_i.mean() - Pe) / (1 - Pe))


def cohen_kappa(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float); po = np.mean(a == b)
    pe = np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b))
    return float((po - pe) / (1 - pe)) if pe < 1 else np.nan


def majority(sheets: list[pd.DataFrame]) -> pd.DataFrame:
    """sheets: per annotator, columns audit_id, label_0_1. Returns audit_id, r1..rk, n_rated, y_major (ties -> NaN)."""
    m = None
    for k, s in enumerate(sheets, 1):
        x = s[["audit_id", "label_0_1"]].rename(columns={"label_0_1": f"r{k}"})
        x[f"r{k}"] = pd.to_numeric(x[f"r{k}"], errors="coerce")
        m = x if m is None else m.merge(x, on="audit_id", how="outer")
    R = m[[c for c in m.columns if c.startswith("r")]].to_numpy(float)
    m["n_rated"] = np.sum(~np.isnan(R), axis=1)
    s1 = np.nansum(R, axis=1); m["y_major"] = np.where(s1 * 2 > m.n_rated, 1.0, np.where(s1 * 2 < m.n_rated, 0.0, np.nan))
    return m


def block_estimate(pop: pd.DataFrame, sample: pd.DataFrame, link) -> dict:
    """pop: all rows of the block's population with row_id, L, p. sample: row_id, level, n_h, N_h, y (majority).
    Difference estimator under stratified SRS by level (strata and allocation from the key)."""
    pop = pop.reset_index(drop=True)
    L, p = pop.L.to_numpy().astype(int), pop.p.to_numpy(); z = linking.logit(p); h = linking.predict(link, L, z)
    levels = sorted(sample.level.unique()); inv = pd.Series(L).map({lv: i for i, lv in enumerate(levels)}).to_numpy()
    if np.isnan(inv.astype(float)).any():
        raise ValueError("population has a level absent from the sample strata")
    inv = inv.astype(int)
    Nh = np.array([int((L == lv).sum()) for lv in levels]); nh = np.array([int((sample.level == lv).sum()) for lv in levels])
    pos = pd.Series(np.arange(len(pop)), index=pop.row_id)
    idx = pos.loc[sample.row_id].to_numpy()
    s = np.zeros(len(pop), bool); s[idx] = True
    y = np.full(len(pop), np.nan); y[idx] = sample.y.to_numpy()
    pi = (nh / Nh)[inv]
    y_ = np.where(s, y, 0.0)
    out = e21.estimate(y_, p, L, z, h, s, pi, inv, Nh, nh)
    au, se, nl, sen = out["diff"]
    return {"N": len(pop), "n": int(s.sum()), "auroc": au, "auroc_se": se, "auroc_lo": au - 1.96 * se, "auroc_hi": au + 1.96 * se,
            "nll": nl, "nll_se": sen, "auroc_model_only": out["model_only"][0], "auroc_sample_only": out["sample_only"][0],
            "prev_hat": float((h.sum() + np.sum((y_[s] - h[s]) / pi[s])) / len(pop))}
