"""E1: construct linking between annotation regimes via anchor items.

Design. Regime A (BanglaBait) assigns a binary label Y_A; regime B (BanglaClick) assigns an
ordinal rubric level L in {0,1,2,3}. Items whose normalised title is labelled by both regimes
form an anchor set, which plays the role of the common items in a non-equivalent-groups
anchor-test (NEAT) linking design.

Estimand. The linking function g(l) = P(Y_A = 1 | L = l), estimated on anchors under the
constraint that g is non-decreasing in l (isotonic). Under assumption A-link (g is the same
for every platform annotated by regime B), the source-construct label of any regime-B item is
the soft label g(L). The R6 binarisation corresponds to the step function 1[L >= 1].
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .metrics import kappa


def pava(y, w):
    """Weighted pool-adjacent-violators for a non-decreasing fit."""
    y = list(map(float, y)); w = list(map(float, w)); blocks = [[y[i], w[i], 1] for i in range(len(y))]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            a, b = blocks[i], blocks[i + 1]
            tw = a[1] + b[1]
            blocks[i] = [(a[0] * a[1] + b[0] * b[1]) / tw, tw, a[2] + b[2]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out = []
    for v, _, k in blocks:
        out += [v] * k
    return np.array(out)


def linking_function(anc: pd.DataFrame, levels=(0, 1, 2, 3)) -> np.ndarray:
    n = np.array([(anc.bc_level == l).sum() for l in levels], float)
    k = np.array([anc.loc[anc.bc_level == l, "bb_label"].sum() for l in levels], float)
    raw = (k + 0.5) / (n + 1.0)  # Jeffreys posterior mean, avoids 0/1 at small n
    return pava(raw, n)


def threshold_agreement(anc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in (1, 2, 3):
        b = (anc.bc_level >= t).astype(int)
        rows.append({"threshold": t, "disagreement": float((b != anc.bb_label).mean()),
                     "kappa": kappa(anc.bb_label, b), "bc_prevalence": float(b.mean()),
                     "bb_prevalence": float(anc.bb_label.mean())})
    return pd.DataFrame(rows)


def bootstrap(anc: pd.DataFrame, draws: int, seed: int, cluster: str | None = None) -> dict:
    """Resample anchor title classes (or BanglaClick publisher clusters) with replacement."""
    rng = np.random.default_rng(seed)
    if cluster:
        groups = {c: idx for c, idx in anc.groupby(cluster).indices.items()}
        keys = list(groups)
    G, K = [], {1: [], 2: [], 3: []}
    best = []
    for _ in range(draws):
        if cluster:
            pick = rng.choice(len(keys), len(keys), replace=True)
            idx = np.concatenate([groups[keys[i]] for i in pick])
        else:
            idx = rng.integers(0, len(anc), len(anc))
        s = anc.iloc[idx]
        G.append(linking_function(s))
        ks = {t: kappa(s.bb_label, (s.bc_level >= t).astype(int)) for t in (1, 2, 3)}
        for t in ks:
            K[t].append(ks[t])
        best.append(max(ks, key=ks.get))
    G = np.array(G)
    q = lambda a: (float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5)))
    return {"g_ci": [q(G[:, l]) for l in range(4)], "kappa_ci": {t: q(K[t]) for t in K},
            "kappa_diff_2_minus_1_ci": q(np.array(K[2]) - np.array(K[1])),
            "argmax_threshold_freq": {t: float(np.mean(np.array(best) == t)) for t in (1, 2, 3)}}


def residual_disagreement(anc: pd.DataFrame) -> pd.DataFrame:
    """Where do the regimes still disagree after the level-2 cut? Break down by FAST dimension."""
    a = anc.copy()
    dims = {"f": "factual", "a": "attribution", "s": "specificity", "t": "tone"}
    for k, v in dims.items():
        a[f"fail_{v}"] = a.fast_fail.fillna("").str.contains(k).astype(int)
    rows = []
    for v in dims.values():
        for val in (0, 1):
            m = a[f"fail_{v}"] == val
            if m.sum():
                rows.append({"dimension": v, "failed": val, "n": int(m.sum()),
                             "p_bb_clickbait": float(a.loc[m, "bb_label"].mean())})
    for tl in sorted(a.trigger_level.dropna().unique()):
        m = a.trigger_level == tl
        rows.append({"dimension": "trigger_level", "failed": float(tl), "n": int(m.sum()),
                     "p_bb_clickbait": float(a.loc[m, "bb_label"].mean())})
    return pd.DataFrame(rows)


def logistic_dimension_model(anc: pd.DataFrame) -> pd.DataFrame:
    """Which rubric components predict the source label? Logistic regression with HC1 errors."""
    import statsmodels.api as sm
    a = anc.copy()
    for k, v in {"f": "factual", "a": "attribution", "s": "specificity", "t": "tone"}.items():
        a[f"fail_{v}"] = a.fast_fail.fillna("").str.contains(k).astype(int)
    X = sm.add_constant(a[["fail_factual", "fail_attribution", "fail_specificity", "fail_tone", "trigger_level"]].astype(float))
    m = sm.Logit(a.bb_label.astype(float), X).fit(disp=0, cov_type="HC1")
    ci = m.conf_int()
    return pd.DataFrame({"term": m.params.index, "log_odds": m.params.values, "se": m.bse.values,
                         "ci_low": ci[0].values, "ci_high": ci[1].values, "p": m.pvalues.values})
