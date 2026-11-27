"""E20: partial identification of source-construct risk on cells without anchors.

Setting. Cell e (a regime-B platform) has, for every item, the rubric level L in {0,..,3} and the score p = f(x)
of a frozen, anchor-blind scorer. The source-construct label Y_A is unobserved. The estimand is
    R*_e(f) = E_e[ l(p, h_e(L, p)) ],   h_e(l, s) = P_e(Y_A = 1 | L = l, f = s),
with l the soft-label metric of scope.metrics (log loss, expected Brier, weighted AUROC).

Key fact (linearity). For a finite cell with soft labels q_i = h_e(L_i, p_i):
  * log loss and Brier are linear in q:  l_i(q_i) = b(p_i) + q_i d(p_i),  d decreasing in p
      (log loss d = log((1-p)/p); expected Brier d = 1 - 2p, the q(1-q) term included: (p-q)^2 + q(1-q) = p^2 + q(1-2p));
  * weighted AUROC (ties and self-pairs counted 1/2, as scope.metrics.auroc) equals
        AUROC(q) = ( sum_i q_i (r_i - 1/2) - S^2 / 2 ) / ( S (n - S) ),   S = sum_i q_i,
    r_i the mid-rank of p_i: linear in q once S is fixed.

Assumptions (nested).
  (A0) none: q_i in [0, 1].
  (AL) level-link transport: E_e[h_e | L = l] = g(l), the anchor-estimated link P(Y_A = 1 | L = l). Fixes the
       per-level positive mass m_l = g(l) n_l and hence S.
  (AL+M) additionally, within a level h_e is non-decreasing in the score (the scorer's within-level signal about
       Y_A is not reversed; testable on anchors, where it is estimated).

Proposition (sharp bounds). Under AL the identified set of each metric is the interval between its values at
  q_top (within every level, mass m_l on the highest-scoring items: fractional knapsack) and
  q_bottom (mass on the lowest-scoring items).
Under AL+M it is the interval between q_top and q_const = g(L) (soft linking; Chebyshev's sum inequality: a
non-decreasing q with fixed level means has non-negative covariance with any non-decreasing score function, zero
iff constant). q_top is non-decreasing, so it remains attainable. Both are attained, hence sharp. The proof needs
only linearity in q with a monotone coefficient, which holds for all three metrics.

Relaxing AL. Allow logit g_e(l) in [logit g(l) - delta, logit g(l) + delta]. For each delta the outer bounds are
taken over the per-level masses (exact per level for log loss / Brier, which separate across levels; grid search
for AUROC, whose denominator couples levels). The breakdown value delta* is the smallest delta at which a stated
conclusion (e.g. "source-construct AUROC on e is below the source AUROC") is no longer implied.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import rankdata
from scipy.special import expit, logit as _logit

LEVELS = 4
EPS = 1e-12


def _d_coef(p, metric):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    if metric == "nll":
        return np.log(1 - p) - np.log(p), -np.log(1 - p)   # l = b + q d
    if metric == "brier":
        return 1 - 2 * p, p ** 2
    raise ValueError(metric)


def sorted_assignment(L, p, masses, top=True):
    """Soft labels q with sum over level l equal to masses[l], placed on the highest (top) or lowest scores of the
    level; ties in p share mass equally so the assignment is a function of (L, p)."""
    L = np.asarray(L, int); p = np.asarray(p, float); q = np.zeros(len(p))
    for l in range(LEVELS):
        idx = np.flatnonzero(L == l)
        if len(idx) == 0:
            continue
        m = float(np.clip(masses[l], 0, len(idx)))
        vals, inv = np.unique(p[idx], return_inverse=True)
        cnt = np.bincount(inv).astype(float)
        order = np.arange(len(vals))[::-1] if top else np.arange(len(vals))
        fill = np.zeros(len(vals)); left = m
        for v in order:
            take = min(cnt[v], left); fill[v] = take / cnt[v]; left -= take
            if left <= 0:
                break
        q[idx] = fill[inv]
    return q


def level_masses(L, g):
    L = np.asarray(L, int)
    return np.array([g[l] * np.sum(L == l) for l in range(LEVELS)], float)


def identified_sets(L, p, g):
    """Interval endpoints of log loss, Brier and AUROC under A0, AL and AL+M. Returns a flat dict."""
    from . import metrics as M
    L = np.asarray(L, int); p = np.asarray(p, float)
    m = level_masses(L, g)
    qt, qb, qc = sorted_assignment(L, p, m, True), sorted_assignment(L, p, m, False), np.asarray(g, float)[L]
    out = {}
    for name, f in (("nll", M.nll), ("brier", M.brier), ("auroc", M.auroc)):
        top, bot, const = f(qt, p), f(qb, p), f(qc, p)
        lo_al, hi_al = sorted((top, bot)); lo_m, hi_m = sorted((top, const))
        out.update({f"{name}_AL_lo": lo_al, f"{name}_AL_hi": hi_al, f"{name}_ALM_lo": lo_m, f"{name}_ALM_hi": hi_m,
                    f"{name}_const": const})
    # A0 (no assumption): per-item extremes for the linear losses; AUROC is [0, 1] up to ties.
    for name in ("nll", "brier"):
        d, b = _d_coef(p, name)
        out[f"{name}_A0_lo"] = float(np.mean(b + np.minimum(d, 0)))
        out[f"{name}_A0_hi"] = float(np.mean(b + np.maximum(d, 0)))
    out["prevalence_AL"] = float(m.sum() / len(p))
    return out


# ---------------------------------------------------------------- sensitivity: relax AL by delta on the logit scale
def _level_tables(L, p):
    """Per level: scores sorted descending, and the AUROC coefficient (r - 1/2) in the same order (cell-wide ranks)."""
    r = rankdata(p) - 0.5
    tabs = []
    for l in range(LEVELS):
        idx = np.flatnonzero(np.asarray(L) == l)
        o = idx[np.argsort(-p[idx], kind="stable")]
        tabs.append((p[o], r[o]))
    return tabs


def _topmass(vals, m, top=True):
    """Sum of the m largest (top) or smallest values with linear interpolation for fractional m (ties ignored:
    a tie only changes which equal-valued item takes the fraction, not the sum)."""
    v = vals if top else vals[::-1]
    k = int(np.floor(m)); fr = m - k
    s = v[:k].sum()
    if fr > 0 and k < len(v):
        s += fr * v[k]
    return s


def relaxed_bounds(L, p, g, delta, metric, grid=9):
    """Outer bounds of a metric under AL relaxed by +-delta on logit g(l), with M (lower AUROC / upper loss
    from the constant link at the worst masses) or without M (sorted extremes).

    Returns dict(lo_AL, hi_AL, lo_ALM, hi_ALM). Exact over the mass box for nll/brier (separable, each level's
    optimum is at an endpoint or at a kink, which the endpoints and the zero-crossing of d dominate); grid over
    the box for AUROC."""
    from . import metrics as M
    L = np.asarray(L, int); p = np.asarray(p, float); n_l = np.array([np.sum(L == l) for l in range(LEVELS)], float)
    g = np.clip(np.asarray(g, float), 1e-6, 1 - 1e-6)
    lo_g, hi_g = expit(_logit(g) - delta), expit(_logit(g) + delta)
    if metric in ("nll", "brier"):
        d, b = _d_coef(p, metric); base = float(np.mean(b))
        lo_al = hi_al = lo_m = hi_m = base * len(p)
        for l in range(LEVELS):
            idx = np.flatnonzero(L == l)
            if len(idx) == 0:
                continue
            ds = np.sort(d[idx])  # ascending: smallest d = highest p
            # candidate masses: interval endpoints, and the number of items with d < 0 (min) / d > 0 (max) clipped
            mlo, mhi = lo_g[l] * n_l[l], hi_g[l] * n_l[l]
            cand = [mlo, mhi, float(np.clip(np.sum(ds < 0), mlo, mhi)), float(np.clip(np.sum(ds > 0), mlo, mhi))]
            lo_al += min(_topmass(ds, m, True) for m in cand)          # mass on smallest d
            hi_al += max(_topmass(ds[::-1], m, True) for m in cand)    # mass on largest d
            # with M, the worst case is the constant link: linear in its mass, extreme at an endpoint
            hi_m += max(m / len(idx) * ds.sum() for m in (mlo, mhi))
        lo_m = lo_al
        n = len(p)
        return {"lo_AL": lo_al / n, "hi_AL": hi_al / n, "lo_ALM": lo_m / n, "hi_ALM": hi_m / n}
    tabs = _level_tables(L, p); n = len(p)
    axes = [np.linspace(lo_g[l] * n_l[l], hi_g[l] * n_l[l], grid) for l in range(LEVELS)]
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, LEVELS)
    best_hi, best_lo = -np.inf, np.inf; best_lo_m = np.inf
    for ms in mesh:
        S = ms.sum()
        if S <= 0 or S >= n:
            continue
        den = S * (n - S)
        top = sum(_topmass(t[1], ms[l], True) for l, t in enumerate(tabs))
        bot = sum(_topmass(t[1], ms[l], False) for l, t in enumerate(tabs))
        const = sum(ms[l] / max(len(t[1]), 1) * t[1].sum() for l, t in enumerate(tabs))
        best_hi = max(best_hi, (top - S * S / 2) / den)
        best_lo = min(best_lo, (bot - S * S / 2) / den)
        best_lo_m = min(best_lo_m, (const - S * S / 2) / den)
    return {"lo_AL": best_lo, "hi_AL": best_hi, "lo_ALM": best_lo_m, "hi_ALM": best_hi}


def breakdown_delta(L, p, g, metric, threshold, side, grid=9, deltas=np.linspace(0, 3, 61)):
    """Smallest delta at which the ALM identified set reaches `threshold` (side='below': the set's upper end
    reaches the threshold from below; side='above': its lower end reaches it from above). inf if never within
    the scanned range; 0 if already reached at delta = 0."""
    for dl in deltas:
        b = relaxed_bounds(L, p, g, dl, metric, grid)
        if (side == "below" and b["hi_ALM"] >= threshold) or (side == "above" and b["lo_ALM"] <= threshold):
            return float(dl)
    return float("inf")
