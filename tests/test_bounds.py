"""Mathematical properties of scope.bounds (E20 partial identification)."""
import numpy as np
from scipy.stats import rankdata
from scope import bounds as B
from scope import metrics as M


def _cell(n=400, seed=0, ties=False):
    rng = np.random.default_rng(seed)
    L = rng.integers(0, 4, n)
    p = rng.beta(2, 2, n)
    if ties:
        p = np.round(p, 1)
    g = np.array([0.1, 0.3, 0.6, 0.85])
    return L, p, g, rng


def _random_feasible(L, p, g, rng, monotone=False):
    """Random soft labels with level sums g(l) n_l, optionally non-decreasing in p within each level."""
    q = np.zeros(len(p))
    for l in range(4):
        idx = np.flatnonzero(L == l); m = g[l] * len(idx)
        if monotone:
            o = idx[np.argsort(p[idx])]
            # mixture of a constant and a top step: non-decreasing, correct mean
            w = rng.random()
            step = B.sorted_assignment(np.zeros(len(o), int), p[o], [m, 0, 0, 0], True)
            v = w * step + (1 - w) * g[l]
            # ties must share a value to be a function of p
            _, inv = np.unique(p[o], return_inverse=True)
            v = (np.bincount(inv, v) / np.bincount(inv))[inv]
            q[o] = v
        else:
            v = rng.random(len(idx)); v = v / v.sum() * m
            while v.max() > 1:  # project onto [0,1] keeping the sum
                ex = (v[v > 1] - 1).sum(); v[v > 1] = 1; free = v < 1; v[free] += ex * (1 - v[free]) / (1 - v[free]).sum()
            q[idx] = v
    return q


def test_auroc_linear_identity_matches_metric():
    for ties in (False, True):
        L, p, g, rng = _cell(ties=ties)
        q = rng.random(len(p)); S = q.sum(); r = rankdata(p) - 0.5
        lin = ((q * r).sum() - S * S / 2) / (S * (len(p) - S))
        assert abs(lin - M.auroc(q, p)) < 1e-10


def test_sorted_assignment_respects_masses_and_bounds():
    L, p, g, _ = _cell(ties=True)
    m = B.level_masses(L, g)
    for top in (True, False):
        q = B.sorted_assignment(L, p, m, top)
        assert q.min() >= 0 and q.max() <= 1 + 1e-12
        for l in range(4):
            assert abs(q[L == l].sum() - m[l]) < 1e-9


def test_feasible_labels_lie_inside_AL_and_monotone_inside_ALM():
    for seed in range(5):
        L, p, g, rng = _cell(seed=seed, ties=seed % 2 == 1)
        s = B.identified_sets(L, p, g)
        for _ in range(30):
            q = _random_feasible(L, p, g, rng)
            qm = _random_feasible(L, p, g, rng, monotone=True)
            for name, f in (("nll", M.nll), ("brier", M.brier), ("auroc", M.auroc)):
                v, vm = f(q, p), f(qm, p)
                assert s[f"{name}_AL_lo"] - 1e-9 <= v <= s[f"{name}_AL_hi"] + 1e-9
                assert s[f"{name}_ALM_lo"] - 1e-9 <= vm <= s[f"{name}_ALM_hi"] + 1e-9
                assert s[f"{name}_A0_lo"] - 1e-9 <= v <= s[f"{name}_A0_hi"] + 1e-9 if name != "auroc" else True


def test_bounds_are_attained():
    L, p, g, _ = _cell()
    s = B.identified_sets(L, p, g); m = B.level_masses(L, g)
    qt = B.sorted_assignment(L, p, m, True)
    assert abs(M.auroc(qt, p) - s["auroc_ALM_hi"]) < 1e-12
    assert abs(M.auroc(g[L], p) - s["auroc_ALM_lo"]) < 1e-12
    assert abs(M.nll(qt, p) - s["nll_ALM_lo"]) < 1e-12


def test_relaxed_bounds_reduce_to_exact_at_zero_and_widen():
    L, p, g, _ = _cell(seed=3)
    s = B.identified_sets(L, p, g)
    prev = None
    for dl in (0.0, 0.5, 1.0):
        for metric in ("nll", "brier", "auroc"):
            b = B.relaxed_bounds(L, p, g, dl, metric)
            if dl == 0:
                for k in ("AL_lo", "AL_hi", "ALM_lo", "ALM_hi"):
                    assert abs(b[f"lo_{k[:-3]}" if k.endswith("lo") else f"hi_{k[:-3]}"] - s[f"{metric}_{k}"]) < 1e-9, (metric, k)
        b = B.relaxed_bounds(L, p, g, dl, "auroc")
        if prev is not None:
            assert b["lo_ALM"] <= prev["lo_ALM"] + 1e-12 and b["hi_ALM"] >= prev["hi_ALM"] - 1e-12
        prev = b


def test_relaxed_linear_loss_bounds_contain_feasible_labels_at_perturbed_link():
    L, p, g, rng = _cell(seed=7)
    from scipy.special import expit, logit
    dl = 0.6; b = B.relaxed_bounds(L, p, g, dl, "nll")
    for _ in range(50):
        g2 = expit(logit(g) + rng.uniform(-dl, dl, 4))
        q = _random_feasible(L, p, g2, rng)
        assert b["lo_AL"] - 1e-9 <= M.nll(q, p) <= b["hi_AL"] + 1e-9
