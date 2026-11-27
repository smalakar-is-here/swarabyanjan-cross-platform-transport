"""Design-based properties of scope.e21 (model-assisted anchor auditing)."""
import numpy as np
from scipy.special import expit
from scope import e21
from scope import metrics as M


def _pop(N=600, seed=0):
    rng = np.random.default_rng(seed)
    L = rng.integers(0, 4, N); z = rng.normal(0, 1.5, N); p = expit(z)
    y = (rng.random(N) < expit(-1.5 + 0.9 * L + 0.8 * z)).astype(float)
    return L, z, p, y, rng


def test_allocation_sums_to_budget_and_respects_strata():
    L, z, p, y, rng = _pop()
    h = np.full(len(p), 0.4); w = rng.random(len(p))
    for kind in ("prop", "neyman"):
        inv, Nh, nh = e21.allocate(L * 3 + rng.integers(0, 3, len(p)), 60, h, w, kind)
        assert nh.sum() == 60 and (nh >= 2).all() and (nh <= Nh).all()


def test_difference_estimator_unbiased_and_covers_with_a_wrong_link():
    L, z, p, y, rng = _pop()
    h = expit(-0.5 + 0.2 * L - 0.3 * z)  # deliberately misspecified (reversed score slope)
    tau_nl, tau_au = M.nll(y, p), M.auroc(y, p)
    inv, Nh, nh = e21.allocate(L, 80, h, np.ones(len(p)), "prop")
    e_nl, e_au, cov = [], [], []
    for _ in range(1500):
        s, pi = e21.draw(inv, Nh, nh, rng)
        au, se, nl, sen = e21.estimate(y, p, L, z, h, s, pi, inv, Nh, nh)["diff"]
        e_nl.append(nl - tau_nl); e_au.append(au - tau_au); cov.append(abs(nl - tau_nl) <= 1.96 * sen)
    e_nl = np.array(e_nl)
    assert abs(e_nl.mean()) < 4 * e_nl.std() / np.sqrt(len(e_nl))   # exactly unbiased for a linear total
    assert abs(np.mean(e_au)) < 0.01                                 # consistent (ratio), small bias at n = 80
    assert 0.92 <= np.mean(cov) <= 0.98
