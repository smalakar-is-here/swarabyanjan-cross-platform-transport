"""scope.latent: analytic gradient and parameter recovery of the multi-regime cut-point model."""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import check_grad
from scipy.special import expit
from scope import latent


def _data(n=3000, D=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, D)); w = rng.normal(0, 1, D); f = X @ w
    reg = rng.choice([0, 1, 2], n, p=[0.4, 0.4, 0.2])
    tA, tC, th = 0.5, -0.5, np.array([-1.0, 0.3, 1.5])
    y = np.zeros(n)
    m = reg == 0; y[m] = rng.random(m.sum()) < expit(f[m] - tA)
    m = reg == 2; y[m] = rng.random(m.sum()) < expit(f[m] - tC)
    m = reg == 1; u = rng.random(m.sum()); y[m] = (u[:, None] < expit(f[m][:, None] - th[None, :])).sum(1)
    return X, reg, y, w, tA, tC, th


def test_gradient_matches_finite_differences():
    X, reg, y, *_ = _data(n=300, D=5)
    v = np.random.default_rng(1).normal(0, 0.3, X.shape[1] + 5)
    f = lambda v: latent.nll_grad(v, X, reg, y)[0]; g = lambda v: latent.nll_grad(v, X, reg, y)[1]
    assert check_grad(f, g, v) < 1e-4 * max(1.0, np.linalg.norm(g(v)))


def test_recovers_cut_points_and_direction():
    X, reg, y, w, tA, tC, th = _data()
    m = latent.fit(sp.csr_matrix(X), reg, y, lam=1e-3)
    assert m["converged"]
    assert np.corrcoef(m["w"], w)[0, 1] > 0.98
    assert abs(m["tau_A"] - tA) < 0.25 and abs(m["tau_C"] - tC) < 0.3
    assert np.max(np.abs(m["theta"] - th)) < 0.25
    assert np.all(np.diff(m["theta"]) > 0)


def test_missing_regime_C_falls_back_to_source_cut():
    X, reg, y, *_ = _data(n=800)
    keep = reg != 2
    m = latent.fit(X[keep], reg[keep], y[keep])
    assert not m["has_C"] and latent.cut(m, "C") == m["tau_A"]


def test_fit_bias_recovers_regime_intercepts():
    """E23 ablation model: shared weights with one intercept per regime (binary labels)."""
    rng = np.random.default_rng(3); n, D = 4000, 6
    X = rng.normal(0, 1, (n, D)); w = rng.normal(0, 1, D); reg = rng.choice([0, 1, 2], n)
    b = np.array([-0.5, 1.0, 0.2]); y = (rng.random(n) < expit(X @ w + b[reg])).astype(float)
    m = latent.fit_bias(X, reg, y, lam=1e-3)
    assert m["converged"] and np.corrcoef(m["w"], w)[0, 1] > 0.98
    assert np.max(np.abs(m["b"] - b)) < 0.2
    keep = reg != 2; m2 = latent.fit_bias(X[keep], reg[keep], y[keep])
    assert not m2["has_C"] and np.allclose(latent.score_bias(m2, X[:5], np.full(5, 2)), latent.score_bias(m2, X[:5], np.zeros(5, int)))
