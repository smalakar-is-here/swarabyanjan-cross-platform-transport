"""Mathematical checks for score-conditional linking (scope.linking). Synthetic data only; no private inputs."""
import numpy as np
import statsmodels.api as sm
from scipy.special import expit, logit
from scope import linking, metrics as M


def _sim(n, rng, alpha=(-2.5, -1.0, 0.6, 1.5), beta=0.8):
    L = rng.choice(4, n, p=[0.1, 0.35, 0.35, 0.2]); z = rng.normal(0.3 * L, 1.0)
    ya = (rng.random(n) < expit(np.array(alpha)[L] + beta * z)).astype(float)
    return L, z, ya


def test_full_fit_matches_statsmodels_mle():
    rng = np.random.default_rng(1); L, z, ya = _sim(4000, rng)
    X = np.column_stack([np.eye(4)[L], z])
    ref = sm.Logit(ya, X).fit(disp=0).params
    f = linking.fit(L, z, ya, gamma=1.0)
    assert np.allclose(np.r_[f["alpha"], f["beta_hat"]], ref, atol=2e-3)


def test_gamma_zero_alpha_is_level_logit_frequency():
    rng = np.random.default_rng(2); L, z, ya = _sim(3000, rng)
    f = linking.fit(L, z, ya, gamma=0.0)
    freq = np.array([ya[L == l].mean() for l in range(4)])
    assert np.allclose(expit(f["alpha"]), freq, atol=1e-3)
    # intercept score equations hold on the fitting data at every gamma (calibration-in-the-large by level)
    for gm in (0.0, 0.5, 1.0):
        h = linking.predict(linking.fit(L, z, ya, gm), L, z)
        for l in range(4):
            assert abs(h[L == l].mean() - ya[L == l].mean()) < 2e-3


def test_tower_rule_soft_risk_equals_true_risk():
    """With the true conditional link, soft-label risk estimates the risk against Y_A (any scorer p = f(z))."""
    rng = np.random.default_rng(3); alpha = np.array([-2.5, -1.0, 0.6, 1.5]); beta = 0.8
    L, z, ya = _sim(400000, rng, alpha, beta)
    h = expit(alpha[L] + beta * z); p = expit(0.7 * z - 0.2)
    for fn in (M.nll, M.brier, M.auroc):
        assert abs(fn(h, p) - fn(ya, p)) < 3e-3


def test_score_selected_anchors_bias_level_link_not_conditional_link():
    """Assumption 1 fails (selection on z), Assumption 1' holds: level-only link is biased, SCL is not."""
    rng = np.random.default_rng(4); L, z, ya = _sim(200000, rng)
    sel = rng.random(len(L)) < expit(1.5 * (z - z.mean()))
    p = expit(z)
    truth = M.auroc(ya[~sel], p[~sel]), M.nll(ya[~sel], p[~sel])
    f1 = linking.fit(L[sel], z[sel], ya[sel], 1.0); f0 = linking.fit(L[sel], z[sel], ya[sel], 0.0, beta=f1["beta_hat"])
    q1 = linking.predict(f1, L[~sel], z[~sel]); q0 = linking.predict(f0, L[~sel], z[~sel])
    assert abs(M.nll(q1, p[~sel]) - truth[1]) < 5e-3 and abs(M.auroc(q1, p[~sel]) - truth[0]) < 5e-3
    assert abs(M.nll(q0, p[~sel]) - truth[1]) > 3e-2  # level-only link is visibly biased
    assert abs(np.mean(q1) - ya[~sel].mean()) < 5e-3 < abs(np.mean(q0) - ya[~sel].mean())


def test_cell_labels_use_observed_source_label_where_available():
    f = {"alpha": np.zeros(4), "beta_hat": 0.0, "score": {"coef": np.array([0.0]), "knots": np.array([])}, "gamma": 1.0}
    q = linking.cell_labels(f, np.array([0, 1, 2]), np.zeros(3), np.array([1.0, np.nan, 0.0]))
    assert np.allclose(q, [1.0, 0.5, 0.0])


def test_logit_clip_is_finite():
    assert np.isfinite(linking.logit(np.array([0.0, 1.0, 0.5]))).all()
    assert np.allclose(linking.logit(np.array([0.25])), logit(0.25))


def test_flexible_link_recovers_nonlinear_score_function_under_score_selection():
    """True logit is piecewise in z (bimodal, transformer-like scores). Linear link is misspecified and biased
    under score selection; the hinge link is not."""
    rng = np.random.default_rng(5); n = 200000
    L = rng.choice(4, n, p=[0.1, 0.35, 0.35, 0.2])
    z = np.where(rng.random(n) < 0.6, rng.normal(-1.5, 0.3, n), rng.normal(1.0, 0.8, n)) + 0.2 * L
    s = 2.5 * np.minimum(z + 1.2, 0) + 0.6 * z
    ya = (rng.random(n) < expit(np.array([-1.5, -0.8, 0.4, 1.2])[L] + s)).astype(float)
    sel = rng.random(n) < expit(1.0 * (z - z.mean()) / z.std() + 0.4); p = expit(z)
    truth = M.auroc(ya[~sel], p[~sel])
    lin = M.auroc(linking.predict(linking.fit(L[sel], z[sel], ya[sel], 1.0), L[~sel], z[~sel]), p[~sel]) - truth
    flx = M.auroc(linking.predict(linking.fit(L[sel], z[sel], ya[sel], 1.0, flex=True), L[~sel], z[~sel]), p[~sel]) - truth
    assert abs(flx) < 0.01 and abs(flx) < abs(lin)
