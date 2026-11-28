"""E30-E32 building blocks on synthetic data."""
import numpy as np
import pandas as pd
from scipy import stats
from scope import jackknife, e30, priors


def test_jackknife_of_mean_equals_classical_se():
    # for the mean of G equal-size cluster means, the delete-one jackknife SE is the usual SE of the mean
    rng = np.random.default_rng(1); G = 12; v = rng.normal(size=G)
    loo = [{"m": np.delete(v, k).mean()} for k in range(G)]
    r = jackknife.summarize({"m": v.mean()}, loo, G)[0]
    assert abs(r["jk_se"] - v.std(ddof=1) / np.sqrt(G)) < 1e-12
    assert abs(r["jk_hi"] - r["estimate"] - stats.t.ppf(0.975, G - 1) * r["jk_se"]) < 1e-12
    assert r["loco_min"] <= r["estimate"] <= r["loco_max"]


def test_g_tilde_reduces_to_anchor_frequencies_when_every_item_is_an_anchor():
    rng = np.random.default_rng(2); n = 800
    L = rng.integers(0, 4, n); z = rng.normal(size=n) + 0.5 * L
    ya = (rng.random(n) < 1 / (1 + np.exp(-(z - 1)))).astype(float)
    d = pd.DataFrame({"L": L, "z": z, "ya": ya})
    news = pd.DataFrame({"L": L, "z": z, "ya_obs": ya})
    gt = e30.g_tilde(d, news)
    assert np.allclose(gt, [ya[L == l].mean() for l in range(4)])


def test_g_tilde_corrects_score_selected_anchors():
    # anchors over-represent high scores within level (Assumption 1 fails, conditional MAR holds):
    # g_tilde recovers the population link; anchor frequencies over-state it
    rng = np.random.default_rng(3); n = 20000
    L = rng.integers(0, 4, n); z = rng.normal(size=n)
    h = 1 / (1 + np.exp(-(-1.5 + 0.7 * L + 1.2 * z))); ya = (rng.random(n) < h).astype(float)
    sel = rng.random(n) < 1 / (1 + np.exp(-(-1 + 1.5 * z)))
    d = pd.DataFrame({"L": L[sel], "z": z[sel], "ya": ya[sel]})
    news = pd.DataFrame({"L": L, "z": z, "ya_obs": np.where(sel, ya, np.nan)})
    truth = np.array([ya[L == l].mean() for l in range(4)])
    naive = np.array([ya[sel & (L == l)].mean() for l in range(4)])
    gt = e30.g_tilde(d, news)
    assert np.max(np.abs(gt - truth)) < 0.02 and np.min(naive - truth) > 0.05


def test_prior_adjust_round_trip_and_mlls_fixed_point():
    rng = np.random.default_rng(4); p = rng.uniform(0.01, 0.99, 500)
    assert np.allclose(priors.prior_adjust(priors.prior_adjust(p, 0.3, 0.6), 0.6, 0.3), p)
    pi = priors.mlls_em(p, 0.4)
    assert abs(np.mean(priors.prior_adjust(p, 0.4, pi)) - pi) < 1e-6
