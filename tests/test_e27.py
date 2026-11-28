"""E27 analysis functions on synthetic data (no real annotations involved)."""
import numpy as np
import pandas as pd
from scope import e27, linking
from scope import metrics as M


def test_fleiss_and_cohen_known_values():
    perfect = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0]], float)
    assert abs(e27.fleiss_kappa(perfect) - 1) < 1e-12
    assert abs(e27.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) - 1) < 1e-12
    # textbook 2x2: po = 0.7, pe = 0.5 -> kappa 0.4
    a = [1] * 5 + [0] * 5; b = [1] * 4 + [0] + [1] * 2 + [0] * 3
    assert abs(e27.cohen_kappa(a, b) - 0.4) < 1e-12


def test_majority_and_ties():
    s1 = pd.DataFrame({"audit_id": ["a", "b", "c"], "label_0_1": [1, 0, 1]})
    s2 = pd.DataFrame({"audit_id": ["a", "b", "c"], "label_0_1": [1, 1, ""]})
    s3 = pd.DataFrame({"audit_id": ["a", "b", "c"], "label_0_1": [0, 0, 0]})
    m = e27.majority([s1, s2, s3]).set_index("audit_id")
    assert m.loc["a", "y_major"] == 1 and m.loc["b", "y_major"] == 0
    assert np.isnan(m.loc["c", "y_major"])  # 1 vs 0 with one missing -> tie


def test_block_estimate_unbiased_on_synthetic_population():
    rng = np.random.default_rng(0)
    N = 1500; L = rng.integers(0, 4, N); z = rng.normal(0, 1.5, N) + 0.6 * L
    y_true = (rng.random(N) < 1 / (1 + np.exp(-(z - 1.5)))).astype(float); p = 1 / (1 + np.exp(-z))
    pop = pd.DataFrame({"row_id": np.arange(N), "L": L, "p": p})
    # working link deliberately mis-specified (fitted on shifted labels) -> difference estimator must stay unbiased
    aux = rng.integers(0, N, 400); ya_aux = (rng.random(400) < 1 / (1 + np.exp(-(z[aux] - 0.5)))).astype(float)
    link = linking.fit(L[aux], z[aux], ya_aux, 1.0, flex=True)
    truth = M.auroc(y_true, p); est = []
    for r in range(150):
        smp = []
        for lv in range(4):
            idx = np.flatnonzero(L == lv); take = rng.choice(idx, 25, replace=False)
            smp.append(pd.DataFrame({"row_id": take, "level": lv, "y": y_true[take]}))
        e = e27.block_estimate(pop, pd.concat(smp), link); est.append(e["auroc"])
    assert abs(np.mean(est) - truth) < 0.01
