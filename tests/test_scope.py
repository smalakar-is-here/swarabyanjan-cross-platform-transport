"""Unit, leakage and reproducibility tests for the SCOPE package.

Unit tests need no data. Integration tests need SCOPE_PRIVATE_DATA and are skipped otherwise.
Run: PYTHONPATH=src pytest -q tests
"""
import json, os
from pathlib import Path
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from scope import metrics as M, priors, decompose, construct, transport

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)


def test_weighted_auroc_matches_sklearn_on_hard_labels():
    y = rng.integers(0, 2, 500).astype(float); p = np.round(rng.random(500), 2)  # with ties
    assert abs(M.auroc(y, p) - roc_auc_score(y, p)) < 1e-12


def test_soft_metrics_reduce_to_hard():
    y = rng.integers(0, 2, 200).astype(float); p = rng.random(200)
    assert M.nll(y, p) == pytest.approx(M.nll(y.astype(int), p))
    assert M.auprc(y, p) > 0


def test_prior_adjust_identity_and_mlls_recovers_prior():
    p = rng.random(100)
    assert np.allclose(priors.prior_adjust(p, 0.3, 0.3), p)
    # simulate a calibrated model under label shift
    y = (rng.random(20000) < 0.6).astype(int)
    score = rng.normal(loc=y * 1.5, scale=1.0)
    # posterior under training prior 0.3 for N(1.5,1) vs N(0,1)
    lr = np.exp(1.5 * score - 1.125)
    p_src = 0.3 * lr / (0.3 * lr + 0.7)
    assert abs(priors.mlls_em(p_src, 0.3) - 0.6) < 0.02


def test_decomposition_identity_is_exact():
    y = rng.integers(0, 2, 300).astype(float); ya = rng.integers(0, 2, 300).astype(float); p = rng.random(300)
    rs = {"nll": 0.4, "brier": 0.15}
    c = decompose.components(p, y, ya, 0.33, rs, 0.8)
    for k in ("nll", "brier"):
        assert c[f"{k}_construct"] + c[f"{k}_prior"] + c[f"{k}_residual"] == pytest.approx(c[f"{k}_gap"], abs=1e-12)


def test_pava_monotone():
    out = construct.pava([0.1, 0.4, 0.3, 0.9], [10, 5, 5, 1])
    assert np.all(np.diff(out) >= -1e-12)
    assert out[1] == pytest.approx(out[2])


def test_log_or_invariant_to_label_shift_rd_not():
    n = 200000
    y = (rng.random(n) < 0.3).astype(int); c = rng.random(n) < np.where(y == 1, 0.5, 0.2)
    y2 = (rng.random(n) < 0.8).astype(int); c2 = rng.random(n) < np.where(y2 == 1, 0.5, 0.2)
    assert abs(transport.log_or(y, c) - transport.log_or(y2, c2)) < 0.05
    assert abs(transport.risk_diff(y, c) - transport.risk_diff(y2, c2)) > 0.05


def test_wild_cluster_pvalue_in_unit_interval():
    cl = np.repeat(np.arange(12), 30); x = (rng.random(len(cl)) < 0.4).astype(float); y = (rng.random(len(cl)) < 0.5).astype(float)
    b, p = transport.wild_cluster_rd_pvalue(y, x, cl, 199, rng)
    assert 0 <= p <= 1


def test_cue_v11_detaches_punctuation_and_v10_does_not():
    from scope import cues
    t = ["সে কেন? চলে গেল"]
    assert cues.extract(t, "1.1").interrogative_structure.iloc[0] == 1  # কেন counted, '?' not double counted
    assert cues.extract(t, "1.0").interrogative_structure.iloc[0] == 1  # v1.0 misses কেন? but counts '?'


def test_leakage_guards_present():
    """Every training path removes items whose normalised title occurs in the held-out cell."""
    cpu = (ROOT / "src/scope/train_cpu.py").read_text(encoding="utf-8")
    gpu = (ROOT / "experiments/kaggle/scope_loeo_transformer/run_template.py").read_text(encoding="utf-8")
    assert "~train.normalized_title_sha256.isin(test.normalized_title_sha256)" in cpu
    # base is the source cell (source_only) or every other cell (other protocols); both exclude test titles
    assert "train = base[~base.normalized_title_sha256.isin(test.normalized_title_sha256)]" in gpu
    assert "pool.env != tgt" in gpu and 'pool[pool.env == "banglabait:news"]' in gpu


def test_no_text_in_public_results():
    for f in (ROOT / "results/scope_v1").rglob("*.csv"):
        head = f.read_text(encoding="utf-8").splitlines()[0].split(",")
        assert not {"title", "text", "original_title", "neutralized_title"} & set(head), f


needs_data = pytest.mark.skipif(not os.environ.get("SCOPE_PRIVATE_DATA"), reason="private data not available")


@needs_data
def test_restricted_artifacts_verified():
    from scope import paths
    st = [r["status"] for r in paths.verify_restricted()]
    assert st.count("VERIFIED") >= 22 and "HASH_MISMATCH" not in st


@needs_data
def test_r6_label_is_level_at_least_one_and_reproduces_h1_auroc():
    from scope import data
    df = data.load_predictions()  # asserts label == 1[level>=1]
    h1 = json.loads((ROOT / "results/primary/stage05/H1_RESULTS.json").read_text(encoding="utf-8"))
    envmap = {"source_test": "banglabait:news", "baitbuster:youtube_human": "baitbuster:youtube"}
    checked = 0
    for r in h1["metric_rows"]:
        env = envmap.get(r["domain"], r["domain"])
        e = df[(df.env == env) & (df.model == r["model"]) & (df.seed == int(r["seed"]))]
        if len(e) and r["calibrated"].get("auroc") is not None:
            assert M.auroc(e.label.to_numpy(), e.p_calibrated.to_numpy()) == pytest.approx(r["calibrated"]["auroc"], abs=1e-9)
            assert M.nll(e.label.to_numpy(), e.p_calibrated.to_numpy()) == pytest.approx(r["calibrated"]["nll"], abs=1e-6)
            checked += 1
    assert checked == 120


def test_soft_brier_is_expected_hard_brier():
    import numpy as np
    from scope import metrics as M
    rng = np.random.default_rng(0); q = rng.random(4000); p = rng.random(4000)
    sims = [M.brier((rng.random(4000) < q).astype(float), p) for _ in range(400)]
    assert abs(np.mean(sims) - M.brier(q, p)) < 2e-3


def test_synthetic_soft_proxy_unbiased_when_score_runs_through_level():
    import numpy as np
    from scope import synthetic
    g = np.array([0.10, 0.25, 0.58, 0.66]); pl = np.array([.15, .5, .17, .18])
    _, s = synthetic.run(g, pl, [1083], [0.0], [0.0], [1.0], reps=6, boot=2, seed=3, n_jobs=1)
    soft = s[(s.proxy == "soft")].set_index("metric").bias
    assert abs(soft["nll"]) < 0.02 and abs(soft["brier"]) < 0.01 and abs(soft["auroc"]) < 0.02


def test_kaggle_worker_units_render_and_parse():
    import ast, importlib.util, json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("wp", root / "scripts/scope_make_worker_packages.py")
    wp = importlib.util.module_from_spec(spec); spec.loader.exec_module(wp)
    src = (root / "experiments/kaggle/scope_loeo_transformer/run_template.py").read_text(encoding="utf-8")
    for u in wp.UNITS.values():
        s = src
        for k, v in {"__MODEL__": u["model"], "__MODE__": "full", "__GIT_SHA__": "x", "__PROTOCOL__": u["protocol"],
                     "__LABELS__": json.dumps([u["label"]]), "__TARGETS__": "[]", "__SEEDS__": "[]"}.items():
            s = s.replace(k, v)
        ast.parse(s)
        assert "__" + "MODEL__" not in s and f'PROTOCOL = "{u["protocol"]}"' in s
        assert "normalized_title_sha256.isin(test.normalized_title_sha256)" in s  # cross-cell dedup kept


def _load_objective():
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "experiments/kaggle/scope_loeo_transformer/run_template.py").read_text(encoding="utf-8")
    code = src[src.index("# OBJECTIVE-BEGIN"):src.index("# OBJECTIVE-END")]
    import torch
    ns = {"torch": torch, "PROTOCOL": "group_dro", "DRO_ETA": 0.01}
    exec(code, ns)
    return ns["objective"], torch


def test_group_dro_objective_matches_hand_computation():
    import numpy as np, pytest
    pytest.importorskip("torch")
    objective, torch = _load_objective()
    lv = torch.tensor([1.0, 3.0, 2.0, 4.0, 0.5], requires_grad=True)
    gb = torch.tensor([0, 0, 1, 1, 3])            # group 2 absent from the batch
    q0 = torch.tensor([0.1, 0.2, 0.3, 0.4])
    loss, q = objective(lv, gb, q0, 4, "group_dro", eta=0.5)
    gm = np.array([2.0, 3.0, 0.0, 0.5])            # per-group batch means, absent group 0
    qe = np.array([0.1, 0.2, 0.3, 0.4]) * np.exp(0.5 * gm); qe /= qe.sum()
    assert np.allclose(q.numpy(), qe, atol=1e-6)
    assert abs(loss.item() - float(qe @ gm)) < 1e-6
    loss.backward()                                 # d loss / d l_i = q_g(i) / n_g(i); weights are not differentiated
    assert np.allclose(lv.grad.numpy(), [qe[0] / 2, qe[0] / 2, qe[1] / 2, qe[1] / 2, qe[3]], atol=1e-6)


def test_group_balanced_and_erm_objectives():
    import numpy as np, pytest
    pytest.importorskip("torch")
    objective, torch = _load_objective()
    lv = torch.tensor([1.0, 3.0, 2.0, 4.0, 0.5]); gb = torch.tensor([0, 0, 1, 1, 3]); q0 = torch.ones(4) / 4
    l, q = objective(lv, gb, q0, 4, "group_balanced")
    assert abs(l.item() - np.mean([2.0, 3.0, 0.5])) < 1e-6 and torch.equal(q, q0)  # absent group excluded
    l, _ = objective(lv, gb, q0, 4, "pooled_erm")
    assert abs(l.item() - 2.1) < 1e-6


def test_holm_and_paired_bootstrap_identity():
    import numpy as np
    from scope import compare
    assert np.allclose(compare.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    rng = np.random.default_rng(0); y = (rng.random(300) < .4).astype(float); p = rng.random(300)
    r = compare.cell({"source_only": [p], "same": [p.copy()]}, y, np.repeat(np.arange(30), 10), "source_only", 50, rng)
    assert np.allclose(r["diff"], 0) and np.allclose(r.ci_low, 0) and np.allclose(r.ci_high, 0)


def test_template_latent_loss_matches_cpu_latent_model():
    """GPU latent_cuts loss (3 levels) == scope.latent likelihood with theta_3 -> +inf, lam = 0."""
    import numpy as np, pytest
    torch = pytest.importorskip("torch")
    from pathlib import Path
    from scope import latent
    src = (Path(__file__).resolve().parents[1] / "experiments/kaggle/scope_loeo_transformer/run_template.py").read_text(encoding="utf-8")
    ns = {"torch": torch, "PROTOCOL": "latent_cuts", "DRO_ETA": 0.01}
    exec(src[src.index("# OBJECTIVE-BEGIN"):src.index("# OBJECTIVE-END")], ns)
    rng = np.random.default_rng(0); n = 200
    f = rng.normal(0, 2, n); reg = rng.choice([0, 1, 2], n); y = np.where(reg == 1, rng.integers(0, 3, n), rng.integers(0, 2, n)).astype(float)
    tA, tC, t1, lg = 0.7, -0.4, -1.1, 0.3
    cuts = torch.tensor([tA, tC, t1, lg], dtype=torch.float64, requires_grad=True)
    gpu = ns["latent_loss"](torch.tensor(f), torch.tensor(reg), torch.tensor(y), cuts).sum()
    cpu, _ = latent.nll_grad(np.r_[1.0, tA, tC, t1, lg, 30.0], f[:, None], reg, y, lam=0.0)
    assert abs(gpu.item() - cpu) < 1e-6
    gpu.backward()
    assert torch.isfinite(cuts.grad).all() and cuts.grad.abs().sum() > 0
