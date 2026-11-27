"""E9/E10 (CPU families): cross-environment training with leave-one-environment-out (LOEO).

Environments: A = BanglaBait train (source), B = BanglaClick news/facebook/youtube/twitter,
C = BaitBuster YouTube human subset. For a held-out target cell e*, protocols are
  source_only   train on A only (R6 setting)
  pooled_erm    train on A plus every other cell (excluding e*)
  group_dro     same data, worst-group objective over environments (batch Group DRO:
                exponentiated-gradient group weights, Sagawa et al., 2020)
  in_domain     5-fold cross-fitting inside e*, folds grouped by publisher/channel cluster
                (a reference ceiling that uses target labels; not a deployable baseline)
Each protocol is run under two regime-B label definitions: r6 (1[L>=1]) and aligned (1[L>=2]).
Calibration: scalar temperature fitted on a cluster-grouped 10% validation split of the
training pool (never on e*). Models: TF-IDF word(1-2)+char(3-5) logistic regression (R6
config) and a majority-class baseline. No target data enter training except in_domain.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import FeatureUnion
from . import metrics as M

TARGETS = ["banglaclick:news", "banglaclick:facebook", "banglaclick:youtube", "banglaclick:twitter", "baitbuster:youtube"]


def featurizer():
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True, dtype=np.float32)),
        ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=100000, sublinear_tf=True, dtype=np.float32))])


def fit_temperature(z, y):
    f = lambda T: M.nll(y, expit(z / T))
    return minimize_scalar(f, bounds=(0.05, 20), method="bounded").x


def fit_lr(X, y, w=None, seed=0):
    return LogisticRegression(C=1.0, solver="liblinear", max_iter=2000, random_state=seed).fit(X, y, sample_weight=w)


def fit_group_dro(X, y, groups, seed, rounds=5, eta=1.0):
    """Batch Group DRO: alternate a weighted fit with q_g <- q_g * exp(eta * loss_g)."""
    gs = np.unique(groups); q = np.ones(len(gs)) / len(gs)
    sizes = np.array([(groups == g).sum() for g in gs])
    clf = None
    for _ in range(rounds):
        w = np.empty(len(y))
        for k, g in enumerate(gs):
            w[groups == g] = q[k] * len(y) / sizes[k]
        clf = fit_lr(X, y, w, seed)
        p = clf.predict_proba(X)[:, 1]
        loss = np.array([M.nll(y[groups == g], p[groups == g]) for g in gs])
        q = q * np.exp(eta * loss); q = q / q.sum()
    return clf, dict(zip(gs, q))


def _save_pred(pred_dir, model, proto, lab, tgt, seed, row_id, p):
    if pred_dir is not None:
        pred_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"row_id": np.asarray(row_id), "p": np.asarray(p, float)}).to_parquet(
            pred_dir / f"{model}__{proto}__{lab}__{tgt.replace(':', '-')}__s{seed}.parquet", index=False)


def run_loeo(pool: pd.DataFrame, seeds, label_sets=("r6", "aligned"), pred_dir=None):
    """pool columns: title, env, cluster_id, normalized_title_sha256, y_r6, y_al (y_al == y_r6 outside regime B)."""
    rows = []
    for lab in label_sets:
        ycol = "y_r6" if lab == "r6" else "y_al"
        for tgt in TARGETS:
            test = pool[pool.env == tgt]
            for proto in ("majority", "source_only", "pooled_erm", "group_dro", "in_domain"):
                for seed in seeds:
                    if proto == "in_domain":
                        p = np.empty(len(test)); gkf = GroupKFold(n_splits=5)
                        for tr_i, te_i in gkf.split(test, groups=test.cluster_id):
                            tr = test.iloc[tr_i]; feats = featurizer(); Xtr = feats.fit_transform(tr.title)
                            clf = fit_lr(Xtr, tr[ycol].to_numpy(), seed=seed)
                            p[te_i] = clf.predict_proba(feats.transform(test.iloc[te_i].title))[:, 1]
                    else:
                        train = pool[pool.env == "banglabait:news"] if proto in ("source_only", "majority") else pool[pool.env != tgt]
                        train = train[~train.normalized_title_sha256.isin(test.normalized_title_sha256)]  # cross-cell text dedup
                        gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
                        fit_i, val_i = next(gss.split(train, groups=train.cluster_id))
                        tr, va = train.iloc[fit_i], train.iloc[val_i]
                        if proto == "majority":
                            p = np.full(len(test), tr[ycol].mean())
                        else:
                            feats = featurizer(); Xtr = feats.fit_transform(tr.title)
                            if proto == "group_dro":
                                clf, _ = fit_group_dro(Xtr, tr[ycol].to_numpy(), tr.env.to_numpy(), seed)
                            else:
                                clf = fit_lr(Xtr, tr[ycol].to_numpy(), seed=seed)
                            T = fit_temperature(clf.decision_function(feats.transform(va.title)), va[ycol].to_numpy())
                            p = expit(clf.decision_function(feats.transform(test.title)) / T)
                    y = test[ycol].to_numpy()
                    r = M.all_metrics(y, p, hard=y)
                    r.update(label=lab, target=tgt, protocol=proto, model="tfidf_lr" if proto != "majority" else "majority", seed=seed)
                    rows.append(r)
                    _save_pred(pred_dir, r["model"], proto, lab, tgt, seed, test.row_id, p)
                    if proto in ("majority",):
                        break  # deterministic
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- E16: shift-correction baselines on the same LOEO splits
def density_ratio_weights(Xtr, Xte, seed, clip=(0.1, 10.0), folds=5):
    """Covariate-shift importance weights w(x) = p_te(x)/p_tr(x) from a cross-fitted domain classifier
    (Shimodaira, 2000; Sugiyama et al., 2007). Uses unlabelled target text only."""
    import scipy.sparse as sp
    from sklearn.model_selection import StratifiedKFold
    X = sp.vstack([Xtr, Xte]).tocsr(); d = np.r_[np.zeros(Xtr.shape[0]), np.ones(Xte.shape[0])]
    s = np.empty(len(d))
    for a, b in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, d):
        s[b] = fit_lr(X[a], d[a], seed=seed).predict_proba(X[b])[:, 1]
    s = np.clip(s[: Xtr.shape[0]], 1e-6, 1 - 1e-6)
    w = s / (1 - s) * (Xtr.shape[0] / Xte.shape[0])
    w = np.clip(w, *clip)
    return w / w.mean()


def run_shift_baselines(pool: pd.DataFrame, seeds, label_sets=("r6", "aligned"), pred_dir=None):
    """Same targets, dedup rule, validation split, temperature fit and metric code as run_loeo.
    Protocols:
      source_mlls      source-only model, prior moved to the MLLS estimate from unlabelled target scores
      source_oracle    source-only model, prior moved to the true target prevalence (reference; uses labels)
      iw_source        source data reweighted by target/source density ratio
      iw_pooled        pooled data (target excluded) reweighted by target/pool density ratio
    """
    from .priors import mlls_em, prior_adjust
    rows = []
    for lab in label_sets:
        ycol = "y_r6" if lab == "r6" else "y_al"
        for tgt in TARGETS:
            test = pool[pool.env == tgt]; y = test[ycol].to_numpy()
            for seed in seeds:
                for base in ("source", "pooled"):
                    train = pool[pool.env == "banglabait:news"] if base == "source" else pool[pool.env != tgt]
                    train = train[~train.normalized_title_sha256.isin(test.normalized_title_sha256)]
                    fit_i, val_i = next(GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed).split(train, groups=train.cluster_id))
                    tr, va = train.iloc[fit_i], train.iloc[val_i]
                    feats = featurizer(); Xtr = feats.fit_transform(tr.title); Xte = feats.transform(test.title); Xva = feats.transform(va.title)
                    out = {}
                    if base == "source":
                        clf = fit_lr(Xtr, tr[ycol].to_numpy(), seed=seed)
                        T = fit_temperature(clf.decision_function(Xva), va[ycol].to_numpy())
                        p0 = expit(clf.decision_function(Xte) / T); pi_s = float(tr[ycol].mean())
                        out["source_mlls"] = prior_adjust(p0, pi_s, mlls_em(p0, pi_s))
                        out["source_oracle"] = prior_adjust(p0, pi_s, float(np.mean(y)))
                    w = density_ratio_weights(Xtr, Xte, seed)
                    clf = fit_lr(Xtr, tr[ycol].to_numpy(), w=w, seed=seed)
                    T = fit_temperature(clf.decision_function(Xva), va[ycol].to_numpy())
                    out[f"iw_{base}"] = expit(clf.decision_function(Xte) / T)
                    ess = float(w.sum() ** 2 / (w ** 2).sum() / len(w))
                    for proto, p in out.items():
                        r = M.all_metrics(y, p, hard=y)
                        r.update(label=lab, target=tgt, protocol=proto, model="tfidf_lr", seed=seed,
                                 iw_relative_ess=ess if proto.startswith("iw") else float("nan"))
                        rows.append(r)
                        _save_pred(pred_dir, "tfidf_lr", proto, lab, tgt, seed, test.row_id, p)
    return pd.DataFrame(rows)
