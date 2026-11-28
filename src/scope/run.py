"""Experiment runner. Usage:  python -m scope.run E0 E1 E2 E3 E4 E5 E11 E13   (or: all)

Every output is aggregate and text-free; it is written to results/scope_v1/<EXP>/ together with
a manifest (inputs, config hash, versions, wall time).
"""
from __future__ import annotations
import hashlib, json, platform, sys, time
import numpy as np
import pandas as pd
from . import paths, data, construct, decompose, priors, transport, audit, fewcluster
from . import metrics as M

C = paths.CONFIG
OUT = paths.RESULTS
REPO = paths.REPO
DEV = REPO / "data/derived/primary/stage04/DEV_PREDICTIONS.parquet"


def _write(exp, name, obj):
    d = OUT / exp; d.mkdir(parents=True, exist_ok=True)
    p = d / name
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(p, index=False, float_format="%.6g", lineterminator="\n")
    else:
        p.write_text(json.dumps(obj, indent=1, default=float) + "\n", encoding="utf-8")
    return p


def _manifest(exp, t0, extra=None):
    import sklearn, scipy, statsmodels
    m = {"experiment": exp, "scope_version": "1.0.0", "wall_seconds": round(time.time() - t0, 1),
         "config_sha256": hashlib.sha256((REPO / "configs/scope/scope_config.json").read_bytes()).hexdigest(),
         "python": sys.version.split()[0], "platform": platform.platform(),
         "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                      "scipy": scipy.__version__, "statsmodels": statsmodels.__version__},
         "outputs": sorted(p.name for p in (OUT / exp).glob("*") if p.name != "MANIFEST.json")}
    if extra:
        m.update(extra)
    _write(exp, "MANIFEST.json", m)


_CACHE = {}


def preds():
    """Frozen R6 predictions (inference only): news clustered by publisher (data.publisher_cluster)."""
    if "df" not in _CACHE:
        df = data.load_predictions(); df["cluster_id"] = df.cluster_id.map(data.publisher_cluster)
        _CACHE["df"] = df
    return _CACHE["df"]


def anchors():
    if "anc" not in _CACHE:
        a = data.anchors(); a["bc_cluster"] = a.bc_cluster.map(data.publisher_cluster)
        _CACHE["anc"] = a
    return _CACHE["anc"]


def inference_pool():
    """training_pool() with publisher clusters for inference (bootstrap, cross-fit folds, design halves).
    Training splits must keep training_pool().cluster_id so that every fitted model is unchanged."""
    p = training_pool().copy(); p["cluster_id"] = p.cluster_id.map(data.publisher_cluster)
    return p


def pi_source():
    dev = pd.read_parquet(DEV)
    return float(dev.drop_duplicates("source_row_id").label.mean())


def add_labels(df, g):
    """y_r6 (R6 label), y_al (anchor-selected hard construct, 1[L>=2]), q_al (soft g(L)).
    Regimes A and C keep their own binary labels for all three columns."""
    df = df.copy()
    df["y_r6"] = df["label"].astype(float)
    lv = df["level"]
    isB = df["regime"] == "B"
    df["y_al"] = df["y_r6"]; df.loc[isB, "y_al"] = (lv[isB] >= 2).astype(float)
    df["q_al"] = df["y_r6"]; df.loc[isB, "q_al"] = g[lv[isB].astype(int).to_numpy()]
    return df


# ---------------------------------------------------------------- E0
def E0():
    t0 = time.time()
    r = pd.DataFrame(paths.verify_restricted()); w = pd.DataFrame(paths.verify_raw())
    _write("E0", "restricted_artifact_status.csv", r[["path", "role", "status", "expected_sha256"]])
    _write("E0", "raw_external_status.csv", w)
    df = preds()
    cells = (df.groupby("env").agg(rows=("row_id", "nunique"), clusters=("cluster_id", "nunique"),
                                   prevalence_r6=("label", "mean")).reset_index())
    _write("E0", "cell_inventory.csv", cells)
    _manifest("E0", t0, {"restricted_verified": int((r.status == "VERIFIED").sum()), "restricted_total": len(r),
                         "raw_verified": int((w.status == "VERIFIED").sum()), "raw_total": len(w)})


# ---------------------------------------------------------------- E1
def E1():
    t0 = time.time(); a = anchors()
    _write("E1", "anchor_crosstab.csv", pd.crosstab(a.bb_label, a.bc_level).reset_index())
    _write("E1", "threshold_agreement.csv", construct.threshold_agreement(a))
    g = construct.linking_function(a)
    bi = construct.bootstrap(a, C["bootstrap_draws"], C["bootstrap_seed"])
    bc = construct.bootstrap(a, C["bootstrap_draws"], C["bootstrap_seed"] + 1, cluster="bc_cluster")
    lk = pd.DataFrame({"level": range(4), "n": [(a.bc_level == l).sum() for l in range(4)],
                       "g_isotonic": g, "ci_item_low": [c[0] for c in bi["g_ci"]], "ci_item_high": [c[1] for c in bi["g_ci"]],
                       "ci_cluster_low": [c[0] for c in bc["g_ci"]], "ci_cluster_high": [c[1] for c in bc["g_ci"]]})
    _write("E1", "linking_function.csv", lk)
    _write("E1", "bootstrap_summary.json", {"item_bootstrap": bi, "publisher_cluster_bootstrap": bc})
    _write("E1", "residual_disagreement_by_rubric.csv", construct.residual_disagreement(a))
    _write("E1", "rubric_logit_HC1.csv", construct.logistic_dimension_model(a))
    # anchor representativeness: model scores of anchor vs non-anchor BanglaClick news items, by level
    df = preds(); bcn = df[(df.env == "banglaclick:news")]
    ah = set(a.normalized_title_sha256)
    bcn = bcn.assign(anchor=bcn.normalized_title_sha256.isin(ah))
    rep = (bcn.groupby(["model", "level", "anchor"]).p_calibrated.agg(["mean", "size"]).reset_index())
    _write("E1", "anchor_representativeness.csv", rep)
    comp = {"anchor_classes": int(len(a)), "banglaclick_news_items": int(bcn.row_id.nunique()),
            "anchor_items_in_banglaclick_news": int(bcn[bcn.anchor].row_id.nunique()),
            "banglabait_splits_of_anchors": a.bb_splits.value_counts().to_dict(),
            "classes_with_inconsistent_bc_level": int((a.bc_level != a.bc_level_min).sum()),
            "prior_audit_reported_classes": 1058,
            "note": "1058 in the 2026-11-26 assessment came from the R6 decontamination file (per-split counts of Stage02-cleaned rows); 1083 here joins all 15,056 raw BanglaBait rows by normalised-title hash."}
    _write("E1", "anchor_summary.json", comp)
    _manifest("E1", t0)


# ---------------------------------------------------------------- E2 (+E6 crossed effects)
def E2(cue_frame=None, tag=""):
    t0 = time.time(); a = anchors(); g = construct.linking_function(a)
    df = add_labels(preds(), g) if cue_frame is None else cue_frame
    items = df.drop_duplicates("row_id") if cue_frame is None else df
    rng = np.random.default_rng(C["bootstrap_seed"])
    rows = []
    for env in transport.CELL_DESIGN:
        it = items[items.env == env]
        for ycol in ("y_r6", "y_al"):
            if ycol == "y_al" and env in ("banglabait:news", "baitbuster:youtube"):
                continue  # identical to y_r6 for regimes A and C
            for cue in data.CUES:
                c = it[cue] > 0
                bs = transport.cluster_boot(it, ycol, cue, 1000, rng)
                rd, p_wcr = transport.wild_cluster_rd_pvalue(it[ycol], c.astype(float), it.cluster_id, 999, rng)
                rows.append({"env": env, "platform": transport.CELL_DESIGN[env][0], "regime": transport.CELL_DESIGN[env][1],
                             "label": ycol, "cue": cue, "cue_prevalence": float(c.mean()), "prevalence": float(it[ycol].mean()),
                             "log_or": transport.log_or(it[ycol], c), "log_or_lo": float(np.nanpercentile(bs[:, 0], 2.5)),
                             "log_or_hi": float(np.nanpercentile(bs[:, 0], 97.5)), "rd": rd,
                             "rd_lo": float(np.nanpercentile(bs[:, 1], 2.5)), "rd_hi": float(np.nanpercentile(bs[:, 1], 97.5)),
                             "rd_wcr_p": p_wcr, "clusters": int(it.cluster_id.nunique())})
    cell = pd.DataFrame(rows)
    _write("E2" + tag, "cue_association_by_cell.csv", cell)
    # crossed (just-identified) contrasts, per cue, under the aligned label for regime B
    cr = []
    for cue in data.CUES:
        for lab in ("y_r6", "y_al"):
            b = {}
            for env in transport.CELL_DESIGN:
                s = cell[(cell.env == env) & (cell.cue == cue) & (cell.label == (lab if transport.CELL_DESIGN[env][1] == "B" else "y_r6"))]
                b[env] = float(s.log_or.iloc[0])
            e = transport.crossed_effects(b); e.update(cue=cue, regime_B_label=lab); cr.append(e)
    _write("E2" + tag, "crossed_platform_regime_contrasts.csv", pd.DataFrame(cr))
    _manifest("E2" + tag, t0, {"cluster_bootstrap_draws": 1000, "wild_cluster_draws": 999})


# ---------------------------------------------------------------- E3 + E5
def E3():
    t0 = time.time(); a = anchors(); g = construct.linking_function(a)
    df = add_labels(preds(), g); pi_s = pi_source()
    dev = pd.read_parquet(DEV)
    rows = []
    for (env, m, s), e in df[df.env != "banglabait:news"].groupby(["env", "model", "seed"]):
        d = dev[(dev.model == m) & (dev.seed == s)]
        p = e.p_calibrated.to_numpy()
        est = {"cc": priors.cc(p), "pcc": priors.pcc(p), "bbse": priors.bbse(p, d.p_calibrated, d.label),
               "mlls": priors.mlls_em(p, pi_s)}
        for k, v in est.items():
            rows.append({"env": env, "model": m, "seed": s, "estimator": k, "estimate": v,
                         "true_r6": float(e.y_r6.mean()), "true_aligned_hard": float(e.y_al.mean()),
                         "true_aligned_soft": float(e.q_al.mean())})
    r = pd.DataFrame(rows)
    r["abs_err_r6"] = (r.estimate - r.true_r6).abs(); r["abs_err_aligned"] = (r.estimate - r.true_aligned_hard).abs()
    _write("E3", "prevalence_estimates.csv", r)
    summ = r.groupby(["env", "estimator"])[["estimate", "true_r6", "true_aligned_hard", "abs_err_r6", "abs_err_aligned"]].mean().reset_index()
    _write("E3", "prevalence_estimates_summary.csv", summ)
    _manifest("E3", t0, {"pi_source_dev": pi_s})


def E5():
    """Calibration/adaptation comparison on each target cell (mean over seeds)."""
    t0 = time.time(); a = anchors(); g = construct.linking_function(a)
    df = add_labels(preds(), g); pi_s = pi_source(); dev = pd.read_parquet(DEV)
    rows = []
    for (env, m, s), e in df.groupby(["env", "model", "seed"]):
        y_sets = {"r6": e.y_r6.to_numpy(), "aligned": e.y_al.to_numpy()}
        d = dev[(dev.model == m) & (dev.seed == s)]
        p_unc, p_ts = e.p_uncalibrated.to_numpy(), e.p_calibrated.to_numpy()
        pi_bbse = priors.bbse(p_ts, d.p_calibrated, d.label); pi_mlls = priors.mlls_em(p_ts, pi_s)
        for lab, y in y_sets.items():
            if lab == "aligned" and not (e.regime.iloc[0] == "B"):
                continue
            variants = {"uncalibrated": p_unc, "temperature_scaled_source": p_ts,
                        "prior_bbse": priors.prior_adjust(p_ts, pi_s, pi_bbse) if np.isfinite(pi_bbse) else p_ts,
                        "prior_mlls": priors.prior_adjust(p_ts, pi_s, pi_mlls),
                        "prior_oracle": priors.prior_adjust(p_ts, pi_s, float(y.mean()))}
            for v, p in variants.items():
                r = M.all_metrics(y, p, hard=y); r.update(env=env, model=m, seed=s, label=lab, method=v); rows.append(r)
    res = pd.DataFrame(rows)
    _write("E5", "calibration_adaptation_by_seed.csv", res)
    num = [c for c in res.columns if c not in ("env", "model", "seed", "label", "method")]
    _write("E5", "calibration_adaptation_summary.csv", res.groupby(["env", "label", "model", "method"])[num].mean().reset_index())
    _manifest("E5", t0)


# ---------------------------------------------------------------- E4
def E4(exp="E4", aligned="level"):
    """Decomposition. aligned='level': Y* = 1[L>=2] (anchor kappa over level cuts); aligned='trigger': Y* = 1[trigger level >= 1],
    the rubric binarisation selected by cross-fitted kappa over level and component cuts (E24)."""
    t0 = time.time(); a = anchors(); g = construct.linking_function(a)
    df = add_labels(preds(), g); pi_s = pi_source()
    if aligned == "trigger":
        isB = df.regime == "B"; df = df.copy(); df.loc[isB, "y_al"] = (df.loc[isB, "trigger_level"].fillna(0) >= 1).astype(float)
    rows = []
    src = df[df.env == "banglabait:news"]
    for (m, s), sm in src.groupby(["model", "seed"]):
        rs, ra = decompose.source_risk(sm.p_calibrated.to_numpy(), sm.y_r6.to_numpy())
        tg = df[(df.dataset == "banglaclick") & (df.model == m) & (df.seed == s)]
        na = tg[(tg.env == "banglaclick:news") & ~tg.normalized_title_sha256.isin(set(a.normalized_title_sha256))].assign(env="banglaclick:news_nonanchor")
        for env, e in pd.concat([tg, na]).groupby("env"):
            for lab in ("y_al", "q_al"):
                c = decompose.components(e.p_calibrated.to_numpy(), e.y_r6.to_numpy(), e[lab].to_numpy(), pi_s, rs, ra)
                c.update(env=env, model=m, seed=s, aligned_label=lab); rows.append(c)
    r = pd.DataFrame(rows)
    _write(exp, "decomposition_by_seed.csv", r)
    num = [c for c in r.columns if c not in ("env", "model", "seed", "aligned_label")]
    summ = r.groupby(["aligned_label", "env", "model"])[num].mean().reset_index()
    # seed consistency: share of seeds with same sign as the mean, per component
    for comp in ("nll_construct", "nll_prior", "nll_residual", "auroc_construct"):
        summ[comp + "_seed_sign_agree"] = r.groupby(["aligned_label", "env", "model"])[comp].apply(
            lambda v: float(np.mean(np.sign(v) == np.sign(v.mean())))).to_numpy()
    _write(exp, "decomposition_summary.csv", summ)
    # cluster bootstrap (target clusters + anchor classes), seed-averaged, hard aligned label
    rng = np.random.default_rng(C["bootstrap_seed"]); B = 500
    boots = []
    for env, e_all in df[df.dataset == "banglaclick"].groupby("env"):
        items = e_all.drop_duplicates("row_id")[["row_id", "cluster_id"]].reset_index(drop=True)
        groups = list(items.groupby("cluster_id").indices.values())
        wide = {(m, s): e_all[(e_all.model == m) & (e_all.seed == s)].set_index("row_id").loc[items.row_id]
                for m in C["models"] for s in C["seeds"]}
        src_r = {(m, s): decompose.source_risk(*[src[(src.model == m) & (src.seed == s)][c].to_numpy() for c in ("p_calibrated", "y_r6")])
                 for m in C["models"] for s in C["seeds"]}
        for b in range(B):
            idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            for m in C["models"]:
                acc = []
                for s in C["seeds"]:
                    w = wide[(m, s)].iloc[idx]
                    acc.append(decompose.components(w.p_calibrated.to_numpy(), w.y_r6.to_numpy(), w.y_al.to_numpy(), pi_s, *src_r[(m, s)]))
                acc = pd.DataFrame(acc).mean()
                boots.append({"env": env, "model": m, "draw": b, **{k: acc[k] for k in
                              ("nll_gap", "nll_construct", "nll_prior", "nll_residual", "auroc_R00", "auroc_RA", "auroc_construct")}})
    bt = pd.DataFrame(boots)
    ci = bt.groupby(["env", "model"]).agg(**{f"{k}_{q}": (k, (lambda v, q=q: float(np.percentile(v, 2.5 if q == 'lo' else 97.5))))
                                             for k in ("nll_gap", "nll_construct", "nll_prior", "nll_residual", "auroc_R00", "auroc_RA", "auroc_construct")
                                             for q in ("lo", "hi")}).reset_index()
    _write(exp, "decomposition_cluster_bootstrap_ci.csv", ci)
    _manifest(exp, t0, {"aligned_label": aligned, "pi_source_dev": pi_s, "bootstrap_draws": B,
                         "bootstrap_note": "publisher/channel clusters resampled within each target cell; seed-averaged; source risk held fixed"})


def E27():
    """Source-guideline human audit (docs/scope/ANNOTATION_PROTOCOL.md). Runs only when all returned sheets exist."""
    from . import e27, e19, linking
    import glob, os
    t0 = time.time(); ann = paths.private_root() / "annotation"
    key = pd.read_csv(ann / "AUDIT_KEY_PRIVATE.csv")
    files = sorted(glob.glob(str(ann / "returned" / "sheet_annotator_*.csv")))
    if len(files) < 3:
        raise SystemExit(f"E27: {len(files)} of 3 returned sheets present in {ann / 'returned'}; nothing to analyse yet.")
    sheets = [pd.read_csv(f, encoding="utf-8-sig") for f in files]
    m = e27.majority(sheets)
    R = m[[c for c in m.columns if c.startswith("r") and c[1:].isdigit()]].to_numpy(float)
    k = key.merge(m[["audit_id", "y_major", "n_rated"]], on="audit_id", how="left")
    gold = k[k.block == "gold"].dropna(subset=["y_major"])
    qa = {"items": int(len(m)), "fully_rated": int((m.n_rated == 3).sum()), "ties_or_missing": int(m.y_major.isna().sum()),
          "fleiss_kappa": e27.fleiss_kappa(R),
          "gold_n": int(len(gold)), "gold_cohen_kappa_vs_banglabait": e27.cohen_kappa(gold.y_major, gold.bb_label),
          "gold_agreement": float(np.mean(gold.y_major == gold.bb_label)),
          "gold_confusion": pd.crosstab(gold.bb_label, gold.y_major).to_dict()}
    for b_ in ("facebook", "twitter", "youtube", "news_nonanchor"):
        r_ = R[m.audit_id.isin(k[k.block == b_].audit_id).to_numpy()]
        qa[f"fleiss_kappa_{b_}"] = e27.fleiss_kappa(r_)
    _write("E27", "audit_quality.json", {kk: (float(v) if isinstance(v, (np.floating,)) else v) for kk, v in qa.items()})
    pool = inference_pool(); anc = anchors(); lv = data.banglaclick_levels().set_index("row_id").level.astype(int)
    sc = pd.read_parquet(paths.private_root() / "predictions" / "E19" / "blind_tfidf_scores.parquet")
    scorers = {"tfidf_lr": sc[["row_id", "env", "seed", "p"]]}
    scorers.update({kk: v for kk, v in _xcell_scorers().items() if "|" not in kk})
    anchor_hash = set(anc.normalized_title_sha256)
    rows = []
    for name, s in scorers.items():
        for sd in sorted(s.seed.unique()):
            ss = s[s.seed == sd]
            d = e19.anchor_items(ss, pool, anc, lv)
            link = linking.fit(d.L, d.z, d.ya, 1.0, flex=True)
            for b_, env in (("facebook", "banglaclick:facebook"), ("twitter", "banglaclick:twitter"),
                            ("youtube", "banglaclick:youtube"), ("news_nonanchor", "banglaclick:news")):
                pop = ss[ss.env == env].merge(pool[["row_id", "normalized_title_sha256"]], on="row_id")
                if b_ == "news_nonanchor":
                    pop = pop[~pop.normalized_title_sha256.isin(anchor_hash)]
                pop = pop.drop_duplicates("row_id").assign(L=lambda x: lv.loc[x.row_id].to_numpy().astype(int))
                smp = k[(k.block == b_)].dropna(subset=["y_major"]).rename(columns={"y_major": "y"})
                r = e27.block_estimate(pop, smp, link); r.update(scorer=name, seed=int(sd), block=b_); rows.append(r)
    est = pd.DataFrame(rows)
    _write("E27", "audit_estimates_by_seed.csv", est)
    summ = est.groupby(["scorer", "block"]).agg(auroc=("auroc", "mean"), auroc_se=("auroc_se", "mean"), nll=("nll", "mean"),
                                                 nll_se=("nll_se", "mean"), prev_hat=("prev_hat", "mean"), n=("n", "first"), N=("N", "first")).reset_index()
    summ["auroc_lo"], summ["auroc_hi"] = summ.auroc - 1.96 * summ.auroc_se, summ.auroc + 1.96 * summ.auroc_se
    _write("E27", "audit_estimates_summary.csv", summ)
    _manifest("E27", t0, {"returned_sheets": [os.path.basename(f) for f in files], "estimator": "difference (Eq. 8), stratified by level",
                          "working_link": "flexible SCL fitted on all news anchors per scorer seed",
                          "seed_aggregation": "mean of estimates and of design SEs over seeds"})


def E26(n_perm=200):
    """Placebo null for the decomposition. Within each target cell, the aligned label is permuted (same prevalence,
    no information about the item). The joint label-and-base-rate component J = R00 - RAP of the placebo label
    measures what a prevalence change alone yields; J_obs - mean(J_placebo) is the part due to the aligned label's
    information. Both proxies (level >= 2, trigger >= 1), NLL and Brier, frozen models, all seeds."""
    t0 = time.time(); a = anchors(); g = construct.linking_function(a)
    df = add_labels(preds(), g); pi_s = pi_source()
    isB = df.regime == "B"; df = df.copy()
    df["y_tr"] = np.where(isB, (df.trigger_level.fillna(0) >= 1).astype(float), np.nan)
    rng = np.random.default_rng(C["bootstrap_seed"]); rows = []
    src = df[df.env == "banglabait:news"]
    for (m, s), sm in src.groupby(["model", "seed"]):
        rs, ra = decompose.source_risk(sm.p_calibrated.to_numpy(), sm.y_r6.to_numpy())
        tg = df[(df.dataset == "banglaclick") & (df.model == m) & (df.seed == s)]
        for env, e in tg.groupby("env"):
            p, y = e.p_calibrated.to_numpy(), e.y_r6.to_numpy()
            for proxy, col in (("level2", "y_al"), ("trigger", "y_tr")):
                ya = e[col].to_numpy()
                c = decompose.components(p, y, ya, pi_s, rs, ra)
                perm = []
                for _ in range(n_perm):
                    cp = decompose.components(p, y, rng.permutation(ya), pi_s, rs, ra)
                    perm.append({k: cp[f"{k}_R00"] - cp[f"{k}_RAP"] for k in ("nll", "brier")})
                perm = pd.DataFrame(perm)
                r = {"env": env, "model": m, "seed": s, "proxy": proxy}
                for k in ("nll", "brier"):
                    j = c[f"{k}_R00"] - c[f"{k}_RAP"]
                    r.update({f"{k}_gap": c[f"{k}_gap"], f"{k}_joint": j, f"{k}_joint_placebo_mean": float(perm[k].mean()),
                              f"{k}_joint_placebo_q975": float(perm[k].quantile(0.975)),
                              f"{k}_joint_excess": j - float(perm[k].mean()),
                              f"{k}_p_perm": float((1 + (perm[k] >= j).sum()) / (1 + n_perm))})
                rows.append(r)
    r = pd.DataFrame(rows)
    _write("E26", "decomposition_placebo_by_seed.csv", r)
    num = [c for c in r.columns if c not in ("env", "model", "seed", "proxy")]
    s_ = r.groupby(["proxy", "env", "model"])[num].mean().reset_index()
    for k in ("nll", "brier"):
        s_[f"{k}_joint_share"] = s_[f"{k}_joint"] / s_[f"{k}_gap"]
        s_[f"{k}_excess_share"] = s_[f"{k}_joint_excess"] / s_[f"{k}_gap"]
        s_[f"{k}_placebo_share"] = s_[f"{k}_joint_placebo_mean"] / s_[f"{k}_gap"]
    _write("E26", "decomposition_placebo_summary.csv", s_)
    _manifest("E26", t0, {"n_perm": n_perm, "pi_source_dev": pi_s, "note": "aligned label permuted within cell; frozen models; seeds averaged in summary"})


# ---------------------------------------------------------------- E7
def E7():
    from . import cues
    t0 = time.time(); a = anchors(); g = construct.linking_function(a)
    it = add_labels(data.load_predictions(with_text=True), g).drop_duplicates("row_id").reset_index(drop=True)
    changes, extra = cues.audit(it)
    _write("E7", "cue_presence_change_by_cell.csv", changes)
    _write("E7", "v10_replica_match.json", extra["v10_replica_exact_match"])
    v11 = extra["v11"]; it11 = it.drop(columns=data.CUES).join(v11)
    _CACHE["cue_v11_items"] = it11
    _manifest("E7", t0)
    E2(cue_frame=it11, tag="_cues_v11")


# ---------------------------------------------------------------- E10 (CPU families)
def training_pool():
    a = anchors(); g = construct.linking_function(a)
    it = add_labels(data.load_predictions(with_text=True), g).drop_duplicates("row_id")
    tg = it[it.env != "banglabait:news"][["row_id", "title", "env", "cluster_id", "normalized_title_sha256", "y_r6", "y_al"]]
    tr = pd.read_parquet(paths.restored("data/derived/primary/stage03/clean_splits/banglabait_train_clean.parquet"))
    src = pd.DataFrame({"row_id": "bbtrain_" + tr.source_row_id.astype(str), "title": tr.title, "env": "banglabait:news", "cluster_id": "bbtitle:" + tr.normalized_title_sha256,
                        "normalized_title_sha256": tr.normalized_title_sha256, "y_r6": tr.label.astype(float), "y_al": tr.label.astype(float)})
    return pd.concat([src, tg], ignore_index=True)


def E10():
    from . import train_cpu
    t0 = time.time(); pool = training_pool()
    res = train_cpu.run_loeo(pool, C["seeds"][:3], pred_dir=paths.private_root() / "predictions" / "E10")
    _write("E10", "loeo_tfidf_results.csv", res)
    num = [c for c in res.columns if c not in ("label", "target", "protocol", "model", "seed")]
    _write("E10", "loeo_tfidf_summary.csv", res.groupby(["label", "target", "protocol", "model"])[num].mean().reset_index())
    _manifest("E10", t0, {"seeds": C["seeds"][:3], "dedup": "training items whose normalised title occurs in the held-out cell are removed"})


def E16():
    """Shift-correction baselines (MLLS prior shift, importance weighting) on the E10 LOEO splits."""
    from . import train_cpu
    t0 = time.time(); pool = training_pool()
    res = train_cpu.run_shift_baselines(pool, C["seeds"][:3], pred_dir=paths.private_root() / "predictions" / "E16")
    _write("E16", "shift_baselines_results.csv", res)
    num = [c for c in res.columns if c not in ("label", "target", "protocol", "model", "seed")]
    _write("E16", "shift_baselines_summary.csv", res.groupby(["label", "target", "protocol", "model"])[num].mean().reset_index())
    _manifest("E16", t0, {"seeds": C["seeds"][:3], "weights": "cross-fitted domain-classifier density ratio, clipped [0.1,10], mean-normalised"})


SYN_MU = 1.0  # score signal strength; see E15 manifest for the implied oracle AUROC


def E15():
    """Synthetic recovery of the decomposition with known ground truth (scope.synthetic)."""
    from . import synthetic
    t0 = time.time(); a = anchors(); g = construct.linking_function(a)
    lv = data.banglaclick_levels().level.astype(int).to_numpy()
    pi_L = np.bincount(lv, minlength=4) / len(lv)
    raw, summ = synthetic.run(g, pi_L, n_anchor_list=[100, 250, 1083], lambdas=[0.0, 0.5, 1.0], deltas=[0.0, 0.5],
                              omegas=[0.0, 0.5, 1.0], reps=200, boot=100, seed=C["bootstrap_seed"], mu=SYN_MU)
    _write("E15", "synthetic_recovery_raw.csv", raw)
    _write("E15", "synthetic_recovery_summary.csv", summ)
    _manifest("E15", t0, {"g_true": [float(x) for x in g], "pi_L": [float(x) for x in pi_L], "mu": SYN_MU,
                          "evaluation_type": "synthetic (known ground truth); not real data"})


def _gather_predictions():
    """All saved LOEO predictions: TF-IDF (E10/E16) and transformers (Kaggle). Returns DataFrame of files."""
    import glob, os
    rows = []
    for f in [x for e in ("E10", "E16", "E22") for x in glob.glob(str(paths.private_root() / "predictions" / e / "*.parquet"))]:
        model, proto, lab, tgt, seed = os.path.basename(f)[:-8].split("__")
        rows.append(dict(model=model, protocol=proto, label=lab, target=tgt.replace("-", ":", 1), seed=int(seed[1:]), file=f, col="p"))
    seen = set()
    for f in sorted(glob.glob(str(paths.private_root() / "kaggle_out" / "full" / "**" / "pred_*.parquet"), recursive=True)):
        b = os.path.basename(f)
        if b in seen:
            continue
        seen.add(b); parts = b[:-8].split("_", 5)
        rec = dict(model=parts[1], protocol=parts[5] if len(parts) > 5 else "pooled_erm", label=parts[2],
                   target=parts[3].replace("-", ":", 1), seed=int(parts[4][1:]), file=f, col="p_calibrated")
        rows.append(rec)
        if rec["protocol"] == "source_only":  # source-only training never sees a regime-B label: one model, scored under both
            rows.append({**rec, "label": "aligned" if rec["label"] == "r6" else "r6"})
    out = pd.DataFrame(rows)
    return out.drop_duplicates(["model", "protocol", "label", "target", "seed"]) if len(out) else out


def E17():
    """Paired cluster-bootstrap comparison of protocols vs source-only (scope.compare)."""
    from . import compare
    t0 = time.time(); pool = inference_pool().set_index("row_id"); files = _gather_predictions()
    rng = np.random.default_rng(C["bootstrap_seed"]); out = []
    for (model, lab, tgt), g in files[files.protocol != "majority"].groupby(["model", "label", "target"]):
        ref = "source_only" if "source_only" in set(g.protocol) else "pooled_erm"
        if ref not in set(g.protocol) or g.protocol.nunique() < 2:
            continue
        ref_ids = pd.read_parquet(g.iloc[0].file).row_id.to_numpy()
        preds = {}
        for _, r in g.iterrows():
            d = pd.read_parquet(r.file).set_index("row_id").loc[ref_ids]
            preds.setdefault(r.protocol, []).append(d[r.col].to_numpy())
        ycol = "y_r6" if lab == "r6" else "y_al"
        res = compare.cell(preds, pool.loc[ref_ids, ycol].to_numpy(), pool.loc[ref_ids, "cluster_id"].to_numpy(),
                           ref, 1000, rng)
        res.insert(0, "target", tgt); res.insert(0, "label", lab); res.insert(0, "model", model); out.append(res)
        if {"latent_cuts", "pooled_erm"} <= set(preds):  # E22 vs the model it nests (equal cuts)
            r2 = compare.cell({k: preds[k] for k in ("pooled_erm", "latent_cuts")}, pool.loc[ref_ids, ycol].to_numpy(),
                              pool.loc[ref_ids, "cluster_id"].to_numpy(), "pooled_erm", 1000, rng)
            r2.insert(0, "target", tgt); r2.insert(0, "label", lab); r2.insert(0, "model", model); out.append(r2)
    res = pd.concat(out, ignore_index=True)
    # Holm family: baseline contrasts within a metric. in_domain and source_oracle use target labels and are
    # references, so they are excluded from the family; their p_holm is left empty.
    res["p_holm"] = np.nan
    fam = ~res.protocol.isin(["in_domain", "source_oracle"]) & res.p_wald.notna()
    for m, g in res[fam].groupby("metric"):
        res.loc[g.index, "p_holm"] = compare.holm(g.p_wald.to_numpy())
    _write("E17", "protocol_contrasts_cluster_bootstrap.csv", res)
    _manifest("E17", t0, {"draws": 1000, "reference": "source_only where available, else pooled_erm", "multiplicity": "Holm within metric over baseline contrasts (in_domain, source_oracle excluded), on Wald p from bootstrap SE",
                          "prediction_sets": int(len(files))})


def E18():
    """Real-data check of the target-label proxies on anchors, where the source label Y_A is observed.

    Uses LOEO models whose training excluded every title of the held-out BanglaClick news cell, so no
    model saw the anchors. For each prediction set on banglaclick:news restricted to anchor items:
    metrics of p against Y_A (truth), against the R6 label 1[L>=1], the hard aligned label 1[L>=t*], and
    the soft label g_hat(L) cross-fitted over 2 folds of anchor classes (g_hat never estimated on the
    item it labels). Also tests whether p predicts Y_A beyond L (logistic Y_A ~ C(L) + logit p, cluster-
    robust by anchor class): a nonzero slope means the soft proxy is not exact (Proposition 2).
    """
    import statsmodels.api as sm
    t0 = time.time(); pool = training_pool().set_index("row_id"); a = anchors().set_index("normalized_title_sha256")
    lv = data.banglaclick_levels().set_index("row_id").level.astype(int)
    files = _gather_predictions(); files = files[(files.target == "banglaclick:news") & ~files.protocol.isin(["majority", "in_domain"])]  # in_domain trains on target labels
    rng = np.random.default_rng(C["bootstrap_seed"]); rows, tests = [], []
    for _, r in files.iterrows():
        d = pd.read_parquet(r.file)
        h = pool.loc[d.row_id, "normalized_title_sha256"].to_numpy(); m = np.isin(h, a.index)
        d = d[m].copy(); d["h"] = h[m]; d["L"] = lv.loc[d.row_id].to_numpy(); d["ya"] = a.loc[d.h, "bb_label"].to_numpy().astype(float)
        p = np.clip(d[r.col].to_numpy(), 1e-6, 1 - 1e-6)
        cls = d.h.unique(); fold = dict(zip(cls, rng.permutation(len(cls)) % 2)); fd = d.h.map(fold).to_numpy()
        q = np.empty(len(d)); thr = {}
        for k in (0, 1):
            tr = a.loc[[c for c in cls if fold[c] != k]]
            g = construct.linking_function(tr); q[fd == k] = g[d.L.to_numpy()[fd == k]]
            thr[k] = max((1, 2, 3), key=lambda t: M.kappa(tr.bb_label, (tr.bc_level >= t).astype(int)))
        hard = np.where(fd == 0, d.L.to_numpy() >= thr[0], d.L.to_numpy() >= thr[1]).astype(float)
        labels = {"truth_YA": d.ya.to_numpy(), "r6": (d.L.to_numpy() >= 1).astype(float), "hard_aligned": hard, "soft_crossfit": q}
        for name, y in labels.items():
            out = {"model": r.model, "protocol": r.protocol, "train_label": r.label, "seed": r.seed, "proxy": name, "n": len(d)}
            out.update({k: f(y, p) for k, f in {"auroc": M.auroc, "nll": M.nll, "brier": M.brier}.items()}); rows.append(out)
        X = pd.get_dummies(pd.Categorical(d.L, categories=[0, 1, 2, 3]), prefix="L", drop_first=True).astype(float)
        X["logit_p"] = np.log(p / (1 - p)); X = sm.add_constant(X)
        fit = sm.Logit(d.ya.to_numpy(), X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(d.h)[0]})
        tests.append({"model": r.model, "protocol": r.protocol, "train_label": r.label, "seed": r.seed,
                      "beta_logit_p_given_L": fit.params["logit_p"], "se": fit.bse["logit_p"], "p_value": fit.pvalues["logit_p"], "n": len(d)})
    res = pd.DataFrame(rows)
    for met in ("auroc", "nll", "brier"):
        tru = res[res.proxy == "truth_YA"].set_index(["model", "protocol", "train_label", "seed"])[met]
        res[met + "_bias_vs_truth"] = res[met].to_numpy() - tru.loc[list(zip(res.model, res.protocol, res.train_label, res.seed))].to_numpy()
    _write("E18", "anchor_proxy_check.csv", res)
    _write("E18", "anchor_proxy_check_summary.csv", res.groupby(["model", "protocol", "train_label", "proxy"]).mean(numeric_only=True).reset_index())
    _write("E18", "score_beyond_level_test.csv", pd.DataFrame(tests))
    _manifest("E18", t0, {"crossfit_folds": 2, "models": "LOEO predictions on banglaclick:news; training excluded every title of that cell"})


# ---------------------------------------------------------------- E10T (GPU transformer predictions pulled from Kaggle)
def E10T():
    """Score transformer LOEO predictions with the same metric code as every other method."""
    import glob, os
    t0 = time.time(); pool = training_pool().set_index("row_id")
    files = sorted(set(glob.glob(str(paths.private_root() / "kaggle_out" / "full" / "**" / "pred_*.parquet"), recursive=True)))
    seen = {}
    for f in files:  # the same prediction may be pulled twice (e.g. from a collaborator copy); keep one per name
        seen.setdefault(os.path.basename(f), f)
    files = sorted(seen.values())
    rows = []; rep = []
    lv = data.banglaclick_levels().set_index("row_id").level
    anc_h = set(anchors().normalized_title_sha256)
    for f in files:
        parts = os.path.basename(f)[:-8].split("_", 5)  # pred_{model}_{label}_{target}_s{seed}[_{protocol}]
        _, model, lab, tgt, seed = parts[:5]; proto = parts[5] if len(parts) > 5 else "pooled_erm"
        tgt = tgt.replace("-", ":", 1); seed = int(seed[1:])
        p = pd.read_parquet(f)
        for lab_s in (("r6", "aligned") if proto == "source_only" else (lab,)):  # source-only: one model, both labels
            y = pool.loc[p.row_id, "y_r6" if lab_s == "r6" else "y_al"].to_numpy()
            r = M.all_metrics(y, p.p_calibrated.to_numpy(), hard=y)
            r.update(label=lab_s, target=tgt, protocol=proto, model=model, seed=seed); rows.append(r)
        if tgt == "banglaclick:news" and proto == "pooled_erm":
            h = pool.loc[p.row_id, "normalized_title_sha256"].isin(anc_h).to_numpy()
            for l in range(4):
                m = (lv.loc[p.row_id].to_numpy() == l)
                for a in (True, False):
                    mm = m & (h == a)
                    rep.append({"model": model, "label": lab, "seed": seed, "level": l, "anchor": a,
                                "mean_p": float(p.p_calibrated.to_numpy()[mm].mean()) if mm.any() else float("nan"), "n": int(mm.sum())})
    res = pd.DataFrame(rows)
    _write("E10", "loeo_transformer_results.csv", res)
    num = [c for c in res.columns if c not in ("label", "target", "protocol", "model", "seed")]
    _write("E10", "loeo_transformer_summary.csv", res.groupby(["label", "target", "protocol", "model"])[num].mean().reset_index())
    _write("E10", "anchor_representativeness_dedup_models.csv", pd.DataFrame(rep))
    _manifest("E10_transformer", t0, {"prediction_files": len(files), "metric_code": "scope.metrics.all_metrics (same as all methods)"})


# ---------------------------------------------------------------- E14: is the label a cluster-level label?
def E14():
    """Between-cluster share of label variance (eta^2) and share of items in label-pure clusters."""
    t0 = time.time(); a = anchors(); g = construct.linking_function(a)
    it = add_labels(preds(), g).drop_duplicates("row_id")
    rows = []
    for env, e in it[it.env != "banglabait:news"].groupby("env"):
        for lab in ("y_r6", "y_al"):
            if lab == "y_al" and e.regime.iloc[0] != "B":
                continue
            y = e[lab].to_numpy(); cm = e.groupby("cluster_id")[lab].transform("mean").to_numpy()
            tot = np.sum((y - y.mean()) ** 2)
            eta2 = float(np.sum((cm - y.mean()) ** 2) / tot) if tot > 0 else float("nan")
            pure = float(np.mean((cm == 0) | (cm == 1)))
            big = e.groupby("cluster_id").size(); top = float(big.max() / len(e))
            rows.append({"env": env, "label": lab, "clusters": int(e.cluster_id.nunique()), "eta2_between_cluster": eta2,
                         "share_items_in_pure_clusters": pure, "largest_cluster_share": top})
    _write("E14", "cluster_label_homogeneity.csv", pd.DataFrame(rows))
    _manifest("E14", t0)


# ---------------------------------------------------------------- E11
def E11():
    t0 = time.time(); a = anchors(); g = construct.linking_function(a)
    df = add_labels(preds(), g); pi_s = pi_source()
    rows = []
    src = df[df.env == "banglabait:news"]
    for env, e_all in df[df.env != "banglabait:news"].groupby("env"):
        for m in C["models"]:
            e = e_all[(e_all.model == m) & (e_all.seed == 42)]
            s = src[(src.model == m) & (src.seed == 42)]
            p = priors.prior_adjust(e.p_calibrated.to_numpy(), pi_s, priors.mlls_em(e.p_calibrated.to_numpy(), pi_s))
            tau = M.brier(s.y_r6.to_numpy(), s.p_calibrated.to_numpy()) + 0.03
            for lab in ("y_r6", "y_al"):
                if lab == "y_al" and e.regime.iloc[0] != "B":
                    continue
                for r in audit.run_audit(p, e[lab].to_numpy(), tau, C["audit_budgets"], C["audit_replications"], C["bootstrap_seed"]):
                    r.update(env=env, model=m, label=lab); rows.append(r)
    _write("E11", "audit_results.csv", pd.DataFrame(rows))
    _manifest("E11", t0, {"seed_model": 42, "tolerance": "tau = source Brier + 0.03",
                          "deployed_model": "temperature-scaled source model, MLLS prior-corrected"})


# ---------------------------------------------------------------- E13
def E13():
    t0 = time.time()
    rows = fewcluster.run(C["cluster_sim_G"], C["cluster_sim_reps"], C["bootstrap_seed"])
    _write("E13", "few_cluster_size.csv", pd.DataFrame(rows))
    _manifest("E13", t0, {"simulation": True, "reps": C["cluster_sim_reps"], "bootstrap_draws_per_rep": 399})


def _source_fit_prevalence(seed):
    """Training prevalence of a news-fold source-only model: BanglaBait pool minus the news cell's titles, fit part
    of the same cluster-grouped 90/10 split that the GPU kernel and e19.blind_tfidf use (training clusters)."""
    key = ("pi_fit", seed)
    if key not in _CACHE:
        from sklearn.model_selection import GroupShuffleSplit
        pool = training_pool(); news = pool[pool.env == "banglaclick:news"]
        tr = pool[pool.env == "banglabait:news"]; tr = tr[~tr.normalized_title_sha256.isin(news.normalized_title_sha256)]
        fit_i, _ = next(GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed).split(tr, groups=tr.cluster_id))
        _CACHE[key] = float(tr.iloc[fit_i].y_r6.mean())
    return _CACHE[key]


def _xcell_scorers():
    """Transformer anchor-blind scorers from GPU source-only units (xcell_ + news pred_ files), if present."""
    import glob, os
    out = {}
    for f in sorted(glob.glob(str(paths.private_root() / "kaggle_out" / "full" / "**" / "xcell_*_source_only.parquet"), recursive=True)):
        parts = os.path.basename(f)[:-8].split("_")  # xcell_{model}_{label}_heldout-..._s{seed}_source_only
        model, seed = parts[1], int(parts[4][1:])
        news = glob.glob(str(paths.private_root() / "kaggle_out" / "full" / "**" / f"pred_{model}_{parts[2]}_banglaclick-news_s{seed}_source_only.parquet"), recursive=True)
        if not news:
            continue
        x = pd.read_parquet(f)[["row_id", "env", "p_calibrated"]]
        n = pd.read_parquet(news[0])[["row_id", "p_calibrated"]].assign(env="banglaclick:news")
        parts_ = [x, n]
        # Source held-out reference (w11/w12 reruns): the same model on its cluster-grouped validation split of
        # BanglaBait. Deterministic fine-tuning makes it the same model as the xcell_ file (checked in E20).
        sv = glob.glob(str(paths.private_root() / "kaggle_out" / "full" / "**" / os.path.basename(f).replace("xcell_", "srcval_")), recursive=True)
        if sv:
            parts_.append(pd.read_parquet(sv[0])[["row_id", "p_calibrated"]].assign(env="banglabait:news"))
        fr = pd.concat(parts_, ignore_index=True).rename(columns={"p_calibrated": "p"}).assign(seed=seed, pi_train=_source_fit_prevalence(seed))
        out.setdefault(model, []).append(fr.drop_duplicates("row_id"))
    out = {k: pd.concat(v, ignore_index=True) for k, v in out.items()}
    # Validation-only scorers: every LOEO prediction set on the news cell (training excluded all its titles,
    # hence every anchor). They enter V1-V3 only; the application needs the same model on every cell.
    f = _gather_predictions()
    f = f[(f.target == "banglaclick:news") & f.protocol.isin(["pooled_erm", "group_dro"])]
    for (m, pr, lab), g in f.groupby(["model", "protocol", "label"]):
        fr = [pd.read_parquet(r.file)[["row_id", r.col]].rename(columns={r.col: "p"}).assign(env="banglaclick:news", seed=r.seed)
              for r in g.itertuples()]
        out[f"{m}|{pr}|{lab}"] = pd.concat(fr, ignore_index=True)
    return out


def E19():
    """Score-conditional anchor linking: validation on anchors and application to every regime-B cell."""
    from . import e19
    t0 = time.time(); pool = training_pool(); anc = anchors()
    lv = data.banglaclick_levels().set_index("row_id").level.astype(int)
    st = data.load_predictions(with_text=True); st = st[st.env == "banglabait:news"].drop_duplicates("row_id")
    src_test = st[["row_id", "env", "title"]]; src_y = st.set_index("row_id").label.astype(float)
    pa = pool[pool.env == "banglabait:news"].set_index("row_id").y_r6.astype(float); src_y = pd.concat([src_y, pa[~pa.index.isin(src_y.index)]])  # transformer scorers: source held-out rows come from the training pool
    sc, v1, v2, v3, pt, bt, gaps = e19.run(pool, src_test, anc, lv, src_y, C["seeds"][:3], C["bootstrap_seed"], _write,
                                    boot=300, extra_scorers=_xcell_scorers(), pool_inf=inference_pool())
    # reproduction check: blind TF-IDF on the news cell == E10 source-only news predictions
    chk = []
    for s in C["seeds"][:3]:
        ref = pd.read_parquet(paths.private_root() / "predictions" / "E10" / f"tfidf_lr__source_only__r6__banglaclick-news__s{s}.parquet").set_index("row_id").p
        mine = sc[(sc.seed == s) & (sc.env == "banglaclick:news")].set_index("row_id").p.loc[ref.index]
        chk.append(float(np.abs(mine.to_numpy() - ref.to_numpy()).max()))
    pdir = paths.private_root() / "predictions" / "E19"; pdir.mkdir(parents=True, exist_ok=True)
    sc.to_parquet(pdir / "blind_tfidf_scores.parquet", index=False)
    _write("E19", "v1_crossfit_anchor_validation.csv", v1)
    _write("E19", "v1_summary.csv", v1.groupby(["scorer", "proxy"]).agg(
        **{f"{m}_bias_mean": (f"{m}_bias", "mean") for m in ("auroc", "nll", "brier")},
        **{f"{m}_bias_sd": (f"{m}_bias", "std") for m in ("auroc", "nll", "brier")},
        label_nll_vs_YA=("label_nll_vs_YA", "mean")).reset_index())
    raw_dir = paths.private_root() / "results_raw" / "E19"; raw_dir.mkdir(parents=True, exist_ok=True)  # per-draw files are large; summaries stay in the repo
    v2.to_csv(raw_dir / "v2_selection_stress.csv", index=False, float_format="%.6g")
    _write("E19", "v2_summary.csv", v2.groupby(["scorer", "kappa_score", "delta_label", "proxy"]).agg(
        **{f"{m}_bias_mean": (f"{m}_bias", "mean") for m in ("auroc", "nll", "brier", "prev")},
        **{f"{m}_rmse": (f"{m}_bias", lambda v: float(np.sqrt(np.mean(np.square(v))))) for m in ("auroc", "nll", "prev")}).reset_index())
    v3.to_csv(raw_dir / "v3_anchor_budget.csv", index=False, float_format="%.6g")
    _write("E19", "selection_gap_calibration.csv", gaps)
    _write("E19", "v3_summary.csv", v3.groupby(["scorer", "n_anchors", "proxy"]).agg(
        auroc_bias_mean=("auroc_bias", "mean"), auroc_sd=("auroc_bias", "std"), nll_bias_mean=("nll_bias", "mean"),
        nll_sd=("nll_bias", "std"), auroc_rmse=("auroc_bias", lambda v: float(np.sqrt(np.mean(np.square(v)))))).reset_index())
    _write("E19", "application_by_seed.csv", pt)
    summ = pt.groupby(["scorer", "env", "label"]).mean(numeric_only=True).reset_index().drop(columns="seed")
    q = lambda v, a: float(np.nanpercentile(v, a))
    ci = bt.groupby(["scorer", "env", "label"]).agg(**{f"{m}_{k}": (m, (lambda v, a=a: q(v, a))) for m in
                                                     ("auroc", "nll", "prevalence", "nll_residual_after_prior") for k, a in (("lo", 2.5), ("hi", 97.5))}).reset_index()
    _write("E19", "application_summary.csv", summ.merge(ci, on=["scorer", "env", "label"]))
    # paired contrasts SCL(gamma=1) - hard aligned, from the same bootstrap draws
    w = bt.pivot_table(index=["scorer", "env", "draw"], columns="label", values=["auroc", "nll"])
    con = []
    for (scn, env), g in w.groupby(level=[0, 1]):
        for m in ("auroc", "nll"):
            for ref in ("hard_aligned", "scl_g0", "scl_lin_g1"):
                dlt = g[(m, "scl_g1")] - g[(m, ref)]
                con.append({"scorer": scn, "env": env, "metric": m, "contrast": f"scl_g1 - {ref}", "mean": float(dlt.mean()),
                            "lo": q(dlt, 2.5), "hi": q(dlt, 97.5)})
    _write("E19", "application_contrasts.csv", pd.DataFrame(con))
    _manifest("E19", t0, {"scorers": sorted(set(sc.scorer)) + sorted(_xcell_scorers()), "seeds": C["seeds"][:3], "gammas": [0, .25, .5, .75, 1],
                          "v1": "5-fold publisher-cluster cross-fit on anchors, 20 fold draws", "v3": "anchor budget 50-800, cluster-wise subsamples, 200 draws", "v2": "200 draws; kappa_score in {0,1,2}, delta_label in {0,1}; 60% target fit share",
                          "bootstrap": "300 draws; anchor publisher clusters (link refit) x target clusters; seed-averaged",
                          "repro_check_max_abs_diff_vs_E10_source_only_news": max(chk),
                          "primary_link": "flexible score function (linear spline, hinges at anchor quartiles of the logit); chosen because the linear link "
                                          "is misspecified for bimodal transformer logits (diagnosed in V2); both are reported, and V1 gives the cross-fitted anchor NLL of each"})


def E20():
    """Sharp partial-identification bounds for source-construct risk on regime-B cells (scope.bounds, scope.e20)."""
    from . import e20
    t0 = time.time(); pool = inference_pool(); anc = anchors(); rng = np.random.default_rng(C["bootstrap_seed"] + 20)
    lv = data.banglaclick_levels().set_index("row_id").level.astype(int)
    st = data.load_predictions(); src_y = st[st.env == "banglabait:news"].drop_duplicates("row_id").set_index("row_id").label.astype(float)
    # BanglaBait training-pool labels (regime A; identical under both regime-B label definitions) for the transformer
    # scorers' source held-out split
    pa = pool[pool.env == "banglabait:news"].set_index("row_id").y_r6.astype(float)
    src_y = pd.concat([src_y, pa[~pa.index.isin(src_y.index)]])
    sc = pd.read_parquet(paths.private_root() / "predictions" / "E19" / "blind_tfidf_scores.parquet")  # E19's anchor-blind scorer
    scorers = {"tfidf_lr": sc[["row_id", "env", "seed", "p"]]}
    scorers.update({k: v for k, v in _xcell_scorers().items() if "|" not in k})  # transformer source-only scorers, when returned
    mc, va, pt, bt, rel = e20.run(scorers, pool, anc, lv, src_y, rng, boot=300)
    _write("E20", "m_check.csv", mc)
    _write("E20", "validation_by_fold.csv", va)
    _write("E20", "validation_summary.csv", va.groupby(["scorer", "metric"]).agg(
        coverage_AL=("in_AL", "mean"), coverage_ALM=("in_ALM", "mean"), width_AL=("width_AL", "mean"), width_ALM=("width_ALM", "mean"),
        mean_miss_ALM=("dist_ALM", "mean"), truth_minus_const=("truth_minus_const", "mean"), n_folds=("truth", "size")).reset_index())
    _write("E20", "identified_sets_by_seed.csv", pt)
    summ = pt.groupby(["scorer", "env"]).mean(numeric_only=True).reset_index().drop(columns="seed")
    q = lambda v, a: float(np.nanpercentile(v, a))
    cs = []
    for (scn, env), g in bt.groupby(["scorer", "env"]):
        row = {"scorer": scn, "env": env}
        for m in ("auroc", "nll", "brier"):
            for a in ("AL", "ALM"):
                row[f"{m}_{a}_cs_lo"] = q(g[f"{m}_{a}_lo"], 2.5); row[f"{m}_{a}_cs_hi"] = q(g[f"{m}_{a}_hi"], 97.5)
        row["prevalence_AL_lo"], row["prevalence_AL_hi"] = q(g.prevalence_AL, 2.5), q(g.prevalence_AL, 97.5)
        cs.append(row)
    _write("E20", "identified_sets_summary.csv", summ.merge(pd.DataFrame(cs), on=["scorer", "env"]))
    r = rel[rel.metric != "auroc_breakdown_vs_source"]
    _write("E20", "relaxation_summary.csv", r.groupby(["scorer", "env", "metric", "delta"]).mean(numeric_only=True).reset_index().drop(columns="seed"))
    bk = rel[rel.metric == "auroc_breakdown_vs_source"]
    _write("E20", "breakdown_vs_source_auroc.csv", bk[["scorer", "env", "seed", "side", "delta_star"]])
    _manifest("E20", t0, {"scorers": sorted(scorers), "bootstrap": "300 draws; anchor publisher clusters (link refit) x target clusters; seed-averaged; "
                          "confidence set = [2.5th pct of lower end, 97.5th pct of upper end]",
                          "assumptions": {"A0": "none", "AL": "E_e[h|L=l] = g(l) (level-link transport)", "M": "h non-decreasing in score within level"},
                          "relaxation": "logit g(l) +- delta, delta in {0, .25, .5, 1}; AUROC by 9-point grid per level",
                          "validation": "5-fold publisher-cluster cross-fit on news anchors, 20 draws"})


def E21():
    """Design-based target auditing: model-assisted anchor sampling, validated on news anchors (truth known)."""
    from . import e21, e19
    t0 = time.time(); pool = inference_pool(); anc = anchors(); rng = np.random.default_rng(C["bootstrap_seed"] + 21)
    lv = data.banglaclick_levels().set_index("row_id").level.astype(int)
    sc = pd.read_parquet(paths.private_root() / "predictions" / "E19" / "blind_tfidf_scores.parquet")
    scorers = {"tfidf_lr": sc[["row_id", "env", "seed", "p"]]}
    scorers.update({k: v for k, v in _xcell_scorers().items() if "|" not in k})
    sims = []
    for name, s in scorers.items():
        for sd in sorted(s.seed.unique()):
            d = e19.anchor_items(s[s.seed == sd], pool, anc, lv)
            r = e21.simulate(d, [25, 50, 100, 150, 200], 300, rng); r["scorer"], r["seed"] = name, sd; sims.append(r)
    sim = pd.concat(sims, ignore_index=True)
    sim[["auroc_cover", "nll_cover"]] = sim[["auroc_cover", "nll_cover"]].astype(float)  # bool/NaN mix is object dtype otherwise
    raw_dir = paths.private_root() / "results_raw" / "E21"; raw_dir.mkdir(parents=True, exist_ok=True)
    sim.to_csv(raw_dir / "design_simulation.csv", index=False, float_format="%.6g")
    rmse = lambda v: float(np.sqrt(np.nanmean(np.square(v))))
    _write("E21", "design_summary.csv", sim.groupby(["scorer", "n", "design", "estimator"]).agg(
        auroc_bias=("auroc_err", "mean"), auroc_rmse=("auroc_err", rmse), nll_bias=("nll_err", "mean"), nll_rmse=("nll_err", rmse),
        auroc_ci_coverage=("auroc_cover", "mean"), nll_ci_coverage=("nll_cover", "mean"), draws=("auroc_err", "size"), N=("N", "mean")).reset_index())
    _manifest("E21", t0, {"scorers": sorted(scorers), "population": "news anchors of a random half of publisher clusters per draw; "
                          "working link = flexible SCL fitted on the other half", "draws_per_seed": 300, "budgets": [25, 50, 100, 150, 200],
                          "designs": ["srs", "level_prop", "level_x_score_neyman"], "estimators": ["sample_only", "model_only", "diff", "scl_refit"],
                          "ci": "95% normal, stratified design variance with fpc; AUROC by linearisation"})


def E22():
    """Latent-intensity model with regime-specific cut-points (scope.latent), TF-IDF features, E10 LOEO splits."""
    from . import latent, train_cpu
    from scipy.special import expit
    from sklearn.model_selection import GroupShuffleSplit
    t0 = time.time(); pool = training_pool()
    lv = data.banglaclick_levels().set_index("row_id").level.astype(int)
    pool["regime"] = pool.env.str.split(":").str[0].map({"banglabait": 0, "banglaclick": 1, "baitbuster": 2})
    pool["yy"] = pool.y_r6
    b = pool.regime == 1; pool.loc[b, "yy"] = lv.loc[pool.loc[b, "row_id"]].to_numpy().astype(float)
    construct_of = {("r6", 1): "B1", ("aligned", 1): "B2"}
    rows, cuts = [], []
    pdir = paths.private_root() / "predictions" / "E22"
    for tgt in train_cpu.TARGETS:
        test = pool[pool.env == tgt]; treg = int(test.regime.iloc[0])
        train = pool[pool.env != tgt]; train = train[~train.normalized_title_sha256.isin(test.normalized_title_sha256)]
        for seed in C["seeds"][:3]:
            fit_i, val_i = next(GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed).split(train, groups=train.cluster_id))
            tr, va = train.iloc[fit_i], train.iloc[val_i]
            feats = train_cpu.featurizer(); Xtr = feats.fit_transform(tr.title)
            m = latent.fit(Xtr, tr.regime.to_numpy(), tr.yy.to_numpy())
            Xva, Xte = feats.transform(va.title), feats.transform(test.title)
            cuts.append({"target": tgt, "seed": seed, "tau_A": m["tau_A"], "tau_C": m["tau_C"], "has_C": m["has_C"],
                         **{f"theta_{k + 1}": m["theta"][k] for k in range(3)}, "converged": m["converged"]})
            for lab in ("r6", "aligned"):
                con = construct_of.get((lab, treg), "A" if treg == 0 else "C")
                # temperature on the validation split, each item under its own regime's binary construct
                fv = Xva @ m["w"]; reg_v = va.regime.to_numpy(); yv = va.yy.to_numpy()
                tcut = {"r6": 1, "aligned": 2}[lab]
                zv = np.where(reg_v == 0, fv - latent.cut(m, "A"), np.where(reg_v == 2, fv - latent.cut(m, "C"), fv - latent.cut(m, f"B{tcut}")))
                ybin = np.where(reg_v == 1, (yv >= tcut).astype(float), yv)
                T = train_cpu.fit_temperature(zv, ybin)
                p = expit(latent.score(m, Xte, con) / T)
                ycol = "y_r6" if lab == "r6" else "y_al"; y = test[ycol].to_numpy()
                r = M.all_metrics(y, p, hard=y)
                r.update(label=lab, target=tgt, protocol="latent_cuts", model="tfidf_lr", seed=seed, construct=con,
                         cut_identified=bool(treg != 2 or m["has_C"]))
                rows.append(r)
                train_cpu._save_pred(pdir, "tfidf_lr", "latent_cuts", lab, tgt, seed, test.row_id, p)
    res = pd.DataFrame(rows)
    _write("E22", "latent_cuts_results.csv", res)
    _write("E22", "cut_points.csv", pd.DataFrame(cuts))
    num = [c for c in res.columns if c not in ("label", "target", "protocol", "model", "seed", "construct", "cut_identified")]
    _write("E22", "latent_cuts_summary.csv", res.groupby(["label", "target", "protocol", "model", "construct"])[num].mean().reset_index())
    _manifest("E22", t0, {"seeds": C["seeds"][:3], "splits": "identical to E10 (same dedup, GroupShuffleSplit seed, featuriser)",
                          "penalty": "||w||^2/2 (liblinear C=1 scale)", "optimizer": "L-BFGS-B, maxiter 500",
                          "baitbuster_target": "regime C absent from training when held out: source cut tau_A used (flagged cut_identified=False)"})


def E23():
    """Ablations of the regime cut-point model (E22), TF-IDF features, identical LOEO splits and seeds.

    regime_bias     shared w, one intercept per regime, regime-B labels binarised at the evaluated cut (one fit per
                    label). Removes the ordinal structure: tests whether the rubric levels matter beyond regime offsets.
    latent_noanchor E22 model with every title that is labelled by two regimes in the training set removed. Tests
                    whether the common items (anchors) matter for the fitted model.
    same_regime     plain logistic regression on the other cells of the target's own regime only (regime-B targets).
                    Tests whether cross-corpus data helps at all once the label definition is held fixed.
    Contrasts: each ablation vs latent_cuts (E22) by paired cluster bootstrap; Holm within metric over this family.
    """
    from . import latent, train_cpu, compare
    from scipy.special import expit
    from sklearn.model_selection import GroupShuffleSplit
    t0 = time.time(); pool = training_pool()
    lv = data.banglaclick_levels().set_index("row_id").level.astype(int)
    pool["regime"] = pool.env.str.split(":").str[0].map({"banglabait": 0, "banglaclick": 1, "baitbuster": 2})
    pool["yy"] = pool.y_r6
    b = pool.regime == 1; pool.loc[b, "yy"] = lv.loc[pool.loc[b, "row_id"]].to_numpy().astype(float)
    construct_of = {("r6", 1): "B1", ("aligned", 1): "B2"}
    rows = []; pdir = paths.private_root() / "predictions" / "E23"
    for tgt in train_cpu.TARGETS:
        test = pool[pool.env == tgt]; treg = int(test.regime.iloc[0])
        train = pool[pool.env != tgt]; train = train[~train.normalized_title_sha256.isin(test.normalized_title_sha256)]
        for seed in C["seeds"][:3]:
            fit_i, val_i = next(GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed).split(train, groups=train.cluster_id))
            tr, va = train.iloc[fit_i], train.iloc[val_i]
            feats = train_cpu.featurizer(); Xtr = feats.fit_transform(tr.title)
            Xva, Xte = feats.transform(va.title), feats.transform(test.title)
            reg_tr, reg_va = tr.regime.to_numpy(), va.regime.to_numpy()

            def record(proto, lab, p, extra=None):
                ycol = "y_r6" if lab == "r6" else "y_al"; y = test[ycol].to_numpy()
                r = M.all_metrics(y, p, hard=y); r.update(label=lab, target=tgt, protocol=proto, model="tfidf_lr", seed=seed, **(extra or {}))
                rows.append(r); train_cpu._save_pred(pdir, "tfidf_lr", proto, lab, tgt, seed, test.row_id, p)

            # (1) latent model without anchors
            h = tr.groupby("normalized_title_sha256").regime.nunique(); multi = set(h[h > 1].index)
            keep = ~tr.normalized_title_sha256.isin(multi).to_numpy()
            m = latent.fit(Xtr[keep], reg_tr[keep], tr.yy.to_numpy()[keep])
            for lab in ("r6", "aligned"):
                con = construct_of.get((lab, treg), "A" if treg == 0 else "C"); tcut = {"r6": 1, "aligned": 2}[lab]
                fv = Xva @ m["w"]; yv = va.yy.to_numpy()
                zv = np.where(reg_va == 0, fv - latent.cut(m, "A"), np.where(reg_va == 2, fv - latent.cut(m, "C"), fv - latent.cut(m, f"B{tcut}")))
                T = train_cpu.fit_temperature(zv, np.where(reg_va == 1, (yv >= tcut).astype(float), yv))
                record("latent_noanchor", lab, expit(latent.score(m, Xte, con) / T), {"anchors_removed": int((~keep).sum())})
            # (2) regime offsets only (binary labels), one fit per label definition
            for lab in ("r6", "aligned"):
                tcut = {"r6": 1, "aligned": 2}[lab]
                ytr = np.where(reg_tr == 1, (tr.yy.to_numpy() >= tcut).astype(float), tr.yy.to_numpy())
                mb = latent.fit_bias(Xtr, reg_tr, ytr)
                yv = va.yy.to_numpy(); yvb = np.where(reg_va == 1, (yv >= tcut).astype(float), yv)
                T = train_cpu.fit_temperature(latent.score_bias(mb, Xva, reg_va), yvb)
                record("regime_bias", lab, expit(latent.score_bias(mb, Xte, np.full(Xte.shape[0], treg)) / T),
                       {"cut_identified": bool(treg != 2 or mb["has_C"])})
            # (3) same-regime training only (regime-B targets)
            if treg == 1:
                s_tr, s_va = reg_tr == 1, reg_va == 1
                for lab in ("r6", "aligned"):
                    tcut = {"r6": 1, "aligned": 2}[lab]
                    lr = train_cpu.fit_lr(Xtr[s_tr], (tr.yy.to_numpy()[s_tr] >= tcut).astype(int), seed=seed)
                    T = train_cpu.fit_temperature(lr.decision_function(Xva[s_va]), (va.yy.to_numpy()[s_va] >= tcut).astype(float))
                    record("same_regime", lab, expit(lr.decision_function(Xte) / T))
    res = pd.DataFrame(rows)
    _write("E23", "ablation_results.csv", res)
    num = [c for c in res.columns if c not in ("label", "target", "protocol", "model", "seed", "cut_identified")]
    _write("E23", "ablation_summary.csv", res.groupby(["label", "target", "protocol", "model"])[num].mean().reset_index())
    _manifest("E23", t0, {"seeds": C["seeds"][:3], "splits": "identical to E10/E22"})
    E23C()


def E23C():
    """Paired cluster-bootstrap contrasts of the E23 ablations vs the full cut-point model (publisher clusters)."""
    from . import compare
    t0 = time.time(); pdir = paths.private_root() / "predictions" / "E23"
    files = _gather_predictions(); files = files[files.model == "tfidf_lr"]
    import glob, os
    e23 = []
    for f in glob.glob(str(pdir / "*.parquet")):
        model, proto, lab, tgt, sd = os.path.basename(f)[:-8].split("__")
        e23.append(dict(model=model, protocol=proto, label=lab, target=tgt.replace("-", ":", 1), seed=int(sd[1:]), file=f, col="p"))
    files = pd.concat([files, pd.DataFrame(e23)], ignore_index=True)
    pidx = inference_pool().set_index("row_id"); rng = np.random.default_rng(C["bootstrap_seed"] + 23); out = []
    keep_p = ["latent_cuts", "latent_noanchor", "regime_bias", "same_regime", "pooled_erm", "source_only"]
    for (lab, tgt), g in files[files.protocol.isin(keep_p)].groupby(["label", "target"]):
        if "latent_cuts" not in set(g.protocol):
            continue
        ref_ids = pd.read_parquet(g[g.protocol == "latent_cuts"].iloc[0].file).row_id.to_numpy(); preds = {}
        for _, r in g.iterrows():
            preds.setdefault(r.protocol, []).append(pd.read_parquet(r.file).set_index("row_id").loc[ref_ids][r.col].to_numpy())
        ycol = "y_r6" if lab == "r6" else "y_al"
        cr = compare.cell(preds, pidx.loc[ref_ids, ycol].to_numpy(), pidx.loc[ref_ids, "cluster_id"].to_numpy(), "latent_cuts", 1000, rng)
        cr.insert(0, "target", tgt); cr.insert(0, "label", lab); out.append(cr)
    ct = pd.concat(out, ignore_index=True); ct["p_holm"] = np.nan
    fam = ct.protocol.isin(["latent_noanchor", "regime_bias", "same_regime"]) & ct.p_wald.notna()
    for mt, g in ct[fam].groupby("metric"):
        ct.loc[g.index, "p_holm"] = compare.holm(g.p_wald.to_numpy())
    _write("E23", "ablation_contrasts_vs_latent_cuts.csv", ct)
    _manifest("E23", t0, {"seeds": C["seeds"][:3], "splits": "identical to E10/E22", "reference": "latent_cuts (E22)", "clusters": "publisher (inference_pool)",
                          "multiplicity": "Holm within metric over the three ablation protocols vs latent_cuts; pooled_erm/source_only rows are descriptive"})


def E24():
    """Reviewer-requested robustness analyses (scope.e24); no model is retrained."""
    from . import e24, e19, decompose
    t0 = time.time(); a = anchors(); g = construct.linking_function(a); rng = np.random.default_rng(C["bootstrap_seed"] + 24)
    # (M-06) binarisations of the rubric against the source label on anchors
    b, cf = e24.binarisations(a, rng)
    _write("E24", "binarisations_on_anchors.csv", b); _write("E24", "binarisations_crossfit.json", cf)
    df = add_labels(preds(), g); pi_s = pi_source()
    best_alt = b[~b.cut.str.startswith("level>=")].sort_values("kappa").cut.iloc[-1]
    rows = []
    for (env, m, s), e in df[df.regime == "B"].groupby(["env", "model", "seed"]):
        cc = e24.candidate_cuts(e, level="level")
        for k in dict.fromkeys(["level>=1", "level>=2", "trigger>=1", best_alt]):
            if cc[k].min() < cc[k].max():
                rows.append({"env": env, "model": m, "seed": s, "cut": k, "auroc": M.auroc(cc[k], e.p_calibrated.to_numpy()),
                             "nll": M.nll(cc[k], e.p_calibrated.to_numpy()), "prevalence": float(cc[k].mean())})
    _write("E24", "frozen_auroc_by_binarisation.csv", pd.DataFrame(rows).groupby(["env", "model", "cut"]).mean(numeric_only=True).reset_index().drop(columns="seed"))
    # (M-02) component-based link transport -> implied delta per platform
    lvf = data.banglaclick_levels(); pool = inference_pool(); aset = set(a.normalized_title_sha256)
    cells = {}
    for env in ("banglaclick:news", "banglaclick:facebook", "banglaclick:youtube", "banglaclick:twitter"):
        ids = pool[pool.env == env]
        cells[env] = lvf[lvf.row_id.isin(ids.row_id)].reset_index(drop=True)
        if env == "banglaclick:news":
            na = ids[~ids.normalized_title_sha256.isin(aset)].row_id
            cells["banglaclick:news_nonanchor"] = lvf[lvf.row_id.isin(na)].reset_index(drop=True)
    _write("E24", "component_link_transport.csv", e24.component_link(a, cells, rng, boot=500))
    _write("E24", "component_mix_by_level.csv", e24.component_mix({**cells, "anchors": a.rename(columns={"bc_level": "level"})}))
    # (M-10) BanglaClick YouTube without its dominant channel
    src = df[df.env == "banglabait:news"]; yt = df[df.env == "banglaclick:youtube"]
    top = yt.drop_duplicates("row_id").cluster_id.value_counts().index[0]; rows = []
    for (m, s), e in yt.groupby(["model", "seed"]):
        sm = src[(src.model == m) & (src.seed == s)]; rs, ra = decompose.source_risk(sm.p_calibrated.to_numpy(), sm.y_r6.to_numpy())
        for sub, ee in (("all", e), ("without_top_channel", e[e.cluster_id != top])):
            c = decompose.components(ee.p_calibrated.to_numpy(), ee.y_r6.to_numpy(), ee.y_al.to_numpy(), pi_s, rs, ra)
            rows.append({"subset": sub, "model": m, "seed": s, "n": len(ee), "clusters": ee.cluster_id.nunique(),
                         "auroc_r6": c["auroc_R00"], "auroc_aligned": c["auroc_RA"], **{k: c[k] for k in ("nll_gap", "nll_construct", "nll_prior", "nll_residual", "prev_r6", "prev_aligned")}})
    _write("E24", "youtube_without_top_channel.csv", pd.DataFrame(rows).groupby(["subset", "model"]).mean(numeric_only=True).reset_index().drop(columns="seed"))
    # (M-14) metric suite of the frozen models
    rows = []
    for (env, m, s), e in df.groupby(["env", "model", "seed"]):
        for lab in (("y_r6", "y_al") if e.regime.iloc[0] == "B" else ("y_r6",)):
            y = e[lab].to_numpy(); r = M.all_metrics(y, e.p_calibrated.to_numpy(), hard=y)
            rows.append({"env": env, "model": m, "seed": s, "label": lab, **r})
    _write("E24", "frozen_metric_suite.csv", pd.DataFrame(rows).groupby(["env", "model", "label"]).mean(numeric_only=True).reset_index().drop(columns="seed"))
    # (M-03) score-conditional link on non-anchor news items only
    sc = pd.read_parquet(paths.private_root() / "predictions" / "E19" / "blind_tfidf_scores.parquet")
    st = data.load_predictions(); src_y = st[st.env == "banglabait:news"].drop_duplicates("row_id").set_index("row_id").label.astype(float)
    pa = pool[pool.env == "banglabait:news"].set_index("row_id").y_r6.astype(float); src_y = pd.concat([src_y, pa[~pa.index.isin(src_y.index)]])
    lv = lvf.set_index("row_id").level.astype(int); outp, outb = [], []
    for name, s in {"tfidf_lr": sc, **{k: v for k, v in _xcell_scorers().items() if "|" not in k}}.items():
        if not s.env.eq("banglabait:news").any():
            continue
        pt, bt = e19.application(s, pool, a, lv, src_y, rng, 300, cells_only=["banglaclick:news"],
                                 cell_filter=lambda e_, c_: c_[c_.ya_obs.isna()])
        pt["scorer"], bt["scorer"] = name, name; outp.append(pt); outb.append(bt)
    pt, bt = pd.concat(outp), pd.concat(outb)
    q = lambda v, a_: float(np.nanpercentile(v, a_))
    summ = pt.groupby(["scorer", "env", "label"]).mean(numeric_only=True).reset_index().drop(columns="seed")
    ci = bt.groupby(["scorer", "env", "label"]).agg(**{f"{m}_{k}": (m, (lambda v, a_=a_: q(v, a_))) for m in ("auroc", "nll", "nll_residual_after_prior")
                                                     for k, a_ in (("lo", 2.5), ("hi", 97.5))}).reset_index()
    _write("E24", "scl_news_nonanchor.csv", summ.merge(ci, on=["scorer", "env", "label"]))
    # (M-08) publisher-cluster jackknife t-tests for the protocol contrasts
    from . import compare
    files = _gather_predictions(); pidx = pool.set_index("row_id"); out = []
    for (model, lab, tgt), gg in files[~files.protocol.isin(["majority"])].groupby(["model", "label", "target"]):
        if "source_only" not in set(gg.protocol):
            continue
        ref_ids = pd.read_parquet(gg.iloc[0].file).row_id.to_numpy(); P = {}
        for _, r in gg.iterrows():
            P.setdefault(r.protocol, []).append(pd.read_parquet(r.file).set_index("row_id").loc[ref_ids][r.col].to_numpy())
        P = {k: np.mean(v, axis=0) for k, v in P.items()}
        y = pidx.loc[ref_ids, "y_r6" if lab == "r6" else "y_al"].to_numpy(); cl = pidx.loc[ref_ids, "cluster_id"].to_numpy()
        pairs = [(p, "source_only") for p in P if p not in ("source_only", "in_domain", "source_oracle")]
        if {"latent_cuts", "pooled_erm"} <= set(P):
            pairs.append(("latent_cuts", "pooled_erm"))
        for pr, ref in pairs:
            for mt in ("auroc", "nll"):
                d0, se, p, G = e24.jackknife_contrast(y, cl, P[ref], P[pr], mt)
                out.append({"model": model, "label": lab, "target": tgt, "protocol": pr, "reference": ref, "metric": mt,
                            "diff": d0, "se_jack": se, "p_jack_t": p, "clusters": G})
    jk = pd.DataFrame(out); jk["p_holm_jack"] = np.nan
    fam = jk.p_jack_t.notna() & ~jk.protocol.isin(["source_mlls"])
    for mt, gg in jk[fam].groupby("metric"):
        jk.loc[gg.index, "p_holm_jack"] = compare.holm(gg.p_jack_t.to_numpy())
    _write("E24", "protocol_contrasts_jackknife.csv", jk)
    _manifest("E24", t0, {"clusters": "publisher (inference_pool)", "binarisation_crossfit": "5 publisher folds x 20 repeats",
                          "component_link_bootstrap": "500 draws over anchor publishers", "scl_nonanchor_bootstrap": 300,
                          "jackknife": "leave-one-cluster-out, t with G-1 df, Holm within metric (prior correction excluded)",
                          "best_alternative_cut": best_alt})


def E21S():
    """E21 with the working link deliberately mis-transported by delta on the logit scale (design validity, M-16)."""
    from . import e21, e19
    t0 = time.time(); pool = inference_pool(); anc = anchors(); rng = np.random.default_rng(C["bootstrap_seed"] + 211)
    lv = data.banglaclick_levels().set_index("row_id").level.astype(int)
    sc = pd.read_parquet(paths.private_root() / "predictions" / "E19" / "blind_tfidf_scores.parquet")
    sims = []
    for sd in sorted(sc.seed.unique()):
        d = e19.anchor_items(sc[sc.seed == sd], pool, anc, lv)
        for sh in (-1.0, -0.5, 0.5, 1.0):
            r = e21.simulate(d, [50, 100, 200], 200, rng, h_shift=sh); r["seed"], r["h_shift"] = sd, sh; sims.append(r)
    sim = pd.concat(sims, ignore_index=True); sim[["auroc_cover", "nll_cover"]] = sim[["auroc_cover", "nll_cover"]].astype(float)
    rmse = lambda v: float(np.sqrt(np.nanmean(np.square(v))))
    _write("E21S", "design_misspecified_link.csv", sim.groupby(["h_shift", "n", "design", "estimator"]).agg(
        auroc_bias=("auroc_err", "mean"), auroc_rmse=("auroc_err", rmse), auroc_cover=("auroc_cover", "mean"),
        nll_bias=("nll_err", "mean"), nll_rmse=("nll_err", rmse), nll_cover=("nll_cover", "mean")).reset_index())
    _manifest("E21S", t0, {"variant": "E21S misspecified working link", "shifts": [-1, -0.5, 0.5, 1], "reps": 200, "scorer": "tfidf_lr", "clusters": "publisher"})


EXPS = {"E27": E27, "E26": E26, "E4T": lambda: E4("E4T", "trigger"), "E24": E24, "E21S": E21S, "E23C": E23C, "E23": E23, "E22": E22, "E21": E21,"E20": E20,"E19": E19,"E0": E0, "E1": E1, "E2": E2, "E3": E3, "E4": E4, "E5": E5, "E7": E7, "E10": E10, "E10T": E10T, "E11": E11, "E13": E13, "E14": E14, "E15": E15, "E16": E16, "E17": E17, "E18": E18}

if __name__ == "__main__":
    todo = sys.argv[1:] or ["all"]
    if todo == ["all"]:
        todo = list(EXPS)
    for k in todo:
        t = time.time(); EXPS[k](); print(f"{k} done in {time.time() - t:.0f}s", flush=True)
