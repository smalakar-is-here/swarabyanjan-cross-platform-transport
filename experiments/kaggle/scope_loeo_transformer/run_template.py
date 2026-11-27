"""SCOPE E10 transformer LOEO fine-tuning (Kaggle GPU kernel).

Placeholders filled at push time (scripts/scope_kaggle_push.py, scripts/scope_make_worker_packages.py):
__MODEL__, __MODE__, __GIT_SHA__, __PROTOCOL__ (pooled_erm | source_only | group_balanced | group_dro),
__LABELS__, __TARGETS__, __SEEDS__ (JSON lists; empty = all).
group_dro is online Group DRO over environments (Sagawa et al., 2020): per batch, group weights
q_g <- q_g exp(eta * loss_g), normalised; the loss is sum_g q_g loss_g. eta = 0.01.
For each regime-B label definition (r6 = 1[L>=1], aligned = 1[L>=2]), held-out target cell and
seed: train pooled ERM on every other cell (source + remaining targets), with training items whose
normalised title occurs in the held-out cell removed; fit a scalar temperature on a cluster-grouped
10% validation split of the training pool; write per-item calibrated probabilities for the held-out
cell. Outputs contain row ids and probabilities only (no text). Hyperparameters = R6 lock.
"""
import json, os, random, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

MODEL, MODE, GIT_SHA = "__MODEL__", "__MODE__", "__GIT_SHA__"
PROTOCOL = "__PROTOCOL__" if not "__PROTOCOL__".startswith("__") else "pooled_erm"
_sel = lambda s, d: (json.loads(s) or d) if not s.startswith("__") else d  # placeholders sit in single quotes
DRO_ETA = 0.01
REPO = {"banglabert": ("csebuetnlp/banglabert", "9ce791f330578f50da6bc52b54205166fb5d1c8c", 2),
        "xlmr": ("FacebookAI/xlm-roberta-base", "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089", 3)}[MODEL]
BS, LR, WD, WARM, MAXLEN = 16, 2e-5, 0.01, 0.1, 128
TARGETS = ["banglaclick:news", "banglaclick:facebook", "banglaclick:youtube", "banglaclick:twitter", "baitbuster:youtube"]
SEEDS = _sel('__SEEDS__', [42, 137, 2718])
TARGETS = _sel('__TARGETS__', TARGETS)
LABELS = _sel('__LABELS__', ["r6", "aligned"])
LABELS_LATENT = ["r6", "aligned"]  # latent_cuts: one model per (target, seed) predicts under both label definitions
OUT = Path("/kaggle/working/results"); OUT.mkdir(parents=True, exist_ok=True)
def _find_pool(root=Path("/kaggle/input")):
    """Locate the training pool. Exact name first; else the single *.parquet whose name contains
    'training_pool' (dataset file renamed on upload). An empty /kaggle/input means the kernel's account
    cannot read the private dataset (Kaggle accepts the push but mounts nothing): fail fast, listing what
    is mounted, before any model download or GPU time."""
    hits = sorted(root.rglob("scope_training_pool_v1.parquet"))
    if not hits:
        hits = sorted(p for p in root.rglob("*.parquet") if "training_pool" in p.name.replace("-", "_"))
    if len(hits) != 1:
        mounted = sorted(str(p) for p in root.rglob("*"))[:50]
        raise SystemExit(f"POOL_NOT_FOUND: {len(hits)} candidate(s) {hits}; /kaggle/input contains {mounted or 'NOTHING'}. "
                         "Empty => this account has no read access to the dataset in kernel-metadata dataset_sources.")
    print("pool", hits[0], flush=True)
    return hits[0]


pool = pd.read_parquet(_find_pool())
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "", flush=True)
tok = AutoTokenizer.from_pretrained(REPO[0], revision=REPO[1])


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def batches(texts, labels=None, shuffle=False, rng=None):
    idx = np.arange(len(texts))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), BS if labels is not None else 64):
        j = idx[i:i + (BS if labels is not None else 64)]
        enc = tok([texts[k] for k in j], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
        enc = {k: v.to(dev) for k, v in enc.items()}
        yield enc, (torch.tensor(labels[j], dtype=torch.long, device=dev) if labels is not None else None), j


# ---- training objective (kept self-contained between these markers; unit-tested in tests/test_scope.py)
# OBJECTIVE-BEGIN
def objective(lv, gb, q, n_groups, protocol=None, eta=None):
    """lv: per-item losses of the batch; gb: environment index of each item; q: current group weights.
    pooled_erm / source_only: mean loss. group_balanced: each environment's batch-mean loss weighted 1/n_groups
    over the environments present (static reweighting). group_dro: online Group DRO (Sagawa et al., 2020),
    q_g <- q_g exp(eta * l_g) on detached losses, renormalised; loss = sum_g q_g l_g; an environment absent
    from the batch has l_g = 0 (reference-implementation behaviour)."""
    protocol = protocol or PROTOCOL; eta = DRO_ETA if eta is None else eta
    if protocol in ("pooled_erm", "source_only"):
        return lv.mean(), q
    gl = torch.zeros(n_groups, dtype=lv.dtype, device=lv.device).index_add(0, gb, lv)
    cnt = torch.zeros(n_groups, dtype=lv.dtype, device=lv.device).index_add(0, gb, torch.ones_like(lv))
    gl = gl / cnt.clamp(min=1)
    if protocol == "group_balanced":
        present = (cnt > 0).to(lv.dtype)
        return (gl * present).sum() / present.sum(), q
    q = q * torch.exp(eta * gl.detach()); q = q / q.sum()
    return (q * gl).sum(), q


def latent_loss(f, reg, yk, cuts):
    """Per-item negative log-likelihood of the regime cut-point model (scope.latent, E22).
    f: latent intensity (logit difference); reg: 0 BanglaBait, 1 BanglaClick, 2 BaitBuster;
    yk: binary label (reg 0/2) or level in {0, 1, 2} = y_r6 + y_al (reg 1; level 3 is merged into 2 because the
    training pool carries only the two binarisations); cuts: tensor [tau_A, tau_C, theta_1, log_gap] with
    theta_2 = theta_1 + exp(log_gap) > theta_1."""
    tA, tC, t1, lg = cuts[0], cuts[1], cuts[2], cuts[3]; t2 = t1 + torch.exp(lg)
    ls = torch.nn.functional.logsigmoid
    tau = torch.where(reg == 0, tA, tC)
    bin_nll = -(yk * ls(f - tau) + (1 - yk) * ls(-(f - tau)))
    a = torch.sigmoid(f - t1); b = torch.sigmoid(f - t2)               # P(L>=1), P(L>=2)
    p_lvl = torch.stack([1 - a, a - b, b], 1).clamp(min=1e-12)
    ord_nll = -torch.log(p_lvl.gather(1, yk.clamp(0, 2).long().unsqueeze(1)).squeeze(1))
    return torch.where(reg == 1, ord_nll, bin_nll)
# OBJECTIVE-END


@torch.no_grad()
def logits(model, texts):
    model.eval(); out = []
    for enc, _, _ in batches(texts):
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            l = model(**enc).logits.float()
        out.append((l[:, 1] - l[:, 0]).cpu().numpy())
    return np.concatenate(out)


def fit_T(z, y):
    best, bT = 1e9, 1.0
    for T in np.exp(np.linspace(np.log(0.05), np.log(20), 200)):
        p = np.clip(1 / (1 + np.exp(-z / T)), 1e-12, 1 - 1e-12)
        v = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if v < best:
            best, bT = v, T
    return bT


log = []
for lab in LABELS:
    ycol = "y_r6" if lab == "r6" else "y_al"
    for tgt in (TARGETS if MODE == "full" else TARGETS[:1]):
        for seed in (SEEDS if MODE == "full" else SEEDS[:1]):
            suffix = "" if PROTOCOL == "pooled_erm" else f"_{PROTOCOL}"
            fn = OUT / f"pred_{MODEL}_{lab}_{tgt.replace(':', '-')}_s{seed}{suffix}.parquet"
            if fn.exists():
                continue
            t0 = time.time(); seed_all(seed)
            test = pool[pool.env == tgt]
            base = pool[pool.env == "banglabait:news"] if PROTOCOL == "source_only" else pool[pool.env != tgt]
            train = base[~base.normalized_title_sha256.isin(test.normalized_title_sha256)]
            fit_i, val_i = next(GroupShuffleSplit(1, test_size=0.1, random_state=seed).split(train, groups=train.cluster_id))
            tr, va = train.iloc[fit_i], train.iloc[val_i]
            if MODE != "full":
                tr = tr.sample(400, random_state=seed); va = va.sample(200, random_state=seed); test = test.sample(300, random_state=seed)
            model = AutoModelForSequenceClassification.from_pretrained(REPO[0], revision=REPO[1], num_labels=2).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
            steps = REPO[2] * int(np.ceil(len(tr) / BS))
            sch = get_linear_schedule_with_warmup(opt, int(WARM * steps), steps)
            scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
            rng = np.random.default_rng(seed); X = tr.title.tolist(); Y = tr[ycol].to_numpy().astype(int)
            envs = sorted(tr.env.unique()); G = torch.tensor(tr.env.map({e: i for i, e in enumerate(envs)}).to_numpy(), device=dev)
            q = torch.ones(len(envs), device=dev) / len(envs)
            LAT = PROTOCOL == "latent_cuts"
            if LAT:
                regime = lambda d: d.env.str.split(":").str[0].map({"banglabait": 0, "banglaclick": 1, "baitbuster": 2}).to_numpy()
                level = lambda d: np.where(regime(d) == 1, d.y_r6.to_numpy() + d.y_al.to_numpy(), d.y_r6.to_numpy())
                R = torch.tensor(regime(tr), device=dev); K = torch.tensor(level(tr), dtype=torch.float32, device=dev)
                cuts = torch.nn.Parameter(torch.tensor([0.0, 0.0, -1.0, 0.0], device=dev))
                copt = torch.optim.Adam([cuts], lr=1e-2); has_C = bool((regime(tr) == 2).any())
            for ep in range(REPO[2]):
                model.train()
                for enc, yb, j in batches(X, Y, shuffle=True, rng=rng):
                    gb = G[torch.as_tensor(j, device=dev)]  # environment of each item in this batch
                    with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                        lo = model(**enc).logits.float()
                    if LAT:
                        jj = torch.as_tensor(j, device=dev)
                        loss = latent_loss(lo[:, 1] - lo[:, 0], R[jj], K[jj], cuts).mean(); copt.zero_grad()
                    else:
                        lv = torch.nn.functional.cross_entropy(lo, yb, reduction="none")
                        loss, q = objective(lv, gb, q, len(envs))
                    opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
                    if LAT:  # cuts are not in opt, so unscale_ did not touch them; skip the step on overflow like scaler.step
                        cuts.grad.div_(scaler.get_scale())
                        if torch.isfinite(cuts.grad).all():
                            copt.step()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step()
            if LAT:
                c = cuts.detach().cpu().numpy(); th = {"A": c[0], "C": c[1] if has_C else c[0], "B1": c[2], "B2": c[2] + np.exp(c[3])}
                fv, zt = logits(model, va.title.tolist()), logits(model, test.title.tolist()); rv = regime(va); kv = level(va)
                treg = int(regime(test)[0])
                for lab2 in LABELS_LATENT:
                    tb = {"r6": 1, "aligned": 2}[lab2]; con = {0: "A", 2: "C"}.get(treg, f"B{tb}")
                    zv = fv - np.where(rv == 0, th["A"], np.where(rv == 2, th["C"], th[f"B{tb}"]))
                    T = fit_T(zv, np.where(rv == 1, (kv >= tb).astype(float), kv))
                    z = zt - th[con]
                    pd.DataFrame({"row_id": test.row_id.to_numpy(), "logit": z, "temperature": T, "p_calibrated": 1 / (1 + np.exp(-z / T))}).to_parquet(
                        OUT / f"pred_{MODEL}_{lab2}_{tgt.replace(':', '-')}_s{seed}_latent_cuts.parquet", index=False)
            else:
                T = fit_T(logits(model, va.title.tolist()), va[ycol].to_numpy())
                z = logits(model, test.title.tolist())
                pd.DataFrame({"row_id": test.row_id.to_numpy(), "logit": z, "temperature": T,
                              "p_calibrated": 1 / (1 + np.exp(-z / T))}).to_parquet(fn, index=False)
            if PROTOCOL == "source_only" and tgt == "banglaclick:news" and MODE == "full":
                # Anchor-blind scorer for E19: this model never saw a BanglaClick-news title (hence no anchor).
                # Score every other non-source cell with the same model and temperature. Separate file prefix
                # (xcell_) so LOEO scoring, which globs pred_*, is unaffected.
                oth = pool[(pool.env != "banglabait:news") & (pool.env != tgt)]
                zo = logits(model, oth.title.tolist())
                pd.DataFrame({"row_id": oth.row_id.to_numpy(), "env": oth.env.to_numpy(), "logit": zo, "temperature": T,
                              "p_calibrated": 1 / (1 + np.exp(-zo / T))}).to_parquet(
                    OUT / f"xcell_{MODEL}_{lab}_heldout-{tgt.replace(':', '-')}_s{seed}_{PROTOCOL}.parquet", index=False)
            rec = {"model": MODEL, "protocol": PROTOCOL, "envs": envs, "dro_q": q.tolist() if PROTOCOL == "group_dro" else None,
                   "cuts": {k: float(v) for k, v in th.items()} if LAT else None, "label": lab, "target": tgt, "seed": seed, "n_train": len(tr), "n_val": len(va),
                   "n_test": len(test), "temperature": float(T), "seconds": round(time.time() - t0, 1),
                   "repo": REPO[0], "revision": REPO[1], "epochs": REPO[2], "git_sha": GIT_SHA, "mode": MODE}
            log.append(rec); print(json.dumps(rec), flush=True)
            del model, opt; torch.cuda.empty_cache()
            (OUT / f"runlog_{MODEL}{'' if PROTOCOL == 'pooled_erm' else '_' + PROTOCOL}.json").write_text(json.dumps(log, indent=1))
print("DONE", flush=True)
