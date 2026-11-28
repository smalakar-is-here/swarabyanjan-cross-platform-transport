"""Split the outstanding transformer GPU work into independent work units and build one Kaggle kernel
folder per unit.

A unit is (model, protocol, label); each covers 5 held-out targets x 3 seeds (15 fine-tunings).
Each folder is self-contained: run.py + kernel-metadata.json. A collaborator launches a unit from
their *own* Kaggle account (their own `kaggle` login on their own machine):

    kaggle kernels push -p <unit folder>

after replacing OWNER in kernel-metadata.json with their username and DATASET_OWNER with the
account that shares the private dataset `scope-training-pool-v1` with them. No credentials are
read, stored or passed by this script.

Usage:
  python scripts/scope_make_worker_packages.py build                 # all units -> experiments/kaggle/build/workers/
  python scripts/scope_make_worker_packages.py probe <unit> <user> [protocol]   # CPU probe (tiny subset, no GPU quota) on the local login
  python scripts/scope_make_worker_packages.py push  <unit> <user>   # full GPU run of one unit on the local login
"""
import json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "experiments/kaggle/scope_loeo_transformer/run_template.py"
BUILD = ROOT / "experiments/kaggle/build/workers"
UNITS = {f"w{i + 1:02d}": dict(model=m, protocol=p, label=l) for i, (m, p, l) in enumerate(
    [(m, p, l) for p in ("group_dro", "source_only") for m in ("banglabert", "xlmr") for l in ("r6", "aligned")])}
# E22 regime cut-point model; one model per (target, seed) writes predictions under both label definitions
UNITS.update({"w09": dict(model="banglabert", protocol="latent_cuts", label="r6"),
              "w10": dict(model="xlmr", protocol="latent_cuts", label="r6")})
# E20 source reference: rerun of the news fold of w05/w07, adding source held-out scores (srcval_*). Deterministic
# fine-tuning (C33) means its xcell_* files must equal those of w05/w07, which checks the rerun.
UNITS.update({"w11": dict(model="banglabert", protocol="source_only", label="r6", targets=["banglaclick:news"]),
              "w12": dict(model="xlmr", protocol="source_only", label="r6", targets=["banglaclick:news"])})
# E28 transformer ablations of the regime cut-point model (as E23 for TF-IDF): seeds 42/137/2718, all targets.
# latent_noanchor writes both label definitions from one model (like latent_cuts); the others are one unit per label.
_E28 = [("latent_noanchor", "r6")] + [(p, l) for p in ("regime_bias", "same_regime") for l in ("r6", "aligned")]
UNITS.update({f"w{13 + i:02d}": dict(model=m, protocol=p, label=l) for i, (m, (p, l)) in enumerate(
    [(m, pl) for m in ("banglabert", "xlmr") for pl in _E28])})
# E29 two additional seeds (31415, 65537) for the main transformer protocols, so transformer inference uses five seeds.
_E29 = [(p, l) for p in ("pooled_erm", "group_dro", "source_only") for l in ("r6", "aligned")] + [("latent_cuts", "r6")]
UNITS.update({f"w{23 + i:02d}": dict(model=m, protocol=p, label=l, seeds=[31415, 65537]) for i, (m, (p, l)) in enumerate(
    [(m, pl) for m in ("banglabert", "xlmr") for pl in _E29])})


def build(unit, mode="full", owner="OWNER", dataset_owner="DATASET_OWNER", gpu=True, protocol=None):
    u = dict(UNITS[unit]); u["protocol"] = protocol or u["protocol"]
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or "uncommitted"
    src = TEMPLATE.read_text(encoding="utf-8")
    for k, v in {"__MODEL__": u["model"], "__MODE__": mode, "__GIT_SHA__": sha, "__PROTOCOL__": u["protocol"],
                 "__LABELS__": json.dumps([u["label"]]), "__TARGETS__": json.dumps(u.get("targets", [])), "__SEEDS__": json.dumps(u.get("seeds", []))}.items():
        src = src.replace(k, v)
    slug = (f"scope-e10-{unit}-{u['model']}-{u['protocol'].replace('_', '')}-{u['label']}" + ("-5seed" if u.get("seeds") else "")
            + ("" if mode == "full" else "-probe"))
    d = BUILD / slug; d.mkdir(parents=True, exist_ok=True)
    (d / "run.py").write_text(src, encoding="utf-8")
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{owner}/{slug}", "title": slug, "code_file": "run.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": gpu, "enable_internet": True,
        **({"machine_shape": "NvidiaTeslaP100"} if gpu else {}),
        "dataset_sources": [f"{dataset_owner}/scope-training-pool-v1"], "competition_sources": [], "kernel_sources": []}, indent=1))
    return d


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        for u in UNITS:
            print(u, UNITS[u], build(u).name)
        (BUILD / "UNITS.json").write_text(json.dumps(UNITS, indent=1))
    else:
        unit, user = sys.argv[2:4]; proto = sys.argv[4] if len(sys.argv) > 4 else None  # protocol override, for A/B probes
        # The dataset lives on the data holder's account, not necessarily the pushing account.
        ds_owner = os.environ.get("SCOPE_DATASET_OWNER") or user
        if ds_owner == user and "SCOPE_DATASET_OWNER" not in os.environ:
            print(f"note: SCOPE_DATASET_OWNER unset; assuming {user} owns scope-training-pool-v1", file=sys.stderr)
        d = build(unit, mode="probe" if cmd == "probe" else "full", owner=user, dataset_owner=ds_owner, gpu=(cmd == "push"), protocol=proto)
        r = subprocess.run(["kaggle", "kernels", "push", "-p", str(d)], capture_output=True, text=True)
        print(d.name, r.stdout[-400:], r.stderr[-400:])
