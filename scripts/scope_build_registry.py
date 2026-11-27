"""Build the single experiment registry (docs/scope/EXPERIMENT_REGISTRY.csv) and the GPU execution
manifest (docs/scope/GPU_EXECUTION_MANIFEST.json).

Inputs: results/scope_v1/*/MANIFEST.json (local CPU runs) and, when $SCOPE_PRIVATE_DATA is set, the
Kaggle runlogs in $SCOPE_PRIVATE_DATA/kaggle_out/<mode>/<model>/**/runlog_*.json.
Public files carry the account alias only (Account01); no username, token or private path.
Kernel timing comes from the kernel records passed in KERNELS below, filled from `kaggle kernels status`
and the kernel logs.
"""
import glob, json, os, sys
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results/scope_v1"
DOCS = REPO / "docs/scope"
EXPERIMENTS = {
    "E0": ("integrity", "SHA-256 check of restricted and raw inputs; R6 H1 reproduction"),
    "E1": ("construct", "anchor linking between annotation regimes"),
    "E2": ("transport", "cue log-OR by cell; crossed platform x regime contrasts"),
    "E2_cues_v11": ("transport", "E2 with corrected cue extractor"),
    "E3": ("prior", "unlabelled prevalence estimation (CC, PCC, BBSE, MLLS)"),
    "E4": ("decomposition", "Shapley split of the target risk gap"),
    "E5": ("calibration", "temperature, prior correction, label definitions"),
    "E7": ("cue extractor", "v1.0 vs v1.1 cue presence"),
    "E10": ("training", "LOEO training, TF-IDF family (+ transformers in E10T)"),
    "E10_transformer": ("training", "LOEO pooled ERM, BanglaBERT and XLM-R on Kaggle GPU"),
    "E11": ("audit", "budgeted target audit (25/50/100/200 labels)"),
    "E13": ("simulation", "few-cluster size of cluster bootstrap, CR1, WCR"),
    "E14": ("diagnostic", "between-cluster share of label variance"),
    "E15": ("simulation", "synthetic recovery of source-construct risk by label proxies"),
    "E16": ("benchmark", "shift-correction baselines: MLLS prior shift, importance weighting"),
    "E17": ("benchmark", "paired cluster-bootstrap contrasts between LOEO protocols, Holm-adjusted"),
    "E18": ("validation", "bias of target-side label proxies against the source label on anchors"),
}


def main(kernels_json=None):
    rows = []
    for m in sorted(RES.glob("*/MANIFEST.json")):
        d = json.loads(m.read_text(encoding="utf-8")); e = d.get("experiment", m.parent.name)
        kind, desc = EXPERIMENTS.get(e, ("", ""))
        rows.append({"experiment_id": e, "kind": kind, "description": desc, "compute": "local CPU (8 cores)",
                     "account_alias": "", "kernel_id": "", "status": "COMPLETE",
                     "wall_hours": round(d.get("wall_seconds", float("nan")) / 3600, 3),
                     "config_sha256": d.get("config_sha256", ""), "output_location": f"results/scope_v1/{m.parent.name}/",
                     "evaluation_type": "synthetic" if d.get("simulation") or "synthetic" in str(d.get("evaluation_type", "")) else "real data"})
    gpu = {"accounts": [], "kernels": []}
    if kernels_json:
        K = json.loads(Path(kernels_json).read_text(encoding="utf-8"))
        gpu["accounts"] = [{"alias": "Account01", "note": "the single Kaggle account configured on the execution machine; "
                            "no other account was used (see EXECUTION_LOG.md, section 'Compute')"}]
        priv = os.environ.get("SCOPE_PRIVATE_DATA")
        for k in K:
            runs = []
            if priv:
                # pooled runs sit in <model>/results/, worker units in <model>/<slug>/; count only this kernel's runlogs
                sub = k.get("subdir", "results")
                for f in glob.glob(os.path.join(priv, "kaggle_out", k["mode"], k["model"], sub, "**", "runlog_*.json"), recursive=True):
                    runs += json.loads(Path(f).read_text(encoding="utf-8"))
            k = dict(k); k["runs_logged"] = len(runs)
            k["train_seconds_sum"] = round(sum(r.get("seconds", 0) for r in runs), 1)
            k["git_sha"] = sorted({r.get("git_sha", "") for r in runs})
            gpu["kernels"].append(k)
            rows.append({"experiment_id": k["experiment_id"], "kind": "training", "description": k["description"],
                         "compute": f"Kaggle {k['gpu']}", "account_alias": "Account01", "kernel_id": k["slug"],
                         "status": k["status"], "wall_hours": k.get("wall_hours", ""), "config_sha256": "",
                         "output_location": "private: $SCOPE_PRIVATE_DATA/kaggle_out/" + k["mode"] + "/" + k["model"] + "/ (scored into results/scope_v1/E10/)",
                         "evaluation_type": "real data"})
        (DOCS / "GPU_EXECUTION_MANIFEST.json").write_text(json.dumps(gpu, indent=1), encoding="utf-8")
    pd.DataFrame(rows).to_csv(DOCS / "EXPERIMENT_REGISTRY.csv", index=False)
    print(len(rows), "registry rows")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
