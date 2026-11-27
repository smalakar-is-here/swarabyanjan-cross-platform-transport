"""Collect the fitted regime cut-points of the GPU latent-cuts units (w09, w10) from their run logs
into results/scope_v1/E22/cut_points_transformer.csv (aggregate numbers only; no text).

Usage: SCOPE_PRIVATE_DATA=<private_data/scope> python scripts/scope_collect_cuts.py
The transformer model uses three levels (0, 1, 2+), so theta_2 is the cut for level >= 2 and there is no theta_3.
"""
import json, os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = Path(os.environ["SCOPE_PRIVATE_DATA"]) / "kaggle_out/full"
rows = []
for f in sorted(P.glob("*/*latentcuts*/results/runlog_*_latent_cuts.json")):
    for r in json.loads(f.read_text(encoding="utf-8")):
        c = r["cuts"]
        rows.append({"model": r["model"], "target": r["target"], "seed": r["seed"], "tau_A": c["A"],
                     "tau_C": c["C"], "has_C": "baitbuster:youtube" in r["envs"], "theta_1": c["B1"], "theta_2": c["B2"],
                     "temperature": r["temperature"], "git_sha": r["git_sha"]})
d = pd.DataFrame(rows).drop_duplicates(["model", "target", "seed"])
d.loc[~d.has_C, "tau_C"] = float("nan")  # regime C absent from training: its cut is not fitted
out = ROOT / "results/scope_v1/E22/cut_points_transformer.csv"
d.to_csv(out, index=False)
print(d.round(2).to_string())
c = d[d.has_C]
print("theta1<tauC<tauA:", ((c.theta_1 < c.tau_C) & (c.tau_C < c.tau_A)).groupby(c.model).mean().to_dict())
print("tauA-theta2:", (d.tau_A - d.theta_2).groupby(d.model).describe().round(2))
