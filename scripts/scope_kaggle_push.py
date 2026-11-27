"""Build and push the E10 transformer kernel for one model.

Usage: python scripts/scope_kaggle_push.py <banglabert|xlmr> <probe|full> <kaggle_user>
Kernel folders are written to experiments/kaggle/build/ (git-ignored); the Kaggle user name is
passed on the command line and never stored in the repository.
"""
import json, subprocess, sys
from pathlib import Path

model, mode, user = sys.argv[1:4]
root = Path(__file__).resolve().parents[1]
sha = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or "uncommitted"
src = (root / "experiments/kaggle/scope_loeo_transformer/run_template.py").read_text(encoding="utf-8")
src = src.replace("__MODEL__", model).replace("__MODE__", mode).replace("__GIT_SHA__", sha)
slug = f"scope-e10-{model}-{mode}"
d = root / "experiments/kaggle/build" / slug; d.mkdir(parents=True, exist_ok=True)
(d / "run.py").write_text(src, encoding="utf-8")
(d / "kernel-metadata.json").write_text(json.dumps({
    "id": f"{user}/{slug}", "title": slug, "code_file": "run.py", "language": "python", "kernel_type": "script",
    "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaP100",
    "dataset_sources": [f"{user}/scope-training-pool-v1"], "competition_sources": [], "kernel_sources": []}, indent=1))
r = subprocess.run(["kaggle", "kernels", "push", "-p", str(d)], capture_output=True, text=True)
print(r.stdout[-600:], r.stderr[-600:])
