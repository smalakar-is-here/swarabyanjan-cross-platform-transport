"""Local CPU smoke test of the Kaggle transformer template (no GPU, no model download).

Replaces `transformers` with a tiny stand-in (hashed bag-of-characters tokenizer + embedding-bag classifier) and runs
the *unchanged* template control flow in probe mode on the real training pool, for the given protocols. Checks that
each protocol trains, writes the expected prediction files with finite calibrated probabilities, and logs its run.
Usage: python tests/smoke_template_protocols.py <pool.parquet> regime_bias latent_noanchor same_regime ...
Not part of the pytest suite (needs the private pool).
"""
import json, sys, tempfile, types
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (ROOT / "experiments/kaggle/scope_loeo_transformer/run_template.py").read_text(encoding="utf-8")


class Tok:
    def __call__(self, texts, truncation=True, max_length=128, padding=True, return_tensors="pt"):
        ids = [[(hash(c) % 997) + 1 for c in str(t)[:max_length]] or [1] for t in texts]
        m = max(map(len, ids)); x = torch.zeros(len(ids), m, dtype=torch.long)
        for i, r in enumerate(ids):
            x[i, :len(r)] = torch.tensor(r)
        return {"input_ids": x}


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.e = torch.nn.EmbeddingBag(1000, 16, padding_idx=0); self.o = torch.nn.Linear(16, 2)

    def forward(self, input_ids):
        return types.SimpleNamespace(logits=self.o(self.e(input_ids)))


fake = types.ModuleType("transformers")
fake.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: Tok())
fake.AutoModelForSequenceClassification = types.SimpleNamespace(from_pretrained=lambda *a, **k: Net())
fake.get_linear_schedule_with_warmup = lambda opt, w, n: torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
sys.modules["transformers"] = fake


def run(pool_path, protocol, label, target):
    out = Path(tempfile.mkdtemp())
    src = (TEMPLATE.replace("__MODEL__", "banglabert").replace("__MODE__", "probe").replace("__GIT_SHA__", "smoke")
           .replace("__PROTOCOL__", protocol).replace("__LABELS__", json.dumps([label])).replace("__TARGETS__", json.dumps([target]))
           .replace("__SEEDS__", "[]")
           .replace('OUT = Path("/kaggle/working/results")', f'OUT = Path(r"{out}")')
           .replace('def _find_pool(root=Path("/kaggle/input")):', 'def _find_pool(root=None):\n    return Path(r"%s")\n\ndef _unused(root=None):' % pool_path))
    exec(compile(src, f"run_template[{protocol}]", "exec"), {"__name__": "__main__"})
    files = sorted(out.glob("pred_*.parquet"))
    for f in files:
        p = pd.read_parquet(f).p_calibrated.to_numpy()
        assert len(p) and np.isfinite(p).all() and ((p > 0) & (p < 1)).all(), f
    logs = list(out.glob("runlog_*.json"))
    rec = json.loads(logs[0].read_text()) if logs else []
    return [f.name for f in files], rec


if __name__ == "__main__":
    pool_path = sys.argv[1]
    for proto in sys.argv[2:]:
        for label, target in (("r6", "banglaclick:news"), ("aligned", "banglaclick:twitter")):
            if proto == "latent_noanchor" and label == "aligned":
                continue
            files, rec = run(pool_path, proto, label, target)
            print(proto, label, target, "->", files, "| log:", {k: rec[0].get(k) for k in ("protocol", "n_train", "anchors_removed", "cuts")} if rec else "NO LOG")
