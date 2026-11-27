"""Keep the local Kaggle login's GPU slots busy with queued work units, within quota.

Usage: python scripts/scope_gpu_queue.py <kaggle_user> <unit> [<unit> ...]
Every 5 minutes: pull outputs of finished SCOPE kernels into $SCOPE_PRIVATE_DATA/kaggle_out/full/<model>/<slug>/,
and if a batch GPU slot is free (Kaggle allows 2) and remaining weekly GPU quota exceeds MIN_QUOTA_H, push the
next queued unit. Appends one JSON line per event to $SCOPE_QUEUE_LOG. Exits when the queue is empty and
nothing it launched is still running. Uses only the CLI's own configured login; reads no credential files.
"""
import json, os, re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIN_QUOTA_H = 2.0
PRIV = Path(os.environ["SCOPE_PRIVATE_DATA"]); LOG = Path(os.environ["SCOPE_QUEUE_LOG"])
sys.path.insert(0, str(ROOT / "scripts"))
from scope_make_worker_packages import UNITS, build  # noqa: E402


def sh(*a):
    r = subprocess.run(list(a), capture_output=True, text=True)
    return "\n".join(l for l in (r.stdout + r.stderr).splitlines() if "outdated" not in l.lower())


def log(**k):
    k["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(k) + "\n")
    print(k, flush=True)


def status(user, slug):
    s = sh("kaggle", "kernels", "status", f"{user}/{slug}")
    m = re.search(r"KernelWorkerStatus\.(\w+)", s)
    return m.group(1) if m else "UNKNOWN"


def quota_left():
    m = re.search(r"GPU\s+[\d.]+h\s+([\d.]+)h", sh("kaggle", "quota"))
    return float(m.group(1)) if m else 0.0


def main(user, queue, watch):
    launched = {}  # slug -> model
    for slug in watch:
        launched[slug] = "xlmr" if "xlmr" in slug else "banglabert"
    while True:
        running = 0
        for slug, model in list(launched.items()):
            st = status(user, slug)
            if st in ("RUNNING", "QUEUED"):
                running += 1
            elif st in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCELLED"):
                dest = PRIV / "kaggle_out" / "full" / model / slug
                dest.mkdir(parents=True, exist_ok=True)
                sh("kaggle", "kernels", "output", f"{user}/{slug}", "-p", str(dest))
                n = len(list(dest.rglob("pred_*.parquet")))
                log(event="finished", kernel=slug, status=st, predictions=n, account="Account01")
                del launched[slug]
        while queue and running < 2:
            q = quota_left()
            if q < MIN_QUOTA_H:
                log(event="stop_quota", gpu_hours_left=q, remaining_units=queue); queue = []
                break
            u = queue.pop(0); d = build(u, owner=user, dataset_owner=user, gpu=True)
            out = sh("kaggle", "kernels", "push", "-p", str(d))
            if "successfully pushed" in out:
                launched[d.name] = UNITS[u]["model"]; running += 1
                log(event="pushed", unit=u, kernel=d.name, gpu_hours_left=q, account="Account01")
            else:
                log(event="push_failed", unit=u, reason=out[-200:]); queue.insert(0, u); break
        if not queue and not launched:
            log(event="queue_done"); return
        time.sleep(300)


if __name__ == "__main__":
    user = sys.argv[1]; args = sys.argv[2:]
    watch = [a[len("watch="):] for a in args if a.startswith("watch=")]
    main(user, [a for a in args if not a.startswith("watch=")], watch)
