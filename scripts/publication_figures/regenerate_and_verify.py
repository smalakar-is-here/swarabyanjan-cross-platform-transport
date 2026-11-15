"""Regenerate the journal-presentation figures from frozen Stage08 records and verify them.

    python scripts/publication_figures/regenerate_and_verify.py [--write-manifest]

Steps (writes only to a temporary directory unless --write-manifest is given):
  1. hash-check every frozen source CSV against results/primary/stage08/figure_source_data/FIGURE_SOURCE_DATA_INDEX.json
  2. run make_fig1_v3.py, make_main_figures.py, make_supp_figures.py into a temporary $SWFIG_OUT
     (built-in exact-value parity checks and text-collision QA run inside the generators)
  3. run check_parity.py (33 main-figure checks) and qa_marker_clearance.py (Figure 1)
  4. native minimum font size of every regenerated PDF >= 7.95 pt (figures are drawn at final print width)
  5. SHA-256 of every regenerated PDF/SVG/PNG equals the committed file under results/publication/...
     and, if present, the entry in PUBLICATION_FIGURE_MANIFEST.json
--write-manifest additionally (re)writes PUBLICATION_FIGURE_MANIFEST.json and FIG01_v3_SOURCE_MANIFEST.json.

Byte identity requires the pinned environment in requirements-publication-figures.txt and the Arial font
(matplotlib falls back to DejaVu Sans otherwise, which changes the bytes but not the plotted data).
Exit code 0 only if every check passes.
"""
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from publication_figure_map import EXTS, F5, FIGURES, PUB_DIR, WORDING_DIFFERENCES  # noqa: E402

h = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
FSD = ROOT / "results/primary/stage08/figure_source_data"
idx = {e["source_csv"]: e["source_csv_sha256"]
       for e in json.loads((FSD / "FIGURE_SOURCE_DATA_INDEX.json").read_text(encoding="utf8"))["source_files"]}
fails = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not ok:
        fails.append(name)


# 1. frozen sources
used = sorted({s for v in FIGURES.values() if isinstance(v[4], list) for s in v[4]})
check(f"{len(used)} frozen Stage08 source CSVs match FIGURE_SOURCE_DATA_INDEX.json",
      all(idx.get(s) == h(FSD / s) for s in used))

# 2-3. regenerate into a temporary directory
out = pathlib.Path(tempfile.mkdtemp(prefix="swfig_"))
env = dict(os.environ, SWFIG_OUT=str(out), PYTHONIOENCODING="utf-8")
for script in ("make_fig1_v3.py", "make_main_figures.py", "make_supp_figures.py", "check_parity.py",
               "qa_marker_clearance.py"):
    r = subprocess.run([sys.executable, "-W", "ignore", script], cwd=HERE, env=env, capture_output=True, text=True,
                       encoding="utf8")
    tail = (r.stdout.strip().splitlines() or [""])[-1]
    qa_issues = [l for l in r.stdout.splitlines() if "[text QA]" in l]
    check(f"{script} exit 0, no text-collision issues", r.returncode == 0 and not qa_issues,
          tail[:90] if r.returncode == 0 else (r.stderr[-300:] or tail))

# 4. native font size
try:
    import pymupdf
    small = {}
    for stem, (gen, *_rest) in FIGURES.items():
        page = pymupdf.open(out / f"{gen}.pdf")[0]
        sizes = [s["size"] for b in page.get_text("dict")["blocks"] for l in b.get("lines", []) for s in l["spans"]
                 if s["text"].strip()]
        if min(sizes) < 7.95:
            small[stem] = round(min(sizes), 2)
    check("all 17 figures: native minimum text size >= 7.95 pt", not small, str(small) if small else "")
except ImportError:
    check("pymupdf available for font-size check", False, "pip install pymupdf (see requirements)")

# 5. hashes
pub = ROOT / PUB_DIR
manifest_path = pub / "PUBLICATION_FIGURE_MANIFEST.json"
recorded = {}
if manifest_path.is_file():
    recorded = {f["path"]: f["sha256"] for f in json.loads(manifest_path.read_text(encoding="utf8"))["files"]}
mism, rows = [], []
for stem, (gen, sub, generator, canonical, sources) in FIGURES.items():
    for ext in EXTS:
        rel = f"{PUB_DIR}/figures/{sub}/{stem}.{ext}"
        new = h(out / f"{gen}.{ext}")
        committed = (ROOT / rel)
        if not committed.is_file() or h(committed) != new or (recorded and recorded.get(rel) != new):
            mism.append(rel)
        rows.append({"path": rel, "sha256": new, "bytes": (out / f"{gen}.{ext}").stat().st_size,
                     "generator": f"scripts/publication_figures/{generator}",
                     "canonical_stage08_counterpart": f"results/primary/stage08/figures/{sub}/{canonical}.{ext}",
                     "frozen_sources": ([{"file": f"results/primary/stage08/figure_source_data/{s}", "sha256": h(FSD / s),
                                          "matches_index": idx.get(s) == h(FSD / s)} for s in sources]
                                        if isinstance(sources, list) else sources)})
check("51 regenerated files byte-identical to committed publication figures"
      + (" and manifest" if recorded else ""), not mism, f"{len(mism)} mismatches" if mism else "")

if "--write-manifest" in sys.argv and not fails:
    (pub / "PUBLICATION_FIGURE_MANIFEST.json").write_bytes(json.dumps({
        "set": "journal-presentation figures 2026-11-15",
        "status": "presentation re-renders for the manuscript; canonical Stage08 figures, Stage08 provenance and "
                  "Stage10 release records are unchanged and remain authoritative",
        "recomputation": "none: plotted values are the stored Stage08 figure source data; statuses and counts are "
                         "read from stored fields",
        "regenerate": "python scripts/publication_figures/regenerate_and_verify.py",
        "environment": "scripts/publication_figures/requirements-publication-figures.txt; Arial font",
        "known_wording_differences_from_canonical_stage08": WORDING_DIFFERENCES,
        "files": rows}, indent=1).encode("utf8") + b"\n")
    shutil.copyfile(out / "FIG01_v3_SOURCE_MANIFEST.json", pub / "FIG01_v3_SOURCE_MANIFEST.json")
    shutil.copyfile(out / "FIG01_v3_SOURCE_MANIFEST.csv", pub / "FIG01_v3_SOURCE_MANIFEST.csv")
    print("wrote PUBLICATION_FIGURE_MANIFEST.json and FIG01_v3_SOURCE_MANIFEST.{json,csv}")
shutil.rmtree(out, ignore_errors=True)
print(f"\nRESULT: {'PASS' if not fails else 'FAIL: ' + '; '.join(fails)}")
sys.exit(0 if not fails else 1)
