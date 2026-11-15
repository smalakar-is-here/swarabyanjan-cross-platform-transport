"""Shared, presentation-only helpers for the SWARABYANJAN figure redesign (2026-11-15).

Every plotted number is read from the frozen Stage08 figure source-data CSVs in
results/primary/stage08/figure_source_data/ of this repository. Each CSV's
SHA-256 is checked against the repository's own FIGURE_SOURCE_DATA_INDEX.json before use.
No statistic is recomputed; no interval, p-value or status is derived.
"""
from __future__ import annotations

import hashlib
import os
import json
import math
import pathlib

import matplotlib as mpl
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "results/primary/stage08/figure_source_data"
TAB = ROOT / "results/primary/stage08/tables"
# output directory: $SWFIG_OUT, default build/publication_figures (git-ignored)
OUT = pathlib.Path(os.environ.get("SWFIG_OUT", str(ROOT / "build" / "publication_figures")))
OUT.mkdir(parents=True, exist_ok=True)

# Final printed size: CAS text block is ~467 pt (6.48 in). Figures are drawn at 6.4 in and
# included at \textwidth, so the font sizes below are the printed sizes (scale ~1.0).
FIG_W = 6.4

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.title_fontsize": 8.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,      # embed TrueType, keep text selectable
    "svg.fonttype": "none",  # keep SVG text as text
    "svg.hashsalt": "swarabyanjan-fig-v2",  # deterministic SVG ids
    "savefig.dpi": 600,
    "figure.dpi": 100,
})

# Okabe-Ito (colour-blind safe) for the four model families; each also has its own marker
# so the encoding survives grayscale printing.
MODELS = ["ebm", "tfidf_lr", "banglabert", "xlmr"]
MODEL_LABEL = {"ebm": "EBM", "tfidf_lr": "TF-IDF + LR", "banglabert": "BanglaBERT", "xlmr": "XLM-R"}
MODEL_COLOR = {"ebm": "#0072B2", "tfidf_lr": "#E69F00", "banglabert": "#009E73", "xlmr": "#CC79A7"}
MODEL_MARKER = {"ebm": "o", "tfidf_lr": "s", "banglabert": "^", "xlmr": "D"}

# Platforms use a separate (Paul Tol) palette so colour never means "model" and "platform" at once.
PLATFORMS = ["news", "facebook", "youtube", "twitter"]
PLATFORM_LABEL = {"news": "News", "facebook": "Facebook", "youtube": "YouTube", "twitter": "Twitter"}
PLATFORM_COLOR = {"news": "#332288", "facebook": "#117733", "youtube": "#AA3377", "twitter": "#DDAA33"}
PLATFORM_MARKER = {"news": "o", "facebook": "s", "youtube": "^", "twitter": "D"}

DOMAIN_LABEL = {
    "source_test": "Source\nTEST",
    "banglaclick:news": "News",
    "banglaclick:facebook": "Facebook",
    "banglaclick:youtube": "YouTube",
    "banglaclick:twitter": "Twitter",
    "baitbuster:youtube_human": "BaitBuster",
}
TARGETS = ["banglaclick:news", "banglaclick:facebook", "banglaclick:youtube", "banglaclick:twitter",
           "baitbuster:youtube_human"]

CUES = ["curiosity_information_withholding", "direct_address_imperative", "exaggeration_superlative",
        "interrogative_structure", "punctuation_intensity", "sensational_emotional_loading",
        "specificity_numerals", "vague_deictic_reference"]
CUE_LABEL = {
    "curiosity_information_withholding": "Curiosity / information withholding",
    "direct_address_imperative": "Direct address / imperative",
    "exaggeration_superlative": "Exaggeration / superlative",
    "interrogative_structure": "Interrogative structure",
    "punctuation_intensity": "Punctuation intensity",
    "sensational_emotional_loading": "Sensational / emotional loading",
    "specificity_numerals": "Specificity / numerals",
    "vague_deictic_reference": "Vague / deictic reference",
}
NE_GREY = "#8C8C8C"
SOURCE_LOG: dict[str, str] = {}


def sha256(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _index() -> dict:
    idx = json.loads((SRC / "FIGURE_SOURCE_DATA_INDEX.json").read_text(encoding="utf8"))
    return {e["source_csv"]: e["source_csv_sha256"] for e in idx["source_files"]}


_IDX = None


def load(name: str) -> pd.DataFrame:
    """Load a frozen source CSV after verifying its hash against the repository index."""
    global _IDX
    _IDX = _IDX or _index()
    p = SRC / name
    h = sha256(p)
    if _IDX.get(name) != h:
        raise SystemExit(f"HASH MISMATCH for {name}: {h} != index {_IDX.get(name)}")
    SOURCE_LOG[name] = h
    # round_trip: pandas' default fast parser can differ from the stored text by 1 ulp.
    return pd.read_csv(p, float_precision="round_trip")


def load_table(rel: str) -> pd.DataFrame:
    p = TAB / rel
    SOURCE_LOG["tables/" + rel] = sha256(p)
    return pd.read_csv(p, float_precision="round_trip")


def ci(v):
    """Parse a frozen '[lo, hi]' CI string. Returns (nan, nan) if absent (NE)."""
    if isinstance(v, str) and v.strip().startswith("["):
        lo, hi = json.loads(v)
        return float(lo), float(hi)
    return (math.nan, math.nan)


def is_true(v) -> bool:
    return v is True or (isinstance(v, str) and v.strip().lower() == "true")


def panel_label(ax, letter, x=-0.02, y=1.02):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold",
            ha="right", va="bottom")


TEXT_QA: dict[str, list] = {}


def text_collisions(fig, pad=0.5):
    """Return pairs of visible text artists whose rendered boxes overlap, or that extend past the figure edge."""
    from matplotlib.text import Text
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    tick_all, tick_drawn = set(), set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                tick_all.update({id(tk.label1), id(tk.label2)})
            if not ax.axison:  # axis("off"): no tick labels are drawn
                continue
            for tk in axis._update_ticks():  # only ticks inside the view interval are drawn
                for lab in (tk.label1, tk.label2):
                    if lab.get_visible():
                        tick_drawn.add(id(lab))
    items = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip() or t.get_alpha() == 0:
            continue
        if id(t) in tick_all and id(t) not in tick_drawn:
            continue
        try:
            bb = t.get_window_extent(r)
        except Exception:
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        items.append((t.get_text().replace("\n", " ")[:30], bb, _quad(t, bb, r)))
    out = []
    for name, bb, _ in items:
        if bb.x0 < -pad or bb.y0 < -pad or bb.x1 > W + pad or bb.y1 > H + pad:
            out.append(f"CLIPPED {name!r}")
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][1], items[j][1]
            ix = min(a.x1, b.x1) - max(a.x0, b.x0) - pad
            iy = min(a.y1, b.y1) - max(a.y0, b.y0) - pad
            if ix > 0 and iy > 0 and _sat_overlap(items[i][2], items[j][2], pad):
                out.append(f"OVERLAP {items[i][0]!r} x {items[j][0]!r}")
    return out


def _quad(t, bb, renderer):
    """True (possibly rotated) text rectangle as 4 corners, rebuilt from its axis-aligned extent."""
    import math as _m
    ang = _m.radians(t.get_rotation() % 180)
    if abs(_m.sin(ang)) < 1e-6:
        return [(bb.x0, bb.y0), (bb.x1, bb.y0), (bb.x1, bb.y1), (bb.x0, bb.y1)]
    rot = t.get_rotation()
    t.set_rotation(0)
    ub = t.get_window_extent(renderer)
    t.set_rotation(rot)
    w, h = ub.width, ub.height
    s, c = abs(_m.sin(ang)), abs(_m.cos(ang))
    if 0 < ang < _m.pi / 2:   # counter-clockwise tilt
        return [(bb.x0 + h * s, bb.y0), (bb.x1, bb.y0 + w * s), (bb.x1 - h * s, bb.y1), (bb.x0, bb.y1 - w * s)]
    return [(bb.x0, bb.y0 + h * c), (bb.x1 - h * s, bb.y0), (bb.x1, bb.y1 - h * c), (bb.x0 + h * s, bb.y1)]


def _sat_overlap(p, q, pad):
    """Separating-axis test for two convex quads (overlap deeper than `pad` pixels)."""
    for poly in (p, q):
        for k in range(4):
            x1, y1 = poly[k]; x2, y2 = poly[(k + 1) % 4]
            nx, ny = y2 - y1, x1 - x2
            n = (nx * nx + ny * ny) ** 0.5 or 1.0
            pa = [(x * nx + y * ny) / n for x, y in p]
            pb = [(x * nx + y * ny) / n for x, y in q]
            if min(max(pa), max(pb)) - max(min(pa), min(pb)) <= pad:
                return False
    return True


def export(fig, stem: str):
    TEXT_QA[stem] = text_collisions(fig)
    if TEXT_QA[stem]:
        print(f"  [text QA] {stem}: {len(TEXT_QA[stem])} issue(s): " + "; ".join(TEXT_QA[stem][:8]))
    paths = {}
    meta = {"Creator": "SWARABYANJAN figure_workbench prototypes (presentation only)",
            "CreationDate": None, "ModDate": None}
    for ext in ("pdf", "svg", "png"):
        p = OUT / f"{stem}.{ext}"
        kw = {"metadata": meta} if ext == "pdf" else ({"metadata": {"Date": None}} if ext == "svg" else {})
        fig.savefig(p, **kw)
        paths[ext] = p
    return paths
