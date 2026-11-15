"""Marker-vs-text clearance QA for Figure 1 v3 (writes nothing).

Rebuilds the figure in memory (importing the generator) and checks that no plotted marker's rendered extent
overlaps any visible text's rendered extent. Exit 1 on any collision.
"""
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.text import Text  # noqa: E402

import make_fig1_v3 as F  # noqa: E402

fig = F.fig
fig.canvas.draw()
r = fig.canvas.get_renderer()
texts = [(t.get_text()[:25], t.get_window_extent(r)) for t in fig.findobj(Text)
         if t.get_visible() and t.get_text().strip()]
marks = []
for ln in fig.findobj(Line2D):
    if ln.get_marker() in (None, "None", "", " ") or not ln.get_visible():
        continue
    xy = ln.get_xydata()
    disp = ln.get_transform().transform(xy)
    rad = ln.get_markersize() * fig.dpi / 72 / 2
    for x, y in disp:
        marks.append((ln.get_marker(), x - rad, y - rad, x + rad, y + rad))
bad = []
for name, bb in texts:
    for mk, x0, y0, x1, y1 in marks:
        if min(bb.x1, x1) - max(bb.x0, x0) > 0.5 and min(bb.y1, y1) - max(bb.y0, y0) > 0.5:
            bad.append(f"{name!r} x marker {mk}")
print(f"markers checked: {len(marks)}; text items: {len(texts)}; collisions: {len(bad)}")
for b in bad[:10]:
    print("  ", b)
sys.exit(1 if bad else 0)
