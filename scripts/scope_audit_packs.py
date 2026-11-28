"""E27 annotator packs: build one zip per annotator for distribution, and take returned sheets back in.

  python scripts/scope_audit_packs.py build    # -> private_data/scope/annotation/distribution/SCOPE_annotation_pack_A{1,2,3}.zip
  python scripts/scope_audit_packs.py intake   # returned/*.xlsx -> returned/sheet_annotator_{k}.csv (+ practice check)

Each pack holds the annotator guideline (HTML), the practice workbook and the annotator's own main workbook.
Packs contain no answer key, no platform or block names, no BanglaClick levels and no model output; the
guideline is written for annotators and omits the sample design (ANNOTATION_PROTOCOL.md section 2), so the gold
items stay hidden. Everything is written under private_data/ (Git-ignored); nothing with text enters the repo.
Requires openpyxl.
"""
from __future__ import annotations
import hashlib, json, sys, time, zipfile
from pathlib import Path
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill, Protection
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scope import paths  # noqa: E402

CATS = ["Questions", "Inflammatory", "Curiosity gap/Teasing", "Ambiguous", "Exaggerate", "Graphic", "Formatting",
        "Bait & Switch", "none"]
FONT = "Nirmala UI"  # ships with Windows; renders Bengali

GUIDE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Clickbait annotation guideline</title><style>
body{font-family:Segoe UI,Nirmala UI,Arial,sans-serif;max-width:52rem;margin:2rem auto;padding:0 1rem;line-height:1.5;color:#1a1a1a;background:#fff}
h1{font-size:1.6rem}h2{font-size:1.2rem;margin-top:2rem;border-bottom:1px solid #ccc}
table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:.4rem .5rem;vertical-align:top;text-align:left}
th{background:#f1f1f1}.box{background:#f6f8fa;border-left:4px solid #3a6ea5;padding:.6rem 1rem;margin:1rem 0}
code{background:#eee;padding:0 .25rem}</style></head><body>
<h1>Clickbait annotation guideline</h1>
<p>Annotator ID: <b>__AID__</b> &middot; Pack version __VER__</p>
<p>Thank you for helping. You will read short Bengali headlines and social-media posts and decide, for each one, whether it is
<b>clickbait</b> under the definition below. There are no trick questions and no "right" proportion of clickbait: label each item
on its own merits.</p>

<h2>1. The definition</h2>
<div class="box"><b>Clickbait:</b> &ldquo;Intentionally luring readers to click on a particular content by exploiting their curiosity
defines a title as clickbait.&rdquo;</div>
<p>This definition and the categories below come from the published BanglaBait annotation guideline (Mahtab et al., RANLP 2023).
Use <b>only</b> this definition. Do not use other clickbait rules you may know.</p>

<h2>2. Clickbait categories</h2>
<p>An item is clickbait if it lures the reader in any of these ways.</p>
<table><tr><th>Category</th><th>What it means</th><th>Example (English translation)</th></tr>
<tr><td>Questions</td><td>Poses a question the reader must click to get answered.</td><td>&ldquo;Salman's actress Rombha mysteriously disappeared, what is she up to?&rdquo;</td></tr>
<tr><td>Inflammatory</td><td>Evokes strong emotion.</td><td>&ldquo;Lost his cool, kicked, then busted out the stamps before abusing the umpire!&rdquo;</td></tr>
<tr><td>Curiosity gap/Teasing</td><td>Leaves the reader in the dark, which tempts them to click.</td><td>&ldquo;Explore how to preserve meat without a refrigerator!&rdquo;</td></tr>
<tr><td>Ambiguous</td><td>Imprecise or unclear in a way that piques interest.</td><td>&ldquo;Not even Mir spared Nusrat&rdquo;</td></tr>
<tr><td>Exaggerate</td><td>Overstates what the content is likely to say.</td><td>&ldquo;Three foods when combined with blackberries, could kill you!&rdquo;</td></tr>
<tr><td>Graphic</td><td>Salacious, unsettling or implausible subject matter, used as a lure.</td><td>(a sensational violent-crime headline)</td></tr>
<tr><td>Formatting</td><td>Excessive punctuation or symbols used to grab attention.</td><td>&ldquo;Faria 'out' Porimoni 'in'!&rdquo;</td></tr>
<tr><td>Bait &amp; Switch</td><td>Promises more than the content can deliver.</td><td>&mdash;</td></tr></table>

<h2>3. How to decide</h2>
<ul>
<li>Label <b>1</b> if the text intentionally lures the reader to click by exploiting curiosity, in any of the ways above.</li>
<li>Label <b>0</b> otherwise. A text that is merely low in quality (informal, vague about its source, badly written, full of hashtags)
but does <b>not</b> try to lure a click is <b>0</b>.</li>
<li>A plain factual report of a dramatic event is not clickbait by itself. Ask: is the wording built to make me click to find out?</li>
<li>You see the text only. For &ldquo;Exaggerate&rdquo; and &ldquo;Bait &amp; Switch&rdquo;, judge from the text alone; if that makes you unsure,
lower your confidence and add a short note such as &ldquo;no landing page&rdquo;.</li>
<li>Ignore links, emoji and hashtags unless they are themselves used as the lure.</li>
</ul>

<h2>4. What to fill in</h2>
<table><tr><th>Column</th><th>What to enter</th></tr>
<tr><td><code>label_0_1</code></td><td><b>Required.</b> 1 = clickbait, 0 = not clickbait.</td></tr>
<tr><td><code>category</code></td><td>The main category if you chose 1 (pick from the list); <code>none</code> if you chose 0.</td></tr>
<tr><td><code>confidence_1_3</code></td><td>1 = unsure, 2 = fairly sure, 3 = sure.</td></tr>
<tr><td><code>notes</code></td><td>Optional, short. E.g. &ldquo;unreadable&rdquo;, &ldquo;not Bengali&rdquo;, &ldquo;no landing page&rdquo;.</td></tr></table>
<p>Do <b>not</b> change the <code>order</code>, <code>ID</code> or <code>text</code> columns, add or delete rows, or re-sort the sheet.
Those columns are locked to prevent accidents.</p>

<h2>5. Steps</h2>
<ol>
<li><b>Practice first.</b> Read sections 1&ndash;4, then label the 20 items in <code>01_practice_sheet___AID__.xlsx</code> and return it
to the coordinator. The coordinator will go through it with you before you start the main sheet.</li>
<li><b>Main sheet.</b> Label all items in <code>02_main_sheet___AID__.xlsx</code>, top to bottom, in the given order. It takes about
3&ndash;4 hours; work in several sittings and take breaks.</li>
<li><b>Work alone.</b> Do not discuss items with the other annotators, search the internet for them, or look up where they came from.</li>
<li><b>Return</b> the completed workbook to the coordinator with its file name unchanged.</li>
</ol>

<h2>6. Confidentiality</h2>
<p>The texts come from research datasets licensed for non-commercial research use. Use them only for this labelling task. Do not copy,
share or publish them. After you have returned your sheets, delete your copy of this pack.</p>
<p>Only your anonymous annotator ID is recorded with your labels.</p>
</body></html>
"""


def _workbook(df: pd.DataFrame, id_col: str, aid: str, title: str) -> Workbook:
    wb = Workbook(); ws = wb.active; ws.title = "labels"
    cols = ["order", "ID", "text", "label_0_1", "category", "confidence_1_3", "notes"]
    ws.append(cols)
    for i, r in enumerate(df.itertuples(index=False), 1):
        ws.append([int(getattr(r, "order", i)), getattr(r, id_col), str(r.text), None, None, None, None])
    head = PatternFill("solid", fgColor="DDE6F0"); locked = PatternFill("solid", fgColor="F2F2F2")
    for c in ws[1]:
        c.font = Font(name=FONT, bold=True); c.fill = head
    widths = {"A": 7, "B": 9, "C": 70, "D": 11, "E": 22, "F": 15, "G": 28}
    for k, w in widths.items():
        ws.column_dimensions[k].width = w
    n = ws.max_row
    for row in ws.iter_rows(min_row=2, max_row=n):
        for j, c in enumerate(row):
            c.font = Font(name=FONT, size=12 if j == 2 else 11)
            c.alignment = Alignment(wrap_text=(j in (2, 6)), vertical="top")
            if j < 3:
                c.fill = locked
            else:
                c.protection = Protection(locked=False)
    for spec, rng, msg in ((f'"0,1"', f"D2:D{n}", "Enter 0 or 1"),
                           (f'"{",".join(CATS)}"', f"E2:E{n}", "Pick a category from the list"),
                           ('"1,2,3"', f"F2:F{n}", "Enter 1, 2 or 3")):
        dv = DataValidation(type="list", formula1=spec, allow_blank=True, showErrorMessage=True, error=msg, errorTitle="Invalid value")
        ws.add_data_validation(dv); dv.add(rng)
    ws.freeze_panes = "D2"
    ws.protection.sheet = True  # no password: guards the order/ID/text columns against accidental edits
    ws.protection.formatColumns = False; ws.protection.formatRows = False
    info = wb.create_sheet("read_me")
    for line in (title, f"Annotator ID: {aid}", "", "Fill in columns D-G of the 'labels' sheet for every row.",
                 "label_0_1: 1 = clickbait, 0 = not (required)", "category: main category if 1; 'none' if 0",
                 "confidence_1_3: 1 unsure, 2 fairly sure, 3 sure", "notes: optional",
                 "Do not re-sort, add or delete rows. Full guideline: 00_READ_FIRST_guideline.html", "",
                 "Categories: " + "; ".join(CATS)):
        info.append([line])
    info.column_dimensions["A"].width = 100
    for c in info["A"]:
        c.font = Font(name=FONT, size=11)
    return wb


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build():
    ann = paths.private_root() / "annotation"; dist = ann / "distribution"
    if dist.exists() and any(dist.glob("*.zip")):
        sys.exit(f"packs already exist in {dist}; refusing to overwrite")
    ver = time.strftime("%Y-%m-%d")
    practice = pd.read_csv(ann / "practice_sheet.csv", encoding="utf-8-sig")
    manifest = {"built": time.strftime("%Y-%m-%dT%H:%M:%S"), "packs": {}}
    for k in (1, 2, 3):
        aid = f"A{k}"; stage = dist / f"SCOPE_annotation_pack_{aid}"; stage.mkdir(parents=True, exist_ok=True)
        main = pd.read_csv(ann / f"sheet_annotator_{k}.csv", encoding="utf-8-sig")
        assert main.audit_id.is_unique and len(main) == 449, (k, len(main))
        (stage / "00_READ_FIRST_guideline.html").write_text(GUIDE.replace("__AID__", aid).replace("__VER__", ver), encoding="utf-8")
        _workbook(practice, "practice_id", aid, "Practice sheet (20 items)").save(stage / f"01_practice_sheet_{aid}.xlsx")
        _workbook(main, "audit_id", aid, f"Main sheet ({len(main)} items)").save(stage / f"02_main_sheet_{aid}.xlsx")
        z = dist / f"SCOPE_annotation_pack_{aid}.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(stage.iterdir()):
                zf.write(f, f"{stage.name}/{f.name}")
        manifest["packs"][aid] = {"zip": z.name, "sha256": _sha(z), "files": {f.name: _sha(f) for f in sorted(stage.iterdir())},
                                  "main_items": len(main), "practice_items": len(practice)}
        print(z.name, manifest["packs"][aid]["sha256"][:16])
    (dist / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")


def _read_returned(path: Path, id_col: str) -> pd.DataFrame:
    ws = load_workbook(path, data_only=True)["labels"]
    rows = list(ws.iter_rows(values_only=True)); df = pd.DataFrame(rows[1:], columns=rows[0])
    return df.rename(columns={"ID": id_col})


def intake():
    ann = paths.private_root() / "annotation"; ret = ann / "returned"; ret.mkdir(exist_ok=True)
    pk = pd.read_csv(ann / "practice_key_PRIVATE.csv")
    for k in (1, 2, 3):
        aid = f"A{k}"
        p = ret / f"01_practice_sheet_{aid}.xlsx"
        if p.exists():
            d = _read_returned(p, "practice_id").merge(pk, on="practice_id")
            lab = pd.to_numeric(d.label_0_1, errors="coerce")
            print(f"{aid} practice: {int(lab.notna().sum())}/20 labelled, agreement with BanglaBait "
                  f"{(lab == d.banglabait_label).mean():.0%} (protocol: repeat once if below 70%)")
        m = ret / f"02_main_sheet_{aid}.xlsx"
        if not m.exists():
            print(f"{aid} main: not returned"); continue
        orig = pd.read_csv(ann / f"sheet_annotator_{k}.csv", encoding="utf-8-sig")
        d = _read_returned(m, "audit_id")
        problems = []
        if list(d.audit_id) != list(orig.audit_id):
            problems.append("rows added, removed or re-ordered")
        elif not (d.text.astype(str) == orig.text.astype(str)).all():
            problems.append("text column changed")
        lab = pd.to_numeric(d.label_0_1, errors="coerce")
        if lab.isna().any():
            problems.append(f"{int(lab.isna().sum())} items without a label")
        if not lab.dropna().isin([0, 1]).all():
            problems.append("labels other than 0/1")
        bad_cat = d.category.dropna()[~d.category.dropna().isin(CATS)]
        if len(bad_cat):
            problems.append(f"{len(bad_cat)} unknown categories")
        out = ret / f"sheet_annotator_{k}.csv"
        if problems:
            print(f"{aid} main: NOT converted (return to annotator): " + "; ".join(problems)); continue
        if out.exists():
            print(f"{aid} main: {out.name} exists; not overwritten"); continue
        d = d.assign(label_0_1=lab.astype("Int64"))
        d[["order", "audit_id", "text", "label_0_1", "category", "confidence_1_3", "notes"]].to_csv(out, index=False, encoding="utf-8-sig")
        print(f"{aid} main: converted -> {out.name} ({int(lab.sum())} clickbait of {len(d)})")


if __name__ == "__main__":
    {"build": build, "intake": intake}[sys.argv[1]]()
