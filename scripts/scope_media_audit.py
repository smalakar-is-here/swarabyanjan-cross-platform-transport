"""E25: count BanglaClick items by platform and MEDIA_TYPE, and the empty content field.

Reads the released xlsx with the standard library only (no openpyxl), writes text-free counts.
Usage: python scripts/scope_media_audit.py
"""
import csv, hashlib, json, re, zipfile, datetime, subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT.parents[1]
XLSX = WS / "private_data/scope/raw_external/BanglaClick/repository_snapshot/Data/Final_4961_Dataset.xlsx"
OUT = ROOT / "results/scope_v1/E25"
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def read_xlsx(path):
    import xml.etree.ElementTree as ET
    z = zipfile.ZipFile(path)
    ss = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(NS + "si"):
            ss.append("".join(t.text or "" for t in si.iter(NS + "t")))
    sheet = sorted(n for n in z.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml", n))[0]
    rows = []
    for r in ET.fromstring(z.read(sheet)).iter(NS + "row"):
        cells = {}
        for c in r.iter(NS + "c"):
            col = re.match(r"[A-Z]+", c.get("r")).group(0)
            v = c.find(NS + "v"); t = c.get("t")
            if t == "s" and v is not None:
                val = ss[int(v.text)]
            elif t == "inlineStr":
                val = "".join(x.text or "" for x in c.iter(NS + "t"))
            else:
                val = v.text if v is not None else ""
            cells[col] = val
        rows.append(cells)
    head = rows[0]
    return [{head[k]: r.get(k, "") for k in head} for r in rows[1:]]


def main():
    rows = read_xlsx(XLSX)
    cols = list(rows[0].keys())
    plat = next(c for c in cols if c.upper() in ("PLATFORM", "SOURCE_PLATFORM", "SOURCE"))
    content = [c for c in cols if "CONTENT" in c.upper()]
    cnt = Counter((r[plat], (r.get("MEDIA_TYPE") or "").strip() or "<empty>") for r in rows)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "banglaclick_media_type_counts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["platform", "media_type", "n"])
        for (p, m), n in sorted(cnt.items()):
            w.writerow([p, m, n])
    summ = {"n_items": len(rows), "platform_column": plat,
            "content_columns": {c: sum(1 for r in rows if str(r.get(c, "")).strip() not in ("", "nan", "None")) for c in content}}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        rev = ""
    man = {"experiment": "E25", "created": datetime.datetime.now().isoformat(timespec="seconds"), "code_revision": rev,
           "inputs": {XLSX.name: hashlib.sha256(XLSX.read_bytes()).hexdigest()},
           "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob("*") if p.name != "MANIFEST.json"}}
    (OUT / "MANIFEST.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
