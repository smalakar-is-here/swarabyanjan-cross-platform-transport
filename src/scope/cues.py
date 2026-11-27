"""E7: cue extractor audit and correction.

v1.0.0 (R6, frozen) splits on whitespace only, so a lexical cue followed by punctuation
(e.g. "কেন?", "আপনার," or a term before the danda "।") is not counted, and the question mark
is counted twice (interrogative_structure and punctuation_intensity).
v1.1.0 (this work) tokenises on whitespace AND punctuation (ASCII, Bengali danda/double danda,
curly quotes, dashes, ellipsis), keeps multi-word phrases as substring matches, and counts "?"
only in punctuation_intensity. Lexicons and all other rules are unchanged, so differences
isolate the tokenisation defect.
"""
from __future__ import annotations
import json, math, re, unicodedata
from collections import Counter
import numpy as np
import pandas as pd
from .paths import REPO

SCHEMA = json.loads((REPO / "configs/CUE_FEATURE_SCHEMA.json").read_text(encoding="utf-8"))
PUNCT = r"[\s!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~।॥‘’“”—–…]+"


def normalize(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    t = unicodedata.normalize("NFKC", str(value))
    for z in ("​", "‌", "‍", "﻿"):
        t = t.replace(z, "")
    return re.sub(r"\s+", " ", t.casefold().strip())


def _count(text, tokens, terms):
    tc = Counter(tokens); total = 0
    for term in terms:
        t = normalize(term)
        total += text.count(t) if " " in t else tc.get(t, 0)
    return total


def extract(titles, version="1.0"):
    fams = SCHEMA["families"]; out = []
    for raw in titles:
        text = normalize(raw)
        toks = text.split() if version == "1.0" else [t for t in re.split(PUNCT, text) if t]
        row = {}
        for name, spec in fams.items():
            if name == "specificity_numerals":
                c = sum(len(re.findall(p, text, flags=re.I)) for p in spec["patterns"])
            elif name == "punctuation_intensity":
                c = sum(text.count(p) for p in spec["punctuation"]) + len(re.findall(spec["repeated_pattern"], text))
            else:
                c = _count(text, toks, spec.get("terms", []))
                if version == "1.0":
                    c += sum(text.count(p) for p in spec.get("punctuation", []))
            row[name] = int(c)
        out.append(row)
    return pd.DataFrame(out)


def audit(items: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """items: one row per item with title, env and the frozen v1.0 cue columns."""
    v10 = extract(items.title, "1.0"); v11 = extract(items.title, "1.1")
    replica = {c: float((v10[c].to_numpy() == items[c].to_numpy()).mean()) for c in v10.columns}
    rows = []
    for env, idx in items.groupby("env").indices.items():
        for c in v10.columns:
            a, b = v10[c].to_numpy()[idx] > 0, v11[c].to_numpy()[idx] > 0
            rows.append({"env": env, "cue": c, "presence_v10": float(a.mean()), "presence_v11": float(b.mean()),
                         "presence_changed": float((a != b).mean()), "gained": float((~a & b).mean()), "lost": float((a & ~b).mean())})
    return pd.DataFrame(rows), {"v10_replica_exact_match": replica, "v11": v11}
