#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_query_expansions.py
-------------------------
Programmatically generate deterministic query expansions for MCQ datasets.

• Input:  JSONL (one JSON object per line) or JSON (list/dict)
• Output: JSONL with an added field (default: "query_expansions")

Expansions are derived from question (+ options/subject/topic if enabled)
using rules: synonyms, US/UK spellings, acronyms, and MCQ phrase heuristics.

IMPORTANT: For fair evaluation, DO NOT include explanation/rationale fields
in dev/test. The --include-explanation flag is provided only for debugging.

Usage
-----
python build_query_expansions.py \
  --input data/dev.jsonl \
  --output data/dev.with_qexp.jsonl \
  --include-options --include-subject --include-topic \
  --max-expansions 30

Later (test set), run the same command with the new file path.
"""

import os
import re
import json
import sys
import argparse
from typing import List, Dict, Any

# ---------------------------
# Normalization
# ---------------------------
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

# ---------------------------
# Synonyms / expansions (lowercase keys)
# Extend as needed; keep stable for reproducibility.
# ---------------------------
SYNONYMS: Dict[str, List[str]] = {
    # general medical spelling/word variants
    "carcinoma": ["cancer", "malignancy", "ca"],
    "oesophagus": ["esophagus"],
    "esophagus": ["oesophagus"],
    "haemoglobin": ["hemoglobin"],
    "haemorrhage": ["hemorrhage", "bleeding"],
    "oedema": ["edema"],
    "foetal": ["fetal"],
    "foetus": ["fetus"],
    "color": ["colour"],
    "anaemia": ["anemia"],
    "hypokalaemia": ["hypokalemia"],
    "hyperkalaemia": ["hyperkalemia"],
    "tympanic membrane": ["eardrum"],

    # high-yield biomedical acronyms / entities
    "hcn": [
        "hyperpolarization-activated cyclic nucleotide-gated",
        "funny current", "if current", "if channels", "pacemaker current"
    ],
    "trali": ["transfusion-related acute lung injury"],
    "siadh": ["syndrome of inappropriate antidiuretic hormone secretion", "inappropriate adh"],
    "pcos": ["polycystic ovary syndrome", "stein leventhal"],
    "rcc": ["renal cell carcinoma", "hypernephroma"],
    "cmv": ["cytomegalovirus"],
    "pet": ["positron emission tomography"],
    "fdg": ["fluorodeoxyglucose", "f-18 fdg"],
    "gdm": ["gestational diabetes mellitus"],
    "bpp": ["biophysical profile", "manning score"],
    "ogtt": ["oral glucose tolerance test"],
    "opg": ["orthopantomogram", "panoramic radiograph"],
    "ebv": ["epstein barr virus"],
    "psgn": ["post streptococcal glomerulonephritis", "poststreptococcal gn"],
    "rhd": ["rheumatic heart disease"],
    "tb": ["tuberculosis"],

    # anatomy / dental / materials
    "ayre's spatula": ["pap smear spatula", "cervical cytology spatula"],
    "frankfort": ["frankfurt", "fh plane", "porion orbitale"],
    "inca bone": ["interparietal bone", "goethe's ossicle", "lambda bone"],
    "wormian": ["sutural bone"],
    "dicor": ["castable glass ceramic"],
    "mta": ["mineral trioxide aggregate"],
    "ah plus": ["epoxy resin sealer"],
    "gutta percha": ["gp"],

    # pharm / adverse effects
    "ethambutol": ["optic neuritis", "red green color blindness"],

    # oncology
    "ewing": ["ewsr1 fli1", "t(11;22)", "primitive neuroectodermal tumor", "pnet"],

    # drugs / mechanisms
    "prucalopride": ["5-ht4 agonist", "chronic constipation"],

    # derm
    "lichen planus": ["violaceous flat topped papules", "wickham striae"],

    # urology
    "dj stent": ["double j stent", "ureteric stent", "ureteral stent", "j stent"],

    # labs
    "vldl": ["very low density lipoprotein"],
    "ldl": ["low density lipoprotein"],
    "hdl": ["high density lipoprotein"],
    "ggt": ["gamma glutamyl transferase"],
    "ast": ["sgot", "aspartate aminotransferase"],
    "alt": ["sgpt", "alanine aminotransferase"],
    "alp": ["alkaline phosphatase"],

    # ventilation
    "niv": ["non-invasive ventilation", "noninvasive ventilation"],
    "simv": ["synchronized intermittent mandatory ventilation"],
    "acmv": ["assist control ventilation", "ac mode"],
    "psv": ["pressure support ventilation"],
}

# ---------------------------
# Phrase-level expanders (regex -> list of extra phrases)
# ---------------------------
PHRASE_EXPANDERS = [
    (re.compile(r"\bwhich of the following\b", re.I),
     ["exam question", "mcq", "select one", "choose the correct", "except", "true/false"]),
    (re.compile(r"\bexcept\b", re.I),
     ["not", "is NOT", "false statement", "all are true except"]),
    (re.compile(r"\bmost common\b", re.I),
     ["commonest", "mc", "frequent", "prevalent"]),
    (re.compile(r"\bsite\b", re.I),
     ["location", "region", "segment"]),
    (re.compile(r"\bacid[- ]?base\b", re.I),
     ["arterial blood gas", "abg", "acidemia alkalemia"]),
    (re.compile(r"\bcarcinoma\b", re.I),
     ["cancer", "malignancy", "ca"]),
    (re.compile(r"\bgingiv\w+", re.I),
     ["periodontal", "gum", "gingiva"]),
    (re.compile(r"\bana?esthesia|an(a)?esthesia\b", re.I),
     ["sedation", "anesthetic", "monitoring"]),
    (re.compile(r"\bfracture\b", re.I),
     ["break", "fx"]),
    (re.compile(r"\bhypotension\b", re.I),
     ["low blood pressure", "bp drop"]),
    (re.compile(r"\bmyocard\w+", re.I),
     ["heart muscle", "cardiac"]),
    (re.compile(r"\bviab(le|ility)\b", re.I),
     ["hibernating myocardium", "myocardial viability"]),
    (re.compile(r"\bpap smear\b", re.I),
     ["cervical cytology", "papanicolaou test"]),
    (re.compile(r"\bperiodontal\b", re.I),
     ["gum", "gingival"]),
    (re.compile(r"\bosseous\b", re.I),
     ["bony"]),
    (re.compile(r"\bureter(al|ic)\b", re.I),
     ["double j stent", "dj stent", "j stent"]),
    (re.compile(r"\bconfidence interval\b|\bCI\b", re.I),
     ["95% CI", "margin of error", "precision"]),
    (re.compile(r"\bcase[- ]?control\b", re.I),
     ["retrospective study", "odds ratio"]),
    (re.compile(r"\bcohort\b", re.I),
     ["prospective study", "relative risk"]),
    (re.compile(r"\brandomi[sz]ed\b", re.I),
     ["randomized", "RCT"]),
    (re.compile(r"\bflow cytometry\b", re.I),
     ["facs", "forward scatter", "side scatter"]),
]

# ---------------------------
# US/UK spelling pairs
# ---------------------------
US_UK_PAIRS = [
    ("hemorrhage", "haemorrhage"),
    ("fetus", "foetus"),
    ("edema", "oedema"),
    ("color", "colour"),
    ("esophagus", "oesophagus"),
]

STOP = set("""
a an the is are was were be been of for to in on at from by and or if with without
which who whom whose that this these those into over under after before during
not except all any most more less least
""".split())

def extract_acronyms(text: str) -> List[str]:
    """Pick tokens mostly uppercase letters (allow digits)."""
    return re.findall(r"\\b[A-Z]{2,}[A-Z0-9]*\\b", text or "")

def expand_text_fields(*texts: str) -> List[str]:
    """Create expansion candidates from provided text fields only."""
    expansions = set()
    for t in texts:
        if not t:
            continue
        s = norm(t)
        if s:
            expansions.add(s)  # include normalized raw text
            # Phrase-level expansions
            for pat, adds in PHRASE_EXPANDERS:
                if pat.search(t):
                    expansions.update(a.lower() for a in adds)
            # US/UK variants
            for us, uk in US_UK_PAIRS:
                if us in s: expansions.add(uk)
                if uk in s: expansions.add(us)
            # Token-level synonyms (uni- and bi-grams)
            words = [w for w in re.split(r"[^a-z0-9']+", s) if w and w not in STOP]
            for i in range(len(words)):
                w = words[i]
                if w in SYNONYMS:
                    expansions.update(SYNONYMS[w])
                if i < len(words)-1:
                    bigram = f"{words[i]} {words[i+1]}"
                    if bigram in SYNONYMS:
                        expansions.update(SYNONYMS[bigram])
            # Acronyms (use original case to detect, map to lowercase key)
            for ac in extract_acronyms(t):
                key = ac.lower()
                if key in SYNONYMS:
                    expansions.update(SYNONYMS[key])
                expansions.add(key)  # keep the acronym itself
    # Deterministic ordering: phrases first, then acronym-ish, then by length/name
    expansions = sorted(set(expansions),
                        key=lambda e: (0 if " " in e else 1,
                                       0 if (e.isupper() or (len(e) <= 5 and e.isalnum())) else 1,
                                       len(e), e))
    return expansions

_BAD = re.compile(r"\\b(answer|ans[:\\-]?|correct(?:\\s+option)?)\\b", re.I)

def sanitize_terms(terms: List[str], max_len: int = 120) -> List[str]:
    """Drop leaked-answer cues and overly long blobs; keep order."""
    out = []
    for t in terms:
        if not t:
            continue
        if _BAD.search(t):
            continue
        if len(t) > max_len:
            continue
        out.append(t)
    # Deduplicate while preserving order
    seen, cleaned = set(), []
    for t in out:
        if t not in seen:
            seen.add(t); cleaned.append(t)
    return cleaned

def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    if not txt:
        return []
    # Try JSON list/dict first
    try:
        obj = json.loads(txt)
        if isinstance(obj, list): return obj
        if isinstance(obj, dict): return [obj]
    except json.JSONDecodeError:
        pass
    # Fallback to JSONL
    out = []
    for line in txt.splitlines():
        line = line.strip()
        if not line: continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out

def save_jsonl(path: str, items: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser(description="Add query_expansions to MCQ data (dev/test-safe).")
    ap.add_argument("--input", required=True, help="Path to .jsonl or .json")
    ap.add_argument("--output", required=True, help="Path to write .jsonl with expansions")
    ap.add_argument("--max-expansions", type=int, default=30, help="Cap per-item expansions")
    ap.add_argument("--include-options", action="store_true", help="Include options (opa..opf) in expansion")
    ap.add_argument("--include-subject", action="store_true", help="Include subject_name in expansion")
    ap.add_argument("--include-topic", action="store_true", help="Include topic_name in expansion")
    ap.add_argument("--include-explanation", action="store_true",
                    help="INSECURE for eval: include 'exp'/'explanation' fields (debug only).")
    args = ap.parse_args()

    # Handle relative paths from project root
    if not os.path.isabs(args.input):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(script_dir, "..", "..")
        args.input = os.path.join(project_root, args.input)
    
    if not os.path.isabs(args.output):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(script_dir, "..", "..")
        args.output = os.path.join(project_root, args.output)

    if args.include_explanation:
        print("WARNING: --include-explanation is not recommended for dev/test (risk of leakage).",
              file=sys.stderr)

    data = load_json_or_jsonl(args.input)
    out = []
    for r in data:
        fields: List[str] = [r.get("question","")]
        if args.include_options:
            for k in ("opa","opb","opc","opd","ope","opf"):
                v = r.get(k)
                if v: fields.append(v)
        if args.include_subject and r.get("subject_name"):
            fields.append(r["subject_name"])
        if args.include_topic and r.get("topic_name"):
            fields.append(r["topic_name"])
        if args.include_explanation:
            # Debug-only; do not use for evaluation
            for k in ("exp","explanation","rationale"):
                if r.get(k):
                    fields.append(r[k])

        exps = expand_text_fields(*fields)
        exps = sanitize_terms(exps)

        # Trim if too long
        if len(exps) > args.max_expansions:
            exps = exps[:args.max_expansions]

        r2 = dict(r)
        r2["query_expansions"] = exps
        out.append(r2)

    save_jsonl(args.output, out)
    print(f"✅ Wrote {len(out)} items → {args.output}")

if __name__ == "__main__":
    main()
