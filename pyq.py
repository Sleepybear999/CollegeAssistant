"""Previous-year-paper search over erp_metadata.csv. No LLM or embeddings needed."""
import csv, os, re, difflib

CSV_PATH = os.environ.get("PYQ_CSV", os.path.join(os.path.dirname(os.path.abspath(__file__)), "erp_metadata.csv"))
MAX_RESULTS = 50

ALIASES = {
    "dbms": "database management system", "os": "operating system", "dsa": "data structure algorithm",
    "coa": "computer organization architecture", "cn": "computer network", "oop": "object oriented programming",
    "oops": "object oriented programming", "toc": "theory computation", "daa": "design analysis algorithm",
    "ml": "machine learning", "ai": "artificial intelligence", "dl": "deep learning",
    "se": "software engineering", "cd": "compiler design", "dip": "digital image processing",
    "dsp": "digital signal processing", "iot": "internet things", "maths": "mathematics",
    "math": "mathematics", "phy": "physics", "chem": "chemistry", "evs": "environmental",
    "bee": "basic electrical engineering", "dld": "digital logic design", "ds": "data structure",
    "wt": "web technology", "cg": "computer graphics", "nlp": "natural language processing",
}
STOP = set("""paper papers pyq pyqs question questions previous year years prev old of the for me give send get
show find need want please pls i a an my and link links pdf exam exams examination semester sem mid end midsem endsem
mst est final term in on to with any all is are do you have there can u of batch session""".split())
PAPER_WORDS = re.compile(r"\b(papers?|pyqs?|question\s*papers?|previous\s*years?|old\s*papers?|prev\s*year)\b", re.I)
MID = re.compile(r"\b(mid\s*-?\s*sem\w*|mid\s*-?\s*term|mst|midsem|mid)\b", re.I)
END = re.compile(r"\b(end\s*-?\s*sem\w*|endsem|final|est)\b", re.I)
COURSES = [("m.tech", r"\bm\.?\s*tech\b"), ("b.tech", r"\bb\.?\s*tech\b"), ("mba", r"\bmba\b"), ("m.sc.", r"\bm\.?\s*sc\b")]
# (regex, case_sensitive, department substring)
DEPTS = [
    (r"\b(cse|computer science)\b", False, "COMPUTER SCIENCE AND ENGINEERING"),
    (r"\bIT\b|information technology", True, "INFORMATION TECHNOLOGY"),
    (r"\b(ece|electronics and communication)\b", False, "ELECTRONICS AND COMMUNICATION"),
    (r"\b(ee|electrical)\b", False, "ELECTRICAL ENGINEERING"),
    (r"\b(mech|mechanical)\b", False, "MECHANICAL ENGINEERING"),
    (r"\bME\b", True, "MECHANICAL ENGINEERING"),
    (r"\bcivil\b", False, "CIVIL ENGINEERING"),
    (r"\btextile\b", False, "TEXTILE"),
    (r"\bchemical\b", False, "CHEMICAL ENGINEERING"),
    (r"\b(ice|instrumentation)\b", False, "INSTRUMENTATION"),
    (r"\b(ipe|production)\b", False, "INDUSTRIAL AND PRODUCTION"),
    (r"\b(biotech|biotechnology|bio technology)\b", False, "BIO"),
    (r"\b(dse|data science)\b", False, "DATA SCIENCE"),
    (r"\bfirst\s*year\b|\b1st\s*year\b", False, "FIRST YEAR"),
]

def _norm(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

def _load():
    rows, seen = [], set()
    with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            url = (r.get("url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            code = (r.get("subject_code") or "").strip()
            name = (r.get("subject_name") or "").strip() or code
            sess = (r.get("Session") or "").strip()
            ym = re.search(r"(\d{4})\s*$", sess)
            rows.append({
                "title": name, "code": code, "exam": (r.get("exam") or "").strip(), "session": sess,
                "course": (r.get("Course") or "").strip(), "dept": (r.get("Department/Group") or "").strip(),
                "batch": (r.get("Batch") or "").strip(), "url": url,
                "_words": set(_norm(name).split()), "_code": _norm(code).replace(" ", ""),
                "_sortkey": (int(ym.group(1)) if ym else 0) * 2 + (1 if sess.lower().startswith("july") else 0),
            })
    return rows

ROWS = _load()
VOCAB = sorted({w for r in ROWS for w in r["_words"] if len(w) > 3})

def _token_in_subject(tok, row):
    if tok in row["_words"]:
        return True
    if len(tok) >= 3 and any(w.startswith(tok) for w in row["_words"]):
        return True
    return False

def _fix_typo(tok):
    if len(tok) < 4 or tok in VOCAB:
        return tok
    m = difflib.get_close_matches(tok, VOCAB, n=1, cutoff=0.82)
    return m[0] if m else tok

def _parse(q):
    info = {"exam": set(), "year": None, "batch": None, "half": None, "course": None, "dept": None}
    rest = q
    if MID.search(q): info["exam"].add("Mid Sem")
    if END.search(q): info["exam"].add("End Sem")
    for key, pat in COURSES:
        if re.search(pat, q, re.I):
            info["course"] = key; rest = re.sub(pat, " ", rest, flags=re.I); break
    for pat, cs, dept in DEPTS:
        flags = 0 if cs else re.I
        if re.search(pat, q, flags):
            info["dept"] = dept; rest = re.sub(pat, " ", rest, flags=flags); break
    m = re.search(r"\b(20\d\d)\s*batch\b|\bbatch\s*(20\d\d)\b", q, re.I)
    if m:
        info["batch"] = m.group(1) or m.group(2); rest = rest.replace(info["batch"], " ")
    else:
        m = re.search(r"\b(20\d\d)\b", q)
        if m: info["year"] = m.group(1); rest = rest.replace(m.group(1), " ")
    if re.search(r"\b(jan|january|june)\b", q, re.I): info["half"] = "January-June"
    elif re.search(r"\b(july|dec|december)\b", q, re.I): info["half"] = "July-December"
    rest = re.sub(r"\b(jan|january|june|july|dec|december)\b", " ", rest, flags=re.I)
    rest = MID.sub(" ", END.sub(" ", rest))
    toks = []
    for t in _norm(rest).split():
        if t in STOP: continue
        toks.extend(ALIASES[t].split() if t in ALIASES else [_fix_typo(t)])
    info["subject"] = toks
    return info

def _match(row, toks):
    if not toks:
        return False
    if any(t == row["_code"] or (len(t) >= 5 and t in row["_code"]) for t in toks):
        return True
    return all(_token_in_subject(t, row) for t in toks)

def _describe(r):
    parts = [f"{r['exam']} paper", r["session"]]
    dept = r["dept"].title().replace(" And ", " and ").replace(" Of ", " of ") if r["dept"] else ""
    if dept.lower().startswith(r["course"].lower()): dept = dept[len(r["course"]):].strip()
    who = ", ".join(x for x in [r["course"], dept] if x)
    if r["batch"]: who += f" ({r['batch']} batch)"
    return " · ".join(p for p in parts + [who] if p)

def search_papers(q):
    """Returns None if q isn't a paper request; else {"message", "groups"}."""
    q = (q or "").strip()
    has_word = bool(PAPER_WORDS.search(q))
    info = _parse(q)
    has_exam = bool(info["exam"])
    toks = info["subject"]
    hits = [r for r in ROWS if _match(r, toks)] if toks else []
    if not (has_word or (has_exam and hits)):
        return None
    if not toks:
        return {"message": "Which subject do you need? Try something like “DBMS mid sem paper” or “operating system end sem 2025”.", "groups": []}
    if not hits:
        return {"message": f"I couldn't find papers for “{' '.join(toks)}”. Try the full subject name or the subject code.", "groups": []}

    def apply(rows, f):
        out = rows
        if f.get("exam"): out = [r for r in out if r["exam"] in info["exam"]]
        if f.get("year"): out = [r for r in out if info["year"] in r["session"]]
        if f.get("half"): out = [r for r in out if r["session"].startswith(info["half"])]
        if f.get("batch"): out = [r for r in out if r["batch"] == info["batch"]]
        if f.get("course"): out = [r for r in out if r["course"].lower().startswith(info["course"])]
        if f.get("dept"): out = [r for r in out if info["dept"] in r["dept"].upper()]
        return out

    active = {k: bool(info[k]) for k in ["exam", "year", "half", "batch", "course", "dept"]}
    res = apply(hits, active)
    note = ""
    if not res:  # relax everything except exam type
        res = apply(hits, {"exam": active["exam"]}) or hits
        note = " I had no exact match for all your filters, so I'm showing the closest papers."
    res.sort(key=lambda r: (-r["_sortkey"], r["exam"], r["dept"]))
    total = len(res)
    res = res[:MAX_RESULTS]

    groups = {}
    for r in res:
        g = groups.setdefault((r["title"], r["code"]), {"subject": r["title"], "code": r["code"], "papers": []})
        g["papers"].append({"label": _describe(r), "url": r["url"]})
    ordered = sorted(groups.values(), key=lambda g: (len(_norm(g["subject"]).split()), -len(g["papers"])))
    msg = f"Found {total} paper{'s' if total != 1 else ''}, newest first."
    if total > MAX_RESULTS:
        msg += f" Showing the latest {MAX_RESULTS}; add a year or branch to narrow it down."
    return {"message": msg + note, "groups": ordered}
