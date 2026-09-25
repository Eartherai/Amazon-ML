"""v2 (CL-006): adds digit deletion vs substitution split and consonant-skeleton address-token cover.

Classical stage-2 pair features for the saved first-stage (51-feature LightGBM) sidecar.

Targets the error patterns measured on the fixed 6k held set (EXP-044):
Indic-script names (phonetic transliteration + consonant skeleton), injected
noise words and legal suffixes (core-name comparison), token rarity (IDF from
the Source 1 corpus of the same split), house numbers with a dropped leading
digit, state/street abbreviation equivalence, domain/initials names, and
corroboration by the same Source 1's strong sibling candidates.

Only provided records are used. The small dictionaries below are hand-written
normalization tables (allowed by the organizer Q&A). No labels, IDs or row
order enter any feature; country is deliberately not a feature.
"""
from __future__ import annotations

import math
import re
import unicodedata
from functools import lru_cache

import numpy as np
from anyascii import anyascii
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

LEGAL = {
    "private", "pvt", "limited", "ltd", "llp", "llc", "inc", "incorporated", "corp", "corporation",
    "co", "company", "plc", "lp", "pllc", "pc", "pa", "sarl", "sas", "sa", "sasu", "eurl", "sci",
    "snc", "gmbh", "ag", "the", "and", "of", "pra", "li", "lt", "pte", "opc", "ms", "m", "s",
    "cie", "ste", "et", "de", "du", "des", "la", "le", "les", "dba",
}
NOISE = {
    "services", "service", "enterprises", "enterprise", "group", "solutions", "solution", "center",
    "centre", "industries", "industry", "trading", "traders", "holdings", "associates", "agency",
    "consultants", "consultancy", "systems", "technologies", "technology", "tech", "ventures",
    "partners", "com", "www", "net", "org", "in", "co", "shop", "store", "stores", "sons",
}
_TRANSLIT_LEGAL = re.compile(r"^(?:pr+a*[iy]?[bv]h?[ae]+t+[a]?|l[iy]m[iy]t+e*d+[a]?|priv[ae]t|limit[ae]d|pvt|ltd)$")

STATES = {
    # US
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california", "co": "colorado",
    "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas", "ky": "kentucky", "la": "louisiana",
    "me": "maine", "md": "maryland", "ma": "massachusetts", "mi": "michigan", "mn": "minnesota",
    "ms": "mississippi", "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "newhampshire", "nj": "newjersey", "nm": "newmexico", "ny": "newyork", "nc": "northcarolina",
    "nd": "northdakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhodeisland", "sc": "southcarolina", "sd": "southdakota", "tn": "tennessee", "tx": "texas",
    "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "westvirginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "districtofcolumbia",
}
IN_STATES = {
    "dl": "delhi", "mh": "maharashtra", "ka": "karnataka", "tn": "tamilnadu", "up": "uttarpradesh",
    "wb": "westbengal", "gj": "gujarat", "rj": "rajasthan", "mp": "madhyapradesh", "ap": "andhrapradesh",
    "ts": "telangana", "tg": "telangana", "kl": "kerala", "pb": "punjab", "hr": "haryana", "od": "odisha",
    "or": "odisha", "orissa": "odisha", "br": "bihar", "jh": "jharkhand", "cg": "chhattisgarh",
    "ct": "chhattisgarh", "as": "assam", "uk": "uttarakhand", "ut": "uttarakhand", "hp": "himachalpradesh",
    "ga": "goa", "jk": "jammukashmir", "ch": "chandigarh", "py": "puducherry", "pondicherry": "puducherry",
}
MULTIWORD = {
    "tamil nadu": "tamilnadu", "uttar pradesh": "uttarpradesh", "west bengal": "westbengal",
    "madhya pradesh": "madhyapradesh", "andhra pradesh": "andhrapradesh", "himachal pradesh": "himachalpradesh",
    "jammu and kashmir": "jammukashmir", "jammu kashmir": "jammukashmir", "new york": "newyork",
    "new jersey": "newjersey", "new mexico": "newmexico", "new hampshire": "newhampshire",
    "north carolina": "northcarolina", "south carolina": "southcarolina", "north dakota": "northdakota",
    "south dakota": "southdakota", "west virginia": "westvirginia", "rhode island": "rhodeisland",
    "district of columbia": "districtofcolumbia",
}
STATE_NAMES = set(STATES.values()) | set(IN_STATES.values())
CITY = {"calcutta": "kolkata", "bombay": "mumbai", "madras": "chennai", "bengaluru": "bangalore",
        "gurugram": "gurgaon", "poona": "pune", "trivandrum": "thiruvananthapuram", "baroda": "vadodara"}
STREET = {
    "st": "street", "str": "street", "ave": "avenue", "av": "avenue", "avn": "avenue", "rd": "road",
    "dr": "drive", "drv": "drive", "ct": "court", "cv": "cove", "pkwy": "parkway", "pky": "parkway",
    "blvd": "boulevard", "bd": "boulevard", "bld": "boulevard", "ln": "lane", "pl": "place",
    "cir": "circle", "hwy": "highway", "sq": "square", "ter": "terrace", "terr": "terrace", "trl": "trail",
    "fl": "floor", "flr": "floor", "apt": "apartment", "ste": "suite", "no": "number", "nos": "number",
    "bldg": "building", "nr": "near", "opp": "opposite", "r": "rue", "ch": "chemin", "imp": "impasse",
    "all": "allee", "fbg": "faubourg", "rte": "route", "pt": "point", "mt": "mount", "ft": "fort",
    "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest", "hts": "heights", "jn": "junction", "jct": "junction",
    "cres": "crescent", "sec": "sector", "ph": "phase", "extn": "extension", "ext": "extension",
}
_NONALNUM = re.compile(r"[^a-z0-9]+")
_DIGITS = re.compile(r"\d+")
_NUMGROUP = re.compile(r"\d+(?:\s*[-/]\s*\d+)*")


@lru_cache(maxsize=2_000_000)
def norm(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    # Unicode decimal digits (e.g. Devanagari) become ASCII before transliteration.
    text = "".join(str(unicodedata.decimal(c)) if c.isdigit() and not c.isascii() and unicodedata.decimal(c, None) is not None else c for c in text)
    return _NONALNUM.sub(" ", anyascii(text).lower()).strip()


def non_latin(raw: str) -> float:
    return float(any(ord(c) > 0x24F and c.isalpha() for c in raw or ""))


def uniq(tokens):
    seen, out = set(), []
    for tok in tokens:
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def core_tokens(normed: str) -> list[str]:
    toks = uniq(normed.split())
    legal_free = [t for t in toks if t not in LEGAL and not _TRANSLIT_LEGAL.match(t)]
    core = [t for t in legal_free if t not in NOISE]
    return core or legal_free or toks


_SK_DIGRAPH = [("ph", "f"), ("bh", "v"), ("th", "t"), ("dh", "d"), ("kh", "k"), ("gh", "g"),
               ("ch", "c"), ("sh", "s"), ("jh", "j"), ("ck", "k"), ("qu", "k")]
_SK_MAP = str.maketrans({"b": "v", "w": "v", "z": "j", "q": "k", "x": "k", "c": "k"})


@lru_cache(maxsize=2_000_000)
def skeleton_token(tok: str) -> str:
    if tok.isdigit():
        return tok
    for a, b in _SK_DIGRAPH:
        tok = tok.replace(a, b)
    tok = tok.translate(_SK_MAP)
    head, rest = tok[:1], re.sub(r"[aeiouy]", "", tok[1:])
    out = head + rest
    return re.sub(r"(.)\1+", r"\1", out)


NAME_TO_CODE = {v: k for k, v in STATES.items()}
NAME_TO_CODE.update({v: k for k, v in IN_STATES.items() if len(k) == 2})
NAME_TO_CODE.update({"orissa": "od", "odisha": "od", "pondicherry": "py"})
# Two-letter codes that are also everyday words (English/French) are not treated as states.
AMBIGUOUS_CODES = {"de", "la", "in", "me", "or", "hi", "id", "ok", "al", "co", "as", "oh", "ne", "pa", "ga", "ms", "ch", "no"}
CODES = (set(STATES) | {k for k in IN_STATES if len(k) == 2}) - AMBIGUOUS_CODES


def address_tokens(normed: str) -> list[str]:
    """Symmetric canonical alpha tokens: state names/codes -> st_<code>, street/city aliases."""
    text = " " + normed + " "
    for k, v in MULTIWORD.items():
        if " " + k + " " in text:
            text = text.replace(" " + k + " ", " " + v + " ")
    out = []
    for tok in text.split():
        if tok.isdigit():
            continue
        tok = CITY.get(tok, tok)
        if tok in NAME_TO_CODE:
            tok = "st_" + NAME_TO_CODE[tok]
        elif len(tok) == 2 and tok in CODES:
            tok = "st_" + tok
        else:
            tok = STREET.get(tok, tok)
        out.append(tok)
    return uniq(out)


def canonical_state_set(tokens):
    """Two-letter codes shared by US/India (e.g. 'ct', 'or') stay ambiguous but comparable."""
    return {t for t in tokens if t.startswith("st_")}


def ascii_raw(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(str(unicodedata.decimal(c)) if c.isdigit() and not c.isascii() and unicodedata.decimal(c, None) is not None else c for c in text)
    return anyascii(text).lower()


def numbers(raw_ascii: str) -> list[str]:
    """Digit tokens plus hyphen/slash-joined groups (\"4-03\" -> 403), leading zeros dropped."""
    nums = [n.lstrip("0") or "0" for n in _DIGITS.findall(raw_ascii)]
    joined = [re.sub(r"\D", "", g).lstrip("0") or "0" for g in _NUMGROUP.findall(raw_ascii) if not g.isdigit()]
    return uniq(nums + joined)


def num_relation(x: str, ys: list[str]) -> int:
    """0 none, 1 exact, 2 one digit inserted/deleted (benign noise), 3 one digit substituted, 4 containment."""
    if x in ys:
        return 1
    best = 0
    for y in ys:
        if len(x) >= 2 and len(y) >= 2 and abs(len(x) - len(y)) == 1 and Levenshtein.distance(x, y) == 1:
            return 2
        if len(x) == len(y) and len(x) >= 2 and Levenshtein.distance(x, y) == 1:
            best = 3
        elif len(x) >= 3 and len(y) >= 3 and (x in y or y in x) and best == 0:
            best = 4
    return best


def fuzzy_cover(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    hit = 0
    for t in a:
        if t in b:
            hit += 1
            continue
        best = max(JaroWinkler.normalized_similarity(t, u) for u in b)
        if best >= 0.9 and len(t) >= 3:
            hit += 1
    return hit / len(a)


def idf_stats(qt, tt, idf, default):
    if not qt or not tt:
        return 0.0, 0.0, 0.0, 0.0
    w = lambda t: idf.get(t, default)
    qs, ts = set(qt), set(tt)
    shared = qs & ts
    num = sum(w(t) ** 2 for t in shared)
    den = math.sqrt(sum(w(t) ** 2 for t in qs) * sum(w(t) ** 2 for t in ts)) or 1.0
    q_missing = [w(t) for t in qs - ts]
    t_missing = [w(t) for t in ts - qs]
    return (num / den, max((w(t) for t in shared), default=0.0),
            max(q_missing, default=0.0), max(t_missing, default=0.0))


NAMES = [
    "base", "base_logit", "rank", "gap_top", "n_strong", "n_mid", "sib_name", "sib_addr", "sib_dup",
    "q_nonlatin", "t_nonlatin", "t_s2", "q_addr_missing", "t_addr_missing",
    "core_eq", "core_jw", "core_tset", "core_tsort", "core_contain", "core_jacc", "core_q_only", "core_t_only",
    "core_fcover_q", "core_fcover_t", "first_core_eq",
    "name_idf_cos", "name_idf_shared_max", "name_idf_qmiss_max", "name_idf_tmiss_max",
    "sk_eq", "sk_jw", "sk_tset", "compact_jw", "initials_match", "t_domain",
    "addr_tset", "addr_jacc", "addr_idf_cos", "addr_idf_shared_max", "addr_idf_qmiss_max", "addr_idf_tmiss_max",
    "addr_fcover_q", "addr_fcover_t", "state_eq", "state_conflict",
    "num_exact", "num_fuzzy", "num_q_unmatched", "num_t_unmatched", "first_num_rel", "digits_jw",
    "num_del", "num_sub", "num_contain", "addr_sk_cover_q", "addr_sk_cover_t", "addr_sk_tset",
]


class Record:
    __slots__ = ("raw_name", "raw_addr", "n", "a", "core", "sk", "addr", "addr_sk", "nums", "nonlatin", "compact", "initials", "domain")

    def __init__(self, name: str, addr: str):
        self.raw_name, self.raw_addr = name or "", addr or ""
        self.n, self.a = norm(self.raw_name), norm(self.raw_addr)
        self.core = core_tokens(self.n)
        self.sk = [skeleton_token(t) for t in self.core]
        self.addr = address_tokens(self.a)
        self.nums = numbers(ascii_raw(self.raw_addr))
        self.addr_sk = uniq(skeleton_token(t) for t in self.a.split() if not t.isdigit() and len(t) >= 3)
        self.nonlatin = non_latin(self.raw_name)
        low = self.raw_name.lower()
        self.domain = float(bool(re.search(r"\.(com|net|org|in|co|biz|info|fr|us)\b", low) or "www" in low))
        toks = [t for t in self.n.split() if t not in {"www", "com", "net", "org", "in", "co", "biz", "info", "fr", "us"}]
        self.compact = "".join(toks)
        self.initials = "".join(t[0] for t in toks if t and t not in {"corp", "inc", "llc", "ltd", "llp", "pvt", "limited", "private", "the"})


def pair_vector(q: Record, t: Record, t_s2: bool, name_idf, name_default, addr_idf, addr_default) -> list[float]:
    cq, ct = " ".join(q.core), " ".join(t.core)
    qs, ts = set(q.core), set(t.core)
    inter = len(qs & ts)
    feats = [
        q.nonlatin, t.nonlatin, float(t_s2), float(not q.a), float(not t.a),
        float(bool(cq) and cq == ct),
        JaroWinkler.normalized_similarity(cq, ct) if cq and ct else 0.0,
        fuzz.token_set_ratio(cq, ct) / 100 if cq and ct else 0.0,
        fuzz.token_sort_ratio(cq, ct) / 100 if cq and ct else 0.0,
        inter / min(len(qs), len(ts)) if qs and ts else 0.0,
        inter / len(qs | ts) if qs or ts else 0.0,
        float(len(qs - ts)), float(len(ts - qs)),
        fuzzy_cover(q.core, t.core), fuzzy_cover(t.core, q.core),
        float(bool(q.core and t.core) and JaroWinkler.normalized_similarity(q.core[0], t.core[0]) >= 0.9),
    ]
    feats += idf_stats(q.n.split(), t.n.split(), name_idf, name_default)
    sq, st = " ".join(q.sk), " ".join(t.sk)
    feats += [
        float(bool(sq) and sq == st),
        JaroWinkler.normalized_similarity(sq, st) if sq and st else 0.0,
        fuzz.token_set_ratio(sq, st) / 100 if sq and st else 0.0,
        JaroWinkler.normalized_similarity(q.compact, t.compact) if q.compact and t.compact else 0.0,
        float(len(t.compact) >= 2 and len(q.initials) >= 2 and (q.initials.startswith(t.compact) or t.compact.startswith(q.initials))),
        t.domain,
    ]
    qa, ta = q.addr, t.addr
    if qa and ta:
        sqa, sta = set(qa), set(ta)
        ai = len(sqa & sta)
        qst, tst = canonical_state_set(qa), canonical_state_set(ta)
        feats += [fuzz.token_set_ratio(" ".join(qa), " ".join(ta)) / 100, ai / len(sqa | sta)]
        feats += idf_stats(qa, ta, addr_idf, addr_default)
        feats += [fuzzy_cover(qa, ta), fuzzy_cover(ta, qa),
                  float(bool(qst & tst)), float(bool(qst) and bool(tst) and not (qst & tst))]
    else:
        feats += [0.0] * 10
    qn, tn = q.nums, t.nums
    if qn and tn:
        rel_q = [num_relation(x, tn) for x in qn]
        rel_t = [num_relation(y, qn) for y in tn]
        first = num_relation(qn[0], [tn[0]])
        dq, dt = "".join(_DIGITS.findall(q.a)), "".join(_DIGITS.findall(t.a))
        feats += [sum(r == 1 for r in rel_q) / len(qn), sum(r >= 2 for r in rel_q) / len(qn),
                  float(sum(r == 0 for r in rel_q)), float(sum(r == 0 for r in rel_t)),
                  float(first if first else 5), JaroWinkler.normalized_similarity(dq, dt),
                  sum(r == 2 for r in rel_q) / len(qn), sum(r == 3 for r in rel_q) / len(qn), sum(r == 4 for r in rel_q) / len(qn)]
    else:
        feats += [0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if q.addr_sk and t.addr_sk:
        feats += [fuzzy_cover(q.addr_sk, t.addr_sk), fuzzy_cover(t.addr_sk, q.addr_sk),
                  fuzz.token_set_ratio(" ".join(q.addr_sk), " ".join(t.addr_sk)) / 100]
    else:
        feats += [0.0, 0.0, 0.0]
    return feats


def s1_context(group_rows, records_t, strong=0.9):
    """Within one Source 1: rank, gap, and corroboration by strong sibling candidates.

    group_rows: list of (target_id, base). records_t: target_id -> Record.
    Returns dict target_id -> list of context features (order matches NAMES[0:9]).
    """
    scored = sorted(group_rows, key=lambda x: -x[1])
    top = scored[0][1] if scored else 0.0
    strong_ids = [tid for tid, b in scored if b >= strong]
    n_mid = sum(1 for _, b in scored if b >= 0.5)
    out = {}
    for rank, (tid, b) in enumerate(scored, start=1):
        rec = records_t[tid]
        sib_name = sib_addr = sib_dup = 0.0
        for sid in strong_ids:
            if sid == tid:
                continue
            other = records_t[sid]
            if rec.core and other.core:
                sib_name = max(sib_name, JaroWinkler.normalized_similarity(" ".join(rec.core), " ".join(other.core)))
            if rec.addr and other.addr:
                sib_addr = max(sib_addr, fuzz.token_set_ratio(" ".join(rec.addr), " ".join(other.addr)) / 100)
            if rec.raw_name == other.raw_name and rec.raw_addr == other.raw_addr:
                sib_dup = 1.0
        clipped = min(max(b, 1e-6), 1 - 1e-6)
        out[tid] = [b, math.log(clipped / (1 - clipped)), float(rank), top - b, float(len(strong_ids)),
                    float(n_mid), sib_name, sib_addr, sib_dup]
    return out


def build_matrix(pairs_by_s1, s1_records, t_records, name_idf, name_default, addr_idf, addr_default):
    """pairs_by_s1: dict s1 -> list of (target_id, base). Returns (keys, X)."""
    keys, rows = [], []
    for s1, group in pairs_by_s1.items():
        ctx = s1_context(group, t_records)
        q = s1_records[s1]
        for tid, _ in group:
            t = t_records[tid]
            rows.append(ctx[tid] + pair_vector(q, t, tid.startswith("S2-"), name_idf, name_default, addr_idf, addr_default))
            keys.append((s1, tid))
    X = np.asarray(rows, dtype=np.float32)
    if X.shape[1] != len(NAMES):
        raise AssertionError(f"feature count {X.shape[1]} != {len(NAMES)}")
    return keys, X
