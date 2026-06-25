#!/usr/bin/env python3
r"""Answer-correctness scorer for the lettercount+arith trap sweep.

Reads a niodoo transcript (.txt, --stdout-profile telemetry), finds the
`=== TURN N OUTPUT ===` blocks, and extracts the model's committed final
answer per turn, then compares to the gold answer.

Why this exists (latch-0006 pivot): the geometric "latch" (min target
distance) proxy is noisy on a single seed. We score the thing we actually
care about — did the model give the RIGHT NUMBER — off vs on basin-coherence.

Extraction rule (documented, deterministic):
  - region = text between this turn's marker and the next marker (or EOF),
    truncated at the first '[REQUEST: LOCK] engaged' / 'Clean shutdown' /
    'Process terminated' (post-lock rambling is not the answer).
  - letter-count turns: prefer  r'appears\s+(\d+)\s+times'  (the model's
    own count statement); else r'(\d+)\s+times'; else first standalone int.
  - arithmetic turns: prefer the last  r'=\s*(\d+)'  (equation result);
    else the first standalone integer line.
  - correct iff extracted == gold.

Nothing is hidden: the script prints the extracted answer AND the first
~220 chars of the raw region for every cell so a human can adjudicate.
"""
import re, sys, json, os

# trap order in harness/traps/sweep_lettercount_arith.txt
TRAPS = [
    {"i": 0, "name": "mississippi-s", "kind": "letter", "gold": 4},
    {"i": 1, "name": "strawberry-r",  "kind": "letter", "gold": 3},
    {"i": 2, "name": "17x24",         "kind": "arith",  "gold": 408},
    {"i": 3, "name": "13x17",         "kind": "arith",  "gold": 221},
]

MARK = re.compile(r"=== TURN (\d+) OUTPUT ===")
POST_LOCK = re.compile(r"\[REQUEST: LOCK\] engaged|Clean shutdown|Process terminated|Connection closed")


def regions(text):
    """Return {turn_index: raw_region_text}."""
    out = {}
    hits = [(m.start(), int(m.group(1))) for m in MARK.finditer(text)]
    for k, (pos, ti) in enumerate(hits):
        start = pos + text[pos:].index("\n") + 1 if "\n" in text[pos:] else len(text)
        end = hits[k + 1][0] if k + 1 < len(hits) else len(text)
        out[ti] = text[start:end]
    return out


def truncate_region(region):
    m = POST_LOCK.search(region)
    return region[: m.start()] if m else region


def first_standalone_int(region):
    for line in region.splitlines():
        s = line.strip()
        if re.fullmatch(r"\d+", s):
            return int(s)
    m = re.search(r"\b(\d+)\b", region)
    return int(m.group(1)) if m else None


def last_match(region, patterns):
    """Return (value, pos) for the LAST (greatest-position) match across the
    given answer-context patterns (the model self-corrects, so last wins)."""
    best = (None, -1)
    for pat in patterns:
        for m in re.finditer(pat, region, re.I):
            if m.start() > best[1]:
                best = (int(m.group(1)), m.start())
    return best


def last_standalone_int(region):
    best = (None, -1)
    pos = 0
    for line in region.splitlines():
        s = line.strip()
        if re.fullmatch(r"\d+", s):
            best = (int(s), pos)
        pos += len(line) + 1
    return best


# answer-context phrasings the model uses to STATE a count / result
LETTER_PATS = [
    r"appears\D{0,12}(\d+)\s*times",
    r"there (?:are|is)\D{0,6}(\d+)\s*(?:occurrence|time)",
    r"(\d+)\s*(?:occurrences?|times)",
    r"(?:correct answer is|the answer is|final answer)\D{0,8}(\d+)",
]
ARITH_PATS = [
    r"(?:final answer|correct answer is|the answer is)\D{0,8}(\d+)",
]


def extract(trap, region):
    r = truncate_region(region)
    if trap["kind"] == "letter":
        v, _ = last_match(r, LETTER_PATS)
        if v is not None:
            return v, "count-statement"
        v, _ = last_standalone_int(r)
        if v is not None:
            return v, "last-standalone"
        return first_standalone_int(r), "leading-int"
    else:  # arith
        # 1) explicit "final/correct answer is N" (last wins on self-correction)
        v, _ = last_match(r, ARITH_PATS)
        if v is not None:
            return v, "answer-statement"
        # 2) else the LEADING committed result (model leads with its answer;
        #    trailing "17x4 = 68" re-derivation steps are NOT the answer)
        v = first_standalone_int(r)
        return v, "leading-result"


def score_file(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    regs = regions(text)
    rows = []
    for trap in TRAPS:
        region = regs.get(trap["i"], "")
        ans, how = extract(trap, region)
        ok = (ans == trap["gold"])
        snippet = " ".join(truncate_region(region).split())[:220]
        rows.append({
            "trap": trap["name"], "kind": trap["kind"], "gold": trap["gold"],
            "answer": ans, "rule": how, "correct": ok, "snippet": snippet,
        })
    return rows


if __name__ == "__main__":
    files = sys.argv[1:]
    grand = []
    for f in files:
        label = os.path.basename(f).replace("ans_", "").replace(".txt", "")
        rows = score_file(f)
        n_ok = sum(r["correct"] for r in rows)
        print(f"\n===== {label}  ({n_ok}/4 correct) =====")
        for r in rows:
            flag = "OK " if r["correct"] else "XX "
            print(f"  {flag}{r['trap']:<14} gold={r['gold']:<4} got={str(r['answer']):<6} [{r['rule']}]")
            print(f"       raw: {r['snippet']}")
            grand.append({"run": label, **r})
    print("\n===== JSON =====")
    print(json.dumps(grand))
