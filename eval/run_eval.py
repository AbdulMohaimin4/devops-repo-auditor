"""Self-evaluation harness (assignment §6): run the agent against the synthetic
fixtures and measure precision and recall.

    python eval/run_eval.py              # all fixtures
    python eval/run_eval.py repo_a repo_c

Each fixture is built into a temp git repo, audited with the real agent loop,
the output validated through the production validator, then scored against
eval/answer_key.json. Precision/recall use the same matching rule the answer
key documents. Results are printed and written to eval/results.json.

Needs an LLM API key (same .env as the agent). The scoring functions below are
pure and covered by tests/test_eval.py without any network.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from build_fixtures import FIXTURES, build  # noqa: E402

ANSWER_KEY = HERE / "answer_key.json"
RESULTS = HERE / "results.json"


def _categories(item):
    cat = item["category"]
    return [c.lower() for c in (cat if isinstance(cat, list) else [cat])]


def matches(expected, violation):
    """True if a reported ``violation`` satisfies an ``expected`` item: category
    is one of the accepted labels and, when a file token is given, it appears in
    the reported file, description, or evidence."""
    if str(violation.get("category", "")).lower() not in _categories(expected):
        return False
    token = expected.get("file")
    if not token:
        return True
    haystack = " ".join(str(violation.get(k, "") or "")
                        for k in ("file", "description", "evidence")).lower()
    return token.lower() in haystack


def _max_matching(expected, reported):
    """Maximum bipartite matching between expected items and reported
    violations (Kuhn's algorithm). Order-independent, so a wildcard expected
    item (file: null, matches any report of its category) never steals a report
    a more specific item needed. Returns {expected index: reported index}."""
    adj = {i: [j for j in range(len(reported)) if matches(item, reported[j])]
           for i, item in enumerate(expected)}
    match_r = {}  # reported index -> expected index

    def augment(i, seen):
        for j in adj[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in match_r or augment(match_r[j], seen):
                match_r[j] = i
                return True
        return False

    for i in range(len(expected)):
        augment(i, set())
    return {i: j for j, i in match_r.items()}


def score(reported, expected):
    """Score one fixture via maximum matching: each expected item pairs with at
    most one reported violation and vice versa. Unmatched expected items are
    misses; unmatched reports are false positives. Returns tp, fp, fn,
    precision, recall and the unmatched (missed / spurious) details."""
    matched = _max_matching(expected, reported)
    tp = len(matched)
    matched_reports = set(matched.values())
    missed = [expected[i].get("note") or expected[i].get("category")
              for i in range(len(expected)) if i not in matched]
    spurious = [
        f"{reported[j].get('category')}: {reported[j].get('description', '')[:80]}"
        for j in range(len(reported)) if j not in matched_reports
    ]
    fp = len(spurious)
    fn = len(missed)
    precision = tp / len(reported) if reported else 1.0
    recall = tp / len(expected) if expected else 1.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 3), "recall": round(recall, 3),
        "missed": missed, "spurious": spurious,
        "reported": len(reported), "expected": len(expected),
    }


def _audit_fixture(name):
    from agent_loop import run_audit  # imported here so scoring stays importable
    from validation import validate_violations
    with tempfile.TemporaryDirectory(prefix=f"eval-{name}-") as tmp:
        build(name, tmp)
        trace = HERE / "traces" / f"{name}.jsonl"
        trace.parent.mkdir(exist_ok=True)
        raw = run_audit(tmp, trace_path=str(trace))
        violations, _ = validate_violations(raw)
        return violations


def main(argv=None):
    from dotenv import load_dotenv
    load_dotenv(HERE.parent / ".env")

    names = argv or list(FIXTURES)
    key = json.loads(ANSWER_KEY.read_text())
    results = {}

    print(f"{'fixture':<10}{'reported':>9}{'TP':>4}{'FP':>4}{'FN':>4}"
          f"{'precision':>11}{'recall':>8}")
    print("-" * 50)
    tot_tp = tot_fp = tot_fn = 0
    for name in names:
        violations = _audit_fixture(name)
        s = score(violations, key[name])
        results[name] = {**s, "violations": violations}
        tot_tp += s["tp"]; tot_fp += s["fp"]; tot_fn += s["fn"]
        print(f"{name:<10}{s['reported']:>9}{s['tp']:>4}{s['fp']:>4}{s['fn']:>4}"
              f"{s['precision']:>11}{s['recall']:>8}")
        for m in s["missed"]:
            print(f"           MISS: {m}")
        for sp in s["spurious"]:
            print(f"           FALSE+: {sp}")

    micro_p = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) else 1.0
    micro_r = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) else 1.0
    print("-" * 50)
    print(f"{'OVERALL':<10}{'':>9}{tot_tp:>4}{tot_fp:>4}{tot_fn:>4}"
          f"{round(micro_p, 3):>11}{round(micro_r, 3):>8}")

    results["_overall"] = {"precision": round(micro_p, 3), "recall": round(micro_r, 3)}
    RESULTS.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or None))
