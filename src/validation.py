"""Output validation and secret redaction.

The report_findings tool schema shapes the model's output, but models still
occasionally omit a field, invent a severity, or echo a secret. This module is
the cheap second net: it enforces the exact 8-field violation contract, repairs
what it safely can, drops what it cannot, and redacts secret-looking values so
they never reach the report.

A malformed report can zero the detection score if the grader's harness cannot
parse it -- so validation runs before anything is written.
"""

import re

from rules import RULE_CATALOG, SEVERITIES, default_severity

# the exact, ordered contract every emitted violation must satisfy
FIELDS = (
    "rule_id",
    "category",
    "severity",
    "file",
    "line",
    "description",
    "evidence",
    "recommendation",
)

# values that mean "not applicable" and must become JSON null
_NULLISH = {"", "null", "none", "n/a", "na", "-"}

# a well-formed rule_id, e.g. CI-002; models sometimes append junk ("CI-002,...")
_RULE_ID_RE = re.compile(r"[A-Za-z]{2,4}-\d{1,4}")

# prefix -> category label, and the reverse, from the catalog
_PREFIX_TO_CATEGORY = {p: cat for p, (cat, _) in RULE_CATALOG.items()}
_CATEGORY_TO_PREFIX = {cat.lower(): p for p, cat in _PREFIX_TO_CATEGORY.items()}

# credential formats + secret-looking assignments; matches are replaced wholesale
_SECRET_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),                     # AWS access key id
    re.compile(r"AIza[0-9A-Za-z_\-]{35}"),               # Google API key
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),           # GitHub tokens
    re.compile(r"github_pat_[A-Za-z0-9_]{22,}"),         # GitHub fine-grained PAT
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),         # Slack tokens
    re.compile(r"sk-[A-Za-z0-9]{20,}"),                  # OpenAI-style keys
    re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),  # JWT
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),   # PEM blocks
]
# e.g.  password = "hunter2"  |  api_key: 'abc123'  -> redact the value only
_SECRET_ASSIGNMENT = re.compile(
    r"""(?i)\b(password|passwd|pwd|secret|token|api[_-]?key|apikey|
        access[_-]?key|secret[_-]?key|client[_-]?secret|auth[_-]?token)\b
        (\s*[:=]\s*)
        (["']?)([^"'\s,;]{4,})(\3)""",
    re.VERBOSE,
)

REDACTED = "[REDACTED]"


def redact(text):
    """Replace secret-looking substrings with a placeholder. Describing and
    locating a secret is the evidence; the value itself must never appear."""
    if not isinstance(text, str) or not text:
        return text
    out = text
    for pat in _SECRET_PATTERNS:
        out = pat.sub(REDACTED, out)
    out = _SECRET_ASSIGNMENT.sub(lambda m: m.group(1) + m.group(2) + m.group(3)
                                 + REDACTED + m.group(5), out)
    return out


def _as_str(value):
    if value is None:
        return ""
    return str(value).strip()


def _nullable_str(value):
    if value is None:
        return None
    s = str(value).strip()
    if s.lower() in _NULLISH:
        return None
    return s


def _nullable_int(value):
    if isinstance(value, bool):  # bool is an int subclass; reject it
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str):
        s = value.strip()
        if s.isdigit():
            n = int(s)
            return n if n > 0 else None
    return None


def _validate_one(item):
    """Return a clean 8-field violation dict, or None to drop it."""
    if not isinstance(item, dict):
        return None

    raw_id = item.get("rule_id")
    match = _RULE_ID_RE.search(raw_id) if isinstance(raw_id, str) else None
    if match:
        rule_id = match.group(0).upper()
    else:
        # no usable id: derive one from the stated category before giving up
        prefix = _CATEGORY_TO_PREFIX.get(_as_str(item.get("category")).lower())
        if not prefix:
            return None
        rule_id = f"{prefix}-000"

    # Category is the catalog's controlled label for the rule's prefix, so the
    # field is always one of the known 9 (models otherwise vary it freely:
    # "hygiene" vs "repository hygiene"). Only an unknown prefix falls back to
    # the model's stated category.
    prefix = rule_id.split("-", 1)[0]
    if prefix in _PREFIX_TO_CATEGORY:
        category = _PREFIX_TO_CATEGORY[prefix]
    else:
        stated = item.get("category")
        category = stated.strip() if isinstance(stated, str) and stated.strip() else "uncategorized"

    severity = item.get("severity")
    severity = severity.strip().lower() if isinstance(severity, str) else None
    if severity not in SEVERITIES:
        severity = default_severity(rule_id) or "medium"

    description = redact(_as_str(item.get("description")))
    evidence = redact(_as_str(item.get("evidence")))
    recommendation = redact(_as_str(item.get("recommendation")))

    # a finding with no description and no evidence is not defensible
    if not description and not evidence:
        return None

    return {
        "rule_id": rule_id,
        "category": category,
        "severity": severity,
        "file": _nullable_str(item.get("file")),
        "line": _nullable_int(item.get("line")),
        "description": description,
        "evidence": evidence,
        "recommendation": recommendation,
    }


def validate_violations(raw):
    """Validate/repair a list of raw model violations.

    Returns ``(clean, dropped)`` where clean is a list of contract-compliant
    dicts and dropped is the count of entries that could not be salvaged.
    """
    if not isinstance(raw, list):
        return [], 1

    clean = []
    dropped = 0
    for item in raw:
        v = _validate_one(item)
        if v is None:
            dropped += 1
        else:
            clean.append(v)
    return clean, dropped
