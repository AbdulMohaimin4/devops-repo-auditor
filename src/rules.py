"""Rule catalog: stable rule_ids per DevOps inspection area, with default
severities.

Kept as data (not prose) so it serves two jobs:
  1. rendered into the system prompt as *hints* the model can anchor to, and
  2. reused by output validation (valid categories / severities / id prefixes).

These are hints, not a checklist. The model chooses what to pursue based on
what it observes, and may mint a new id under an area's prefix (e.g. SEC-009)
for a genuine issue that no listed rule matches.
"""

SEVERITIES = ("critical", "high", "medium", "low")

# area prefix -> (category label, {rule_id: (description, default_severity)})
RULE_CATALOG = {
    "HYG": (
        "repository hygiene",
        {
            "HYG-001": ("Missing README", "medium"),
            "HYG-002": ("Missing or unsuitable .gitignore", "low"),
            "HYG-003": ("Tracked temporary/generated/build artifacts", "low"),
            "HYG-004": ("Tracked local/editor files (e.g. .vscode, .idea)", "low"),
        },
    ),
    "DEP": (
        "dependency management",
        {
            "DEP-001": ("No dependency manifest at all", "high"),
            "DEP-002": ("Missing lockfile where the ecosystem expects one", "medium"),
            "DEP-003": ("Unpinned or weak version constraints", "low"),
            "DEP-004": ("Inconsistent or duplicated dependency specifications", "medium"),
        },
    ),
    "QUA": (
        "code quality",
        {
            "QUA-001": ("No linter/formatter configuration", "low"),
            "QUA-002": ("Obvious maintainability problem", "low"),
        },
    ),
    "TST": (
        "testing",
        {
            "TST-001": ("No tests present", "medium"),
            "TST-002": ("Tests exist but are not run in CI", "medium"),
            "TST-003": ("Failing or clearly broken tests (static signal)", "high"),
        },
    ),
    "CI": (
        "ci",
        {
            "CI-001": ("No CI configured", "medium"),
            "CI-002": ("CI does not run tests / ineffective gate", "high"),
            "CI-003": ("Incorrect or missing triggers (e.g. no PR trigger)", "medium"),
            "CI-004": ("Validation skipped (continue-on-error, disabled steps)", "medium"),
        },
    ),
    "SEC": (
        "security",
        {
            "SEC-001": ("Hardcoded credential or secret in source", "critical"),
            "SEC-002": ("Unsafe workflow permissions (e.g. write-all)", "high"),
            "SEC-003": ("Dangerous pull_request_target / untrusted checkout", "high"),
            "SEC-004": ("Unpinned third-party action (mutable tag/branch ref)", "low"),
        },
    ),
    "CFG": (
        "configuration",
        {
            "CFG-001": ("Machine-specific or absolute paths", "medium"),
            "CFG-002": ("Committed local/environment settings", "low"),
            "CFG-003": ("Poor environment separation / secrets in config", "high"),
        },
    ),
    "COL": (
        "collaboration",
        {
            "COL-001": ("Unsafe PR automation (auto-merge without checks)", "high"),
            "COL-002": ("Missing review guidance where relevant (CONTRIBUTING/CODEOWNERS)", "low"),
        },
    ),
    "BLD": (
        "build and release",
        {
            "BLD-001": ("Missing build validation", "medium"),
            "BLD-002": ("Risky release or deployment automation", "high"),
        },
    ),
}

# flat lookups for validation / reuse
CATEGORIES = {label for label, _ in RULE_CATALOG.values()}
PREFIXES = set(RULE_CATALOG.keys())
KNOWN_RULE_IDS = {
    rid for _, rules in RULE_CATALOG.values() for rid in rules
}


def default_severity(rule_id):
    """Return the catalog default severity for a rule_id, or None if unknown."""
    prefix = rule_id.split("-", 1)[0]
    entry = RULE_CATALOG.get(prefix)
    if not entry:
        return None
    return entry[1].get(rule_id, (None, None))[1]


def render_catalog():
    """Render the catalog as a compact text block for the system prompt."""
    lines = []
    for prefix, (category, rules) in RULE_CATALOG.items():
        lines.append(f"{prefix} ({category}):")
        for rid, (desc, sev) in rules.items():
            lines.append(f"  {rid} [{sev}] {desc}")
    return "\n".join(lines)
