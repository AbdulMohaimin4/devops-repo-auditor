"""Tests for output validation and secret redaction (Milestone 4)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import validation  # noqa: E402
from validation import FIELDS, redact, validate_violations  # noqa: E402


def _valid():
    return {
        "rule_id": "SEC-001",
        "category": "security",
        "severity": "critical",
        "file": "config.py",
        "line": 17,
        "description": "hardcoded credential",
        "evidence": "literal assigned in source",
        "recommendation": "use env var",
    }


def test_valid_passes_through():
    clean, dropped = validate_violations([_valid()])
    assert dropped == 0
    assert clean[0] == _valid()
    assert set(clean[0]) == set(FIELDS)


def test_all_eight_fields_always_present():
    minimal = {"rule_id": "HYG-001", "description": "no readme"}
    clean, _ = validate_violations([minimal])
    assert set(clean[0]) == set(FIELDS)


def test_invalid_severity_falls_back_to_default():
    v = _valid()
    v["severity"] = "SUPER-BAD"
    clean, _ = validate_violations([v])
    assert clean[0]["severity"] == "critical"  # SEC-001 default


def test_severity_case_normalized():
    v = _valid()
    v["severity"] = "Critical"
    clean, _ = validate_violations([v])
    assert clean[0]["severity"] == "critical"


def test_nullish_file_and_line_become_none():
    v = _valid()
    v["file"] = "null"
    v["line"] = ""
    clean, _ = validate_violations([v])
    assert clean[0]["file"] is None
    assert clean[0]["line"] is None


def test_line_string_digit_coerced_and_zero_nulled():
    v = _valid()
    v["line"] = "42"
    assert validate_violations([v])[0][0]["line"] == 42
    v["line"] = 0
    assert validate_violations([v])[0][0]["line"] is None


def test_category_derived_from_rule_id_when_missing():
    v = _valid()
    del v["category"]
    clean, _ = validate_violations([v])
    assert clean[0]["category"] == "security"


def test_rule_id_derived_from_category_when_missing():
    v = _valid()
    del v["rule_id"]
    v["category"] = "security"
    clean, _ = validate_violations([v])
    assert clean[0]["rule_id"] == "SEC-000"


def test_rule_id_with_trailing_junk_is_normalized():
    v = _valid()
    v["rule_id"] = "CI-002,severity:high"  # model appended junk (seen in a real run)
    clean, _ = validate_violations([v])
    assert clean[0]["rule_id"] == "CI-002"


def test_dropped_when_no_rule_id_and_unknown_category():
    clean, dropped = validate_violations([{"category": "astrology"}])
    assert clean == []
    assert dropped == 1


def test_dropped_when_no_description_or_evidence():
    clean, dropped = validate_violations([{"rule_id": "HYG-001"}])
    assert dropped == 1


def test_non_list_input_is_handled():
    clean, dropped = validate_violations("not a list")
    assert clean == []
    assert dropped == 1


def test_non_dict_entries_dropped():
    clean, dropped = validate_violations([_valid(), "garbage", 42])
    assert len(clean) == 1
    assert dropped == 2


def test_redact_known_credential_formats():
    assert "AKIA" not in redact("key AKIAIOSFODNN7EXAMPLE here")
    assert validation.REDACTED in redact("token ghp_" + "a" * 36)
    assert validation.REDACTED in redact("google AIza" + "b" * 35)


def test_redact_secret_assignment_keeps_key_name():
    out = redact('password = "hunter2secret"')
    assert "hunter2secret" not in out
    assert "password" in out
    assert validation.REDACTED in out


def test_evidence_is_redacted_in_pipeline():
    v = _valid()
    v["evidence"] = "AWS key AKIAIOSFODNN7EXAMPLE committed"
    clean, _ = validate_violations([v])
    assert "AKIAIOSFODNN7EXAMPLE" not in clean[0]["evidence"]
