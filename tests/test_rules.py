"""Tests for the rule catalog and its integration into the system prompt."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import rules  # noqa: E402


def test_nine_inspection_areas():
    expected = {"HYG", "DEP", "QUA", "TST", "CI", "SEC", "CFG", "COL", "BLD"}
    assert rules.PREFIXES == expected


def test_catalog_integrity():
    for prefix, (category, rule_map) in rules.RULE_CATALOG.items():
        assert category  # non-empty label
        for rid, (desc, sev) in rule_map.items():
            assert rid.split("-", 1)[0] == prefix, f"{rid} prefix mismatch"
            assert sev in rules.SEVERITIES, f"{rid} bad severity {sev}"
            assert desc


def test_default_severity_lookup():
    assert rules.default_severity("SEC-001") == "critical"
    assert rules.default_severity("HYG-001") == "medium"
    assert rules.default_severity("ZZZ-999") is None


def test_render_catalog_contains_ids_and_severities():
    text = rules.render_catalog()
    assert "SEC-001" in text
    assert "[critical]" in text
    assert "repository hygiene" in text


def test_system_prompt_embeds_catalog():
    # importing agent_loop requires openai to be installed; it is a dep
    import agent_loop  # noqa: E402
    assert "SEC-001" in agent_loop.SYSTEM_PROMPT
    assert "{catalog}" not in agent_loop.SYSTEM_PROMPT  # placeholder replaced
    assert "not a checklist" in agent_loop.SYSTEM_PROMPT.lower()
