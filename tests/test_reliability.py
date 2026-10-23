"""Tests for Milestone 5 reliability: LLM retry/backoff and clone failure."""

import sys
from pathlib import Path

import openai
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import agent  # noqa: E402
import agent_loop  # noqa: E402


def _rate_limit_error():
    # Bypass the constructor (it wants an httpx response) -- we only need an
    # instance of a retryable type for the except clause to catch.
    return openai.RateLimitError.__new__(openai.RateLimitError)


def test_retry_succeeds_after_transient_errors(monkeypatch):
    monkeypatch.setattr(agent_loop.time, "sleep", lambda *_: None)  # no real waiting
    calls = {"n": 0}

    def flaky(**_kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _rate_limit_error()
        return "ok"

    client = type("C", (), {})()
    client.chat = type("Ch", (), {})()
    client.chat.completions = type("Co", (), {"create": staticmethod(flaky)})()

    assert agent_loop._create_with_retry(client) == "ok"
    assert calls["n"] == 3


def test_retry_gives_up_and_raises(monkeypatch):
    monkeypatch.setattr(agent_loop.time, "sleep", lambda *_: None)

    def always_fail(**_kwargs):
        raise _rate_limit_error()

    client = type("C", (), {})()
    client.chat = type("Ch", (), {})()
    client.chat.completions = type("Co", (), {"create": staticmethod(always_fail)})()

    with pytest.raises(openai.RateLimitError):
        agent_loop._create_with_retry(client)


def test_clone_failure_exits_nonzero_without_writing_report(monkeypatch, tmp_path):
    out = tmp_path / "report.json"

    def boom(repo, dest):
        raise agent.CloneError("repository not found")

    monkeypatch.setattr(agent, "clone_repo", boom)
    rc = agent.main(["--repo", "https://github.com/x/y", "--output", str(out)])
    assert rc == 1
    assert not out.exists()  # no half report on failure


def test_llm_failure_exits_nonzero_without_writing_report(monkeypatch, tmp_path):
    out = tmp_path / "report.json"
    monkeypatch.setattr(agent, "clone_repo", lambda repo, dest: None)

    def boom(_root):
        raise RuntimeError("no API key")

    monkeypatch.setattr(agent, "run_audit", boom)
    rc = agent.main(["--repo", "https://github.com/x/y", "--output", str(out)])
    assert rc == 1
    assert not out.exists()
