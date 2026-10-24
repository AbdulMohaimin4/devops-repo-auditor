"""Agent-loop tests with a mocked model -- no API key, no network.

These prove the loop dispatches tools, feeds results back, and terminates on
report_findings (and on the iteration cap). That is the 'agent behavior' the
rubric names, testable without the LLM.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import agent_loop  # noqa: E402


def _tool_call(call_id, name, args):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def _response(content=None, tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class ScriptedClient:
    """Returns pre-scripted responses and records the requests it received."""

    def __init__(self, scripted):
        self._scripted = list(scripted)
        self.calls = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        # snapshot messages (loop mutates the live list in place between calls)
        snapshot = dict(kwargs)
        snapshot["messages"] = list(kwargs["messages"])
        self.calls.append(snapshot)
        return self._scripted.pop(0)


@pytest.fixture
def clone_root(tmp_path):
    (tmp_path / "README.md").write_text("# demo\n")
    (tmp_path / "app.py").write_text("x = 1\n")
    return tmp_path


def _patch_client(monkeypatch, client):
    monkeypatch.setattr(agent_loop, "_make_client", lambda: client)


def test_loop_dispatches_then_reports(monkeypatch, clone_root, tmp_path):
    violation = {
        "rule_id": "HYG-001",
        "category": "repository hygiene",
        "severity": "low",
        "file": "app.py",
        "line": None,
        "description": "demo",
        "evidence": "observed",
        "recommendation": "fix it",
    }
    client = ScriptedClient([
        _response(tool_calls=[_tool_call("c1", "list_files", {})]),
        _response(tool_calls=[_tool_call("c2", "read_file", {"path": "app.py"})]),
        _response(tool_calls=[_tool_call("c3", "report_findings",
                                         {"violations": [violation]})]),
    ])
    _patch_client(monkeypatch, client)

    trace = tmp_path / "trace.jsonl"
    result = agent_loop.run_audit(str(clone_root), trace_path=str(trace))

    assert result == [violation]
    # three model calls were made (list_files, read_file, report_findings)
    assert len(client.calls) == 3
    # the real tools ran: by the final (report_findings) call, both
    # list_files and read_file results had been fed back
    final_msgs = client.calls[-1]["messages"]
    tool_msgs = [m for m in final_msgs if m.get("role") == "tool"]
    assert len(tool_msgs) == 2
    # trace recorded the terminal step
    lines = [json.loads(x) for x in trace.read_text().splitlines()]
    assert any(e.get("type") == "report_findings" for e in lines)


def test_loop_nudges_on_empty_then_reports(monkeypatch, clone_root):
    client = ScriptedClient([
        _response(content="let me think..."),  # no tool call -> nudge
        _response(tool_calls=[_tool_call("c1", "report_findings",
                                         {"violations": []})]),
    ])
    _patch_client(monkeypatch, client)

    result = agent_loop.run_audit(str(clone_root), trace_path=None)
    assert result == []
    # a nudge (user message) was appended before the second call
    assert client.calls[1]["messages"][-1]["role"] == "user"


def test_loop_forces_final_call_at_cap(monkeypatch, clone_root):
    # Model keeps calling list_files forever; loop must force a final call.
    scripted = [
        _response(tool_calls=[_tool_call(f"c{i}", "list_files", {})])
        for i in range(agent_loop.MAX_ITERATIONS - 1)
    ]
    scripted.append(
        _response(tool_calls=[_tool_call("final", "report_findings",
                                         {"violations": []})])
    )
    client = ScriptedClient(scripted)
    _patch_client(monkeypatch, client)

    result = agent_loop.run_audit(str(clone_root), trace_path=None)
    assert result == []
    assert len(client.calls) == agent_loop.MAX_ITERATIONS
    # the final call forced report_findings via tool_choice
    final_choice = client.calls[-1]["tool_choice"]
    assert final_choice["function"]["name"] == "report_findings"


def test_unknown_tool_returns_error_observation(clone_root):
    tools = agent_loop.RepoTools(str(clone_root))
    assert "error" in agent_loop._dispatch(tools, "nope", {})


def test_dispatch_routes_all_inspection_tools(clone_root):
    # every advertised inspection tool must be dispatchable (no typos/omissions)
    tools = agent_loop.RepoTools(str(clone_root))
    for name in agent_loop.INSPECTION_TOOLS:
        result = agent_loop._dispatch(tools, name, {})
        assert isinstance(result, dict), name
