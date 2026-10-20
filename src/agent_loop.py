"""The agent loop: an explicit observe -> reason -> tool-call -> interpret
cycle driven by an LLM's native tool-calling.

No third party agent framework is used, instead there is custom control and 
the model uses a set of inspection tools, dispatches whatever it calls, feeds 
results back, and stops when it calls the terminal ``report_findings`` tool 
(or when the iteration cap is hit and it forces a final call).

The agent is provider-agnostic and works with any OpenAI-compatible endpoint. 
It uses Google Gemini's OpenAI-compatible API by default, but .env is configurable.
"""

import json
import os
import sys
import time

from openai import OpenAI

from tools import RepoTools

# --- provider config (overridable via env) ------------------------------
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

DEFAULT_MODEL = "gemini-flash-lite-latest"

MAX_ITERATIONS = 30       # bound cost and guarantee termination
TEMPERATURE = 0.0         # deterministic for analysis
TRACE_RESULT_CHARS = 2000  # trim tool results in the trace log

SYSTEM_PROMPT = """\
You are a DevOps repository auditor. You inspect a cloned GitHub repository and
identify genuine DevOps best-practice violations, backed by evidence.

You work by calling the provided tools to investigate. Behave like a human
investigator: first map the repository (list_files), then read the files that
look relevant, use search_code to confirm suspicions, and inspect git_log for
history signals. Let what you observe drive what you look at next -- do not run
a fixed checklist.

Scope: software engineering and DevOps only (repo hygiene, dependencies, code
quality, testing, CI/GitHub Actions, security, configuration, collaboration,
build/release). Ignore ML/MLOps concerns.

Evidence discipline:
- Only report a violation you can back with specific evidence from a tool.
- An absent optional file is NOT a defect when an equivalent exists
  (e.g. pyproject.toml instead of requirements.txt).
- Never infer remote GitHub settings (branch protection, required reviews) from
  files in the clone -- you cannot see them.
- Never reproduce a secret value; describe and locate it instead.

When your investigation is complete, call report_findings exactly once with all
violations you can defend (an empty list if the repository is healthy)."""


# --- tool schemas (OpenAI function-calling format) ---------------------------
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all files in the repository as root-relative "
            "paths (the .git directory is skipped). Call this first to map the repo.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a repository file's text content (capped; "
            "truncation is flagged). Use for configs, source, workflows, docs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Root-relative path, e.g. 'src/app.py'.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search all text files for a regular-expression "
            "pattern. Returns path:line:snippet matches. Use to confirm suspicions "
            "(e.g. hardcoded secrets, debug flags) across the repo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "A Python regular expression.",
                    }
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_log",
            "description": "Return recent commit history (hash, author, date, "
            "subject). Use for signals like committed secrets or churn.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "report_findings",
            "description": "Submit the final audit result and end the "
            "investigation. Call exactly once when done.",
            "parameters": {
                "type": "object",
                "properties": {
                    "violations": {
                        "type": "array",
                        "description": "Every defensible violation; empty if healthy.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "rule_id": {"type": "string"},
                                "category": {"type": "string"},
                                "severity": {
                                    "type": "string",
                                    "enum": ["critical", "high", "medium", "low"],
                                },
                                "file": {
                                    "type": ["string", "null"],
                                    "description": "Path, or null if not file-specific.",
                                },
                                "line": {
                                    "type": ["integer", "null"],
                                    "description": "Line number, or null.",
                                },
                                "description": {"type": "string"},
                                "evidence": {
                                    "type": "string",
                                    "description": "What was observed. Never a secret value.",
                                },
                                "recommendation": {"type": "string"},
                            },
                            "required": [
                                "rule_id",
                                "category",
                                "severity",
                                "file",
                                "line",
                                "description",
                                "evidence",
                                "recommendation",
                            ],
                        },
                    }
                },
                "required": ["violations"],
            },
        },
    },
]

# tools the agent can call for observation (report_findings is terminal)
INSPECTION_TOOLS = {"list_files", "read_file", "search_code", "git_log"}


class Tracer:
    """Appends one JSON object per step to trace.jsonl. Graders read this to
    confirm genuine agent behavior; it is also the #1 debugging aid."""

    def __init__(self, path):
        self.path = path
        # truncate at start of run
        if path:
            with open(path, "w", encoding="utf-8"):
                pass

    def log(self, record):
        if not self.path:
            return
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError:
            pass  # tracing must never break the run


def _make_client():
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No API key found. Set LLM_API_KEY (or GEMINI_API_KEY) in your "
            "environment or .env file."
        )
    base_url = os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)
    return OpenAI(api_key=api_key, base_url=base_url)


def _dispatch(tools, name, args):
    """Run one inspection tool by name. Returns tool output (always data)."""
    if name == "list_files":
        return tools.list_files()
    if name == "read_file":
        return tools.read_file(args.get("path", ""))
    if name == "search_code":
        return tools.search_code(args.get("pattern", ""))
    if name == "git_log":
        return tools.git_log()
    return {"error": f"unknown tool: {name}"}


def _parse_args(raw):
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def run_audit(clone_root, trace_path="trace.jsonl", model=None):
    """Drive the agent loop over the cloned repo at ``clone_root``.

    Returns the list of violation dicts the model reported. Raises on fatal
    LLM/setup errors (the caller decides exit behavior).
    """
    model = model or os.environ.get("LLM_MODEL", DEFAULT_MODEL)
    client = _make_client()
    tools = RepoTools(clone_root)
    tracer = Tracer(trace_path)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Audit this repository. Investigate using the tools, then "
            "call report_findings with your conclusions.",
        },
    ]

    for step in range(1, MAX_ITERATIONS + 1):
        force_final = step == MAX_ITERATIONS
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice=(
                {"type": "function", "function": {"name": "report_findings"}}
                if force_final
                else "auto"
            ),
            temperature=TEMPERATURE,
        )
        msg = response.choices[0].message
        tracer.log(
            {
                "step": step,
                "type": "model_response",
                "text": (msg.content or "")[:TRACE_RESULT_CHARS],
                "tool_calls": [tc.function.name for tc in (msg.tool_calls or [])],
            }
        )

        # append the assistant turn (preserving tool_calls for the protocol)
        messages.append(_assistant_message(msg))

        if not msg.tool_calls:
            # No tool call and not terminal: nudge it to conclude.
            tracer.log({"step": step, "type": "nudge",
                        "reason": "model returned no tool call"})
            messages.append({
                "role": "user",
                "content": "Continue investigating with the tools, or call "
                "report_findings if you have enough evidence.",
            })
            continue

        for tc in msg.tool_calls:
            name = tc.function.name
            args = _parse_args(tc.function.arguments)

            if name == "report_findings":
                violations = args.get("violations", [])
                tracer.log({"step": step, "type": "report_findings",
                            "count": len(violations)})
                return violations

            result = _dispatch(tools, name, args)
            tracer.log({
                "step": step,
                "type": "tool_result",
                "tool": name,
                "args": args,
                "result": json.dumps(result, default=str)[:TRACE_RESULT_CHARS],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, default=str),
            })

    # Should be unreachable: the forced final step returns above. Fail safe.
    tracer.log({"type": "exhausted", "note": "no report_findings produced"})
    return []


def _assistant_message(msg):
    """Rebuild the assistant message dict for the next request, keeping any
    tool_calls so the tool-result messages that follow are valid.

    We preserve ``extra_content`` on each tool call: Gemini (thinking models)
    returns a ``thought_signature`` there and rejects the follow-up request if
    it is not echoed back verbatim. Harmless for providers that don't use it.
    """
    out = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        tool_calls = []
        for tc in msg.tool_calls:
            entry = {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "{}",
                },
            }
            extra = getattr(tc, "extra_content", None)
            if extra:
                entry["extra_content"] = extra
            tool_calls.append(entry)
        out["tool_calls"] = tool_calls
    return out
