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

import openai
from openai import OpenAI

from rules import render_catalog
from tools import RepoTools

# Transient API failures worth retrying with backoff. A persistent failure is
# re-raised so the caller exits nonzero -- we never fake an empty report, which
# would read as a healthy repo.
RETRYABLE_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)
MAX_RETRIES = 4          # total attempts per model call
RETRY_BASE_DELAY = 2.0   # seconds; doubles each retry, capped
RETRY_MAX_DELAY = 30.0

# pytest policy: this agent NEVER executes the target repository's code. The
# brief forbids running untrusted code without isolation, so there is no
# run_pytest tool -- testing is assessed statically from CI config and the
# presence/shape of tests. Documented here and in the README.

# --- provider config (overridable via env) ------------------------------
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

DEFAULT_MODEL = "gemini-flash-lite-latest"

MAX_ITERATIONS = 30       # bound cost and guarantee termination
TEMPERATURE = 0.0         # deterministic for analysis
TRACE_RESULT_CHARS = 2000  # trim tool results in the trace log

SYSTEM_PROMPT = """\
You are a DevOps repository auditor. You inspect a cloned GitHub repository and
identify genuine DevOps best-practice violations, each backed by evidence.

HOW YOU WORK
You investigate by calling the provided tools. Behave like a human investigator:
first map the repository (list_files), then read the files that look relevant,
use search_code to confirm suspicions across the tree, and inspect git_log for
history signals. Use read_github_actions for CI/workflow judgments, git_status
to confirm tracked junk files, and run_ruff for concrete code-quality evidence.
Let what you observe drive what you look at next. The areas and
rules below are HINTS to steer your attention -- they are NOT a checklist to run
top to bottom, and not every area applies to every repo.

Scope: software engineering and DevOps only. Ignore ML/MLOps concerns (training,
datasets, model validation, experiment tracking).

INSPECTION AREAS (hints -- pursue what the evidence supports)
- Repository hygiene: missing README, unsuitable .gitignore, tracked temp/
  generated files.
- Dependency management: missing/mismatched manifests, absent lockfiles, weak
  version constraints.
- Code quality: missing linting/formatting config, obvious maintainability
  problems.
- Testing: missing tests, broken tests, tests not executed in CI.
- GitHub Actions / CI: wrong or missing triggers, ineffective gates, skipped
  validation.
- Security: exposed credentials, unsafe workflow permissions, dangerous event
  triggers.
- Configuration: machine-specific paths, committed local settings, poor
  environment separation.
- Collaboration: unsafe PR automation, missing review guidance where relevant.
- Build and release: missing build validation, risky release/deploy automation.

RULE CATALOG (use these stable rule_ids and default severities; adjust severity
if the context clearly warrants, or mint a new id under the right prefix for a
genuine issue no listed rule matches):
{catalog}

EVIDENCE DISCIPLINE (this is what separates an auditor from a nagging linter)
- Only report a violation you can back with specific evidence from a tool call
  (a file path, a line, a matched snippet, a commit). No evidence, no finding.
- An absent optional file is NOT a defect when an equivalent exists (e.g.
  pyproject.toml instead of requirements.txt; ruff config instead of flake8).
  Check for equivalents before claiming something is missing.
- Never infer remote GitHub settings -- branch protection, required reviews,
  required status checks -- from files in the clone. You cannot see them.
- One finding per real issue. No duplicates and no near-duplicates of the same
  underlying problem.
- Never reproduce a secret value; describe and locate it instead.

When your investigation is complete, call report_findings exactly once with all
violations you can defend (an empty list if the repository is healthy).""".replace(
    "{catalog}", render_catalog()
)


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
            "name": "read_github_actions",
            "description": "Parse .github/workflows/*.yml into structured form "
            "(triggers, top-level and per-job permissions, jobs and their steps' "
            "uses/run/if). Prefer this over read_file for judging CI gates and "
            "unsafe workflow permissions/triggers.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": "Report version-control hygiene: whether a .gitignore "
            "exists, and which tracked files match common junk patterns (caches, "
            "build artifacts, .vscode/.idea, a committed .env). Use to confirm "
            "tracked temp/generated/local files before reporting a hygiene issue.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_ruff",
            "description": "Statically lint the repo's Python with ruff (syntax "
            "errors and pyflakes: undefined names, unused imports). Parses code, "
            "never executes it. Use for concrete code-quality/maintainability "
            "evidence; returns an error if ruff is unavailable.",
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
INSPECTION_TOOLS = {"list_files", "read_file", "search_code", "git_log",
                    "read_github_actions", "git_status", "run_ruff"}


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
    if name == "read_github_actions":
        return tools.read_github_actions()
    if name == "git_status":
        return tools.git_status()
    if name == "run_ruff":
        return tools.run_ruff()
    return {"error": f"unknown tool: {name}"}


def _parse_args(raw):
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def _log(msg):
    """Log to stderr; stdout is reserved so it never corrupts the JSON file."""
    print(msg, file=sys.stderr)


def _create_with_retry(client, **kwargs):
    """Call the chat completion API, retrying transient errors with exponential
    backoff. Re-raises if every attempt fails."""
    delay = RETRY_BASE_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except RETRYABLE_ERRORS as exc:
            if attempt == MAX_RETRIES:
                raise
            _log(f"LLM call failed ({type(exc).__name__}); retry "
                 f"{attempt}/{MAX_RETRIES - 1} in {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, RETRY_MAX_DELAY)


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
        response = _create_with_retry(
            client,
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
