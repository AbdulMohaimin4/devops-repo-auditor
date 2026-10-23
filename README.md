# devops-repo-auditor

An AI agent that autonomously inspects a public GitHub repository, identifies
DevOps best-practice violations, and writes an evidence-based JSON report.

It is a genuine LLM **agent/tool loop** — not a fixed checklist. The model
observes the repo, reasons, picks a tool, interprets the result, and decides
what to look at next, concluding by calling a terminal `report_findings` tool.
No agent framework (LangChain/LangGraph/n8n) is used; the loop, tool dispatch,
and control flow are plain Python.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Configure

The agent talks to any OpenAI-compatible chat-completions endpoint. Defaults
target Google Gemini's OpenAI-compatible API. Copy the example env and add a key:

```bash
cp .env.example .env
# edit .env and set LLM_API_KEY=<your key>
```

| Variable | Required | Default |
|---|---|---|
| `LLM_API_KEY` | yes (or `GEMINI_API_KEY`) | — |
| `LLM_MODEL` | no | `gemini-flash-lite-latest` |
| `LLM_BASE_URL` | no | Gemini's OpenAI-compatible endpoint |

## Run

```bash
python src/agent.py --repo <github_repository_url> --output report.json
```

Runs unattended. On success it writes valid JSON (an empty `violations` array if
the repo is healthy) and exits 0. On a fatal error (bad URL, clone/auth failure,
or a persistent LLM API error) it prints a one-line message to stderr and exits
nonzero — it never writes a faked empty report. Logs go to stderr; only the
report goes to the output file. See `sample_report.json` for real output.

## Output contract

Each violation has exactly these fields, always present:

```json
{
  "rule_id": "SEC-001", "category": "security", "severity": "critical",
  "file": "config.py", "line": 17,
  "description": "...", "evidence": "...", "recommendation": "..."
}
```

`severity` is one of `critical|high|medium|low`; `file`/`line` are JSON `null`
when not applicable; secret values are never reproduced (they are redacted and
described). Model output is validated and repaired before writing, so a
malformed finding can't corrupt the report.

## Architecture

```
agent.py        CLI, URL validation, shallow clone (depth 50) into a temp dir,
                validate + write report, temp-dir cleanup in finally.
agent_loop.py   The loop: system prompt + tool schemas -> model call -> dispatch
                tool calls -> feed results back -> repeat until report_findings
                (or an iteration cap forces a final call). Traces every step to
                trace.jsonl. Retries transient API errors with backoff.
tools.py        Read-only inspection tools (errors returned as data, never raised).
rules.py        Rule catalog (stable rule_ids + default severities per area),
                injected into the prompt as hints and reused by validation.
validation.py   Enforces the 8-field contract, severities, nulls, secret redaction.
```

### Inspection tools

| Tool | Purpose |
|---|---|
| `list_files()` | Map the repo (root-relative paths, skips `.git/`) |
| `read_file(path)` | Read a file's text (capped; path-escape guarded) |
| `search_code(pattern)` | Regex search for suspicious patterns |
| `git_log()` | Recent commit history |
| `read_github_actions()` | Parse `.github/workflows/*.yml` into structure (triggers, permissions, jobs/steps) |

All tool output is interpreted by the model; the tools themselves make no
findings.

## Safety

- The target repo is **never modified** and its code is **never executed**.
  There is deliberately no `run_pytest` tool — running untrusted code needs
  isolation the brief forbids, so testing is assessed statically from CI config
  and the shape of the test suite.
- Tool paths are confined to the clone root (`../../etc/passwd` is rejected).
- Subprocesses (clone, `git log`) run with hard timeouts.
- Secrets are redacted from evidence; the location is the evidence, not the value.

## Tests

```bash
PYTHONPATH=src python -m pytest -q
```

Covers each tool against a fixture repo, the agent loop with a mocked model
(no API key needed), output validation/redaction, and reliability paths.
