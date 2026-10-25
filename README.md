# devops-repo-auditor

An AI agent that autonomously inspects a public GitHub repository, identifies
DevOps best-practice violations, and writes an evidence-based JSON report.

The application is a **agent/tool loop**, and does not simply rely on a checklist
of predetermined violation. The model observes the repo, reasons, picks a tool, 
interprets the result, and decides what to look at next, concluding by calling a 
terminal `report_findings` tool. No agent framework (LangChain/LangGraph/n8n) is 
used; the loop, tool dispatch, and control flow are all written in plain Python.

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
python agent.py --repo <github_repository_url> --output report.json
```

(`agent.py` at the repo root is a thin entry point; the implementation lives in
`src/`. `python src/agent.py ...` works identically.)

Runs unattended. On success it writes valid JSON (an empty `violations` array if
the repo is healthy) and exits 0. On a fatal error (bad URL, clone/auth failure,
or a persistent LLM API error) it prints a one-line message to stderr and exits
nonzero. Logs go to stderr, and only the report goes to the output file. 
See `sample_report.json` for real output.

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
eval/           Self-evaluation: synthetic fixtures + precision/recall harness.
```

### Inspection tools

| Tool | Purpose |
|---|---|
| `list_files()` | Map the repo (root-relative paths, skips `.git/`) |
| `read_file(path)` | Read a file's text (capped and path-escape guarded) |
| `search_code(pattern)` | Regex search for suspicious patterns |
| `git_log()` | Recent commit history |
| `read_github_actions()` | Parse `.github/workflows/*.yml` into structure (triggers, permissions, jobs/steps) |
| `git_status()` | Hygiene signals: `.gitignore` presence and tracked files matching common junk patterns |
| `run_ruff()` | Static lint of the repo's Python (syntax errors, undefined names, unused imports) |

All tool output is interpreted by the model; the tools themselves make no
findings.

## Safety

- The target repo is **never modified** and its code is **never executed**.
  There is deliberately no `run_pytest` tool as running untrusted code needs
  isolation, so testing is assessed statically from CI config
  and the shape of the test suite. `run_ruff()` is safe because ruff only
  *parses* Python (it never imports or runs it) and uses `--isolated` so the
  target's own ruff config cannot alter the result.
- Tool paths are confined to the clone root (`../../etc/passwd` is rejected).
- Subprocesses (clone, `git log`) run with hard timeouts.
- Secrets are redacted from evidence.

## Tests

```bash
PYTHONPATH=src python -m pytest -q
```

Covers each tool against a fixture repo, the agent loop with a mocked model
(no API key needed), output validation/redaction, and reliability paths.

## Evaluation (precision / recall)

`eval/` simulates its three categories with synthetic fixtures:
`repo_a` (obvious defects), `repo_b` (subtler CI/security/config defects), and
`repo_c` (healthy, the false-positive trap). Each fixture is defined as data in
`eval/build_fixtures.py` and built into a throwaway git repo at run time; the
expected defects live in `eval/answer_key.json`.

```bash
python eval/run_eval.py            # audit all fixtures, print precision/recall
python eval/run_eval.py repo_c     # one fixture
```

It runs the real agent on each fixture, scores reports against the key (a report
is a true positive when its category matches and the expected file/evidence
token appears), and writes per-fixture results, including every reported
violation, to `eval/results.json`. The scoring logic is pure and unit-tested in
`tests/test_eval.py` without any network. Use it to tune: low recall -> read the
trace and strengthen a prompt hint or add a tool; false positives on `repo_c` ->
tighten the evidence rules.
