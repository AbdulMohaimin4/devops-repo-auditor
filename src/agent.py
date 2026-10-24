import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlparse

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is a convenience; env vars still work without it
    def load_dotenv(*_a, **_k):
        return False

from agent_loop import run_audit
from validation import validate_violations

CLONE_DEPTH = 50
CLONE_TIMEOUT = 300  # seconds; a clone that hangs must not hang the run


class CloneError(Exception):
    """Raised when cloning the target repository fails."""


def validate(repo, output):
    """Return True if the repo URL looks like a public GitHub repo and the
    output path is a .json file."""
    parsed = urlparse(repo)

    if parsed.scheme != "https" or parsed.netloc != "github.com":
        return False

    if len(parsed.path.strip("/").split("/")) != 2:
        return False

    if not output.endswith(".json"):
        return False

    return True


def clone_repo(repo, dest):
    """Shallow-clone the target repo into dest. Depth 50 keeps history
    meaningful for git_log while keeping the clone fast. Runs as a subprocess
    with a hard timeout and raises CloneError on any failure."""
    try:
        proc = subprocess.run(
            ["git", "clone", "--depth", str(CLONE_DEPTH), "--", repo, dest],
            capture_output=True,
            text=True,
            timeout=CLONE_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        raise CloneError(f"clone timed out after {CLONE_TIMEOUT}s")
    except OSError as exc:
        raise CloneError(str(exc))
    if proc.returncode != 0:
        raise CloneError((proc.stderr or "git clone failed").strip())


def write_report(output, repo, violations):
    with open(output, "w", encoding="utf-8") as f:
        json.dump({"repository": repo, "violations": violations}, f, indent=2)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Clone a public GitHub repository, identify DevOps "
        "best-practice violations, and generate an evidence-based JSON report."
    )
    parser.add_argument(
        "--repo",
        required=True,
        type=str,
        help="URL of a public GitHub repository",
    )
    parser.add_argument(
        "--output",
        default="report.json",
        type=str,
        help="path of the output JSON file (default: report.json)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    load_dotenv()
    args = parse_args(argv)

    if not validate(args.repo, args.output):
        print(
            "error: invalid GitHub repository URL or output filename "
            "(expected https://github.com/<owner>/<repo> and a .json output)",
            file=sys.stderr,
        )
        return 2

    tmpdir = tempfile.mkdtemp(prefix="repo-auditor-")
    try:
        try:
            clone_repo(args.repo, tmpdir)
        except CloneError as exc:
            print(f"error: failed to clone {args.repo}: {exc}", file=sys.stderr)
            return 1

        # Run the LLM-driven agent loop over the clone. On any failure we exit
        # nonzero rather than writing an empty report, which would be an
        # untruthful "healthy repo" result.
        try:
            raw = run_audit(tmpdir)
        except Exception as exc:  # noqa: BLE001 - fatal: report and exit nonzero
            print(f"error: audit failed: {exc}", file=sys.stderr)
            return 1

        # Enforce the output contract before writing (schema, severities,
        # null file/line, secret redaction). Logs go to stderr; stdout stays
        # clean so it never corrupts a consumer reading the JSON file.
        violations, dropped = validate_violations(raw)
        if dropped:
            print(f"warning: dropped {dropped} malformed violation(s)",
                  file=sys.stderr)
        print(f"audit complete: {len(violations)} violation(s)", file=sys.stderr)

        write_report(args.output, args.repo, violations)
        return 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
