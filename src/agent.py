import argparse
import json
import shutil
import sys
import tempfile
from urllib.parse import urlparse

import git

CLONE_DEPTH = 50


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
    meaningful for git_log while keeping the clone fast."""
    git.Repo.clone_from(repo, dest, depth=CLONE_DEPTH)


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
        except git.GitCommandError as exc:
            print(f"error: failed to clone {args.repo}: {exc}", file=sys.stderr)
            return 1

        # Walking skeleton: the inspection loop is not wired in yet, so we
        # report no violations. Milestones 1+ populate this.
        violations = []
        write_report(args.output, args.repo, violations)
        return 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
