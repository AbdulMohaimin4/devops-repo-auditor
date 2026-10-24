"""Synthetic evaluation repositories, defined as data.

Three fixtures mirror the hidden-repo categories from the assignment (§6):
  A - obvious defects, B - subtler CI/security/config defects, C - healthy
  (the false-positive trap).

They are built into a throwaway git repo at run time (not committed here) so
that git_log / git_status have real history to inspect, and so this repo never
tracks the junk files a fixture deliberately contains. The expected findings
live beside this file in answer_key.json.
"""

import subprocess
from pathlib import Path

# name -> {relative path: file text}. A fixture is the sum of its files.
FIXTURES = {
    # ---- A: obvious, high-signal defects ---------------------------------
    "repo_a": {
        "config.py": (
            "import os\n\n"
            "# hardcoded credentials committed straight into source\n"
            'AWS_ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"\n'
            'DB_PASSWORD = "sup3rSecretPassw0rd"\n\n'
            "def connect():\n"
            "    return (AWS_ACCESS_KEY_ID, DB_PASSWORD)\n"
        ),
        "app.py": (
            "import os\n"
            "import sys\n\n"  # unused imports -> ruff F401
            "def main():\n"
            "    print('running')\n"
        ),
        "requirements.txt": "flask\nrequests\n",  # unpinned -> weak constraints
        ".env": "SECRET_TOKEN=ghp_exampletokenvalue1234567890abcd\n",  # committed secret file
        "__pycache__/app.cpython-311.pyc": "fake bytecode\n",  # tracked build artifact
        ".github/workflows/ci.yml": (
            "name: CI\n"
            "on: [push]\n"
            "permissions: write-all\n"  # over-privileged
            "jobs:\n"
            "  build:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - uses: actions/checkout@v4\n"
            "      - run: pytest || true\n"  # gate neutralised
        ),
        # deliberately no README.md
    },

    # ---- B: subtler CI / security / configuration defects ----------------
    "repo_b": {
        "README.md": "# service-b\n\nA small service.\n",
        "pyproject.toml": (
            "[project]\n"
            'name = "service-b"\n'
            'version = "0.1.0"\n'
            'dependencies = ["flask==3.0.0", "requests==2.32.3"]\n'
        ),
        "src/service.py": "def handler(event):\n    return {'ok': True}\n",
        "tests/test_service.py": (
            "from src.service import handler\n\n"
            "def test_handler():\n"
            "    assert handler({})['ok'] is True\n"
        ),
        "settings.ini": (
            "[paths]\n"
            "# machine-specific absolute path committed into config\n"
            "cache_dir = /Users/alice/dev/service-b/.cache\n"
        ),
        ".gitignore": ".venv/\n__pycache__/\n*.pyc\n",
        ".github/workflows/ci.yml": (
            "name: CI\n"
            "on:\n"
            "  pull_request_target:\n"  # runs with secrets on untrusted PRs
            "    branches: [main]\n"
            "jobs:\n"
            "  lint:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - uses: actions/checkout@main\n"  # unpinned action + PR head
            "        with:\n"
            "          ref: ${{ github.event.pull_request.head.sha }}\n"
            "      - run: ruff check .\n"  # lints, but never runs the tests
        ),
    },

    # ---- C: healthy repo (false-positive trap) ---------------------------
    "repo_c": {
        "README.md": (
            "# widget\n\n"
            "A tiny, well-kept library.\n\n"
            "## Install\n\n`pip install -e .`\n\n## Test\n\n`pytest`\n"
        ),
        "pyproject.toml": (
            "[project]\n"
            'name = "widget"\n'
            'version = "1.2.0"\n'
            'dependencies = ["httpx==0.27.0"]\n\n'
            "[tool.ruff]\n"
            "line-length = 100\n"
        ),
        "src/widget/__init__.py": "from .core import add\n\n__all__ = ['add']\n",
        "src/widget/core.py": (
            '"""Core arithmetic."""\n\n\n'
            "def add(a, b):\n"
            '    """Return the sum of a and b."""\n'
            "    return a + b\n"
        ),
        "tests/test_core.py": (
            "from widget.core import add\n\n\n"
            "def test_add():\n"
            "    assert add(2, 3) == 5\n"
        ),
        ".gitignore": ".venv/\n__pycache__/\n*.pyc\ndist/\n*.egg-info/\n",
        ".github/workflows/ci.yml": (
            "name: CI\n"
            "on:\n"
            "  push:\n"
            "    branches: [main]\n"
            "  pull_request:\n"
            "    branches: [main]\n"
            "permissions:\n"
            "  contents: read\n"  # least privilege
            "jobs:\n"
            "  test:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - uses: actions/checkout@v4\n"
            "      - uses: actions/setup-python@v5\n"
            "        with:\n"
            "          python-version: '3.12'\n"
            "      - run: pip install -e . ruff pytest\n"
            "      - run: ruff check .\n"
            "      - run: pytest\n"  # tests actually gate the build
        ),
    },
}


def build(name, dest):
    """Materialise fixture ``name`` into ``dest`` as a real git repo with one
    commit, so git_log/git_status work. Returns the Path to ``dest``."""
    if name not in FIXTURES:
        raise KeyError(f"unknown fixture: {name}")
    dest = Path(dest)
    for rel, text in FIXTURES[name].items():
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    def git(*args):
        subprocess.run(["git", "-C", str(dest), *args],
                       check=True, capture_output=True, text=True)

    git("init", "-q")
    git("config", "user.name", "Fixture")
    git("config", "user.email", "fixture@example.com")
    git("add", "-A", "-f")  # -f: commit the junk files a fixture wants tracked
    git("commit", "-q", "-m", f"seed {name}")
    return dest
