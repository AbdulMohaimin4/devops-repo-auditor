"""Unit tests for the inspection tools, run against a tiny fixture repo built
in a temp dir without the need of an API key."""

import sys
from pathlib import Path

import git
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tools import (  # noqa: E402
    MAX_MATCHES,
    MAX_READ_LINES,
    RepoTools,
)


@pytest.fixture
def fixture_repo(tmp_path):
    """A minimal real git repo with a couple of files and two commits."""
    root = tmp_path / "repo"
    root.mkdir()
    repo = git.Repo.init(root)
    repo.config_writer().set_value("user", "name", "Test").release()
    repo.config_writer().set_value("user", "email", "t@example.com").release()

    (root / "README.md").write_text("# Demo\nhello world\n")
    (root / "app.py").write_text("import os\nTOKEN = 'abc'\nprint(os.getcwd())\n")
    sub = root / "pkg"
    sub.mkdir()
    (sub / "util.py").write_text("def add(a, b):\n    return a + b\n")

    repo.index.add(["README.md", "app.py", "pkg/util.py"])
    repo.index.commit("initial commit")
    (root / "app.py").write_text("import os\nTOKEN = 'abc'\nx = 1\n")
    repo.index.add(["app.py"])
    repo.index.commit("tweak app")
    return root


@pytest.fixture
def tools(fixture_repo):
    return RepoTools(fixture_repo)


def test_list_files_relative_and_skips_git(tools):
    result = tools.list_files()
    assert "error" not in result
    files = result["files"]
    assert "README.md" in files
    assert "app.py" in files
    assert "pkg/util.py" in files
    assert not any(f.startswith(".git/") for f in files)


def test_read_file_returns_text(tools):
    result = tools.read_file("README.md")
    assert "error" not in result
    assert "hello world" in result["content"]
    assert result["lines"] == 2


def test_read_file_missing(tools):
    result = tools.read_file("does_not_exist.txt")
    assert "error" in result


def test_read_file_truncation_flagged(tools, fixture_repo):
    big = fixture_repo / "big.txt"
    big.write_text("\n".join(f"line{i}" for i in range(MAX_READ_LINES + 50)))
    result = tools.read_file("big.txt")
    assert result.get("truncated") is True
    assert result["lines"] == MAX_READ_LINES


def test_read_file_rejects_escape(tools):
    for bad in ["../../etc/passwd", "../outside.txt", "/etc/passwd"]:
        result = tools.read_file(bad)
        assert "error" in result, f"expected error for {bad}"
        assert "escape" in result["error"] or "not found" in result["error"]


def test_search_code_finds_pattern(tools):
    result = tools.search_code(r"TOKEN")
    assert "error" not in result
    assert result["count"] >= 1
    hit = result["matches"][0]
    assert hit["path"] == "app.py"
    assert "TOKEN" in hit["snippet"]
    assert isinstance(hit["line"], int)


def test_search_code_invalid_regex(tools):
    result = tools.search_code(r"([unclosed")
    assert "error" in result


def test_search_code_cap(tools, fixture_repo):
    many = "\n".join("match" for _ in range(MAX_MATCHES + 50))
    (fixture_repo / "many.txt").write_text(many)
    result = tools.search_code(r"match")
    assert result["count"] == MAX_MATCHES
    assert result.get("truncated") is True


def test_git_log(tools):
    result = tools.git_log()
    assert "error" not in result
    assert result["count"] == 2
    subjects = [c["subject"] for c in result["commits"]]
    assert "tweak app" in subjects
    assert "initial commit" in subjects
    first = result["commits"][0]
    assert set(first) == {"hash", "author", "date", "subject"}


def test_git_log_non_repo(tmp_path):
    result = RepoTools(tmp_path).git_log()
    assert "error" in result
