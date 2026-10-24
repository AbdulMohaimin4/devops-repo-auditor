"""Tests for the eval harness: fixture building and the pure scoring logic.

These run without an API key -- they cover the matcher and the precision/recall
maths, not the live audit.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from build_fixtures import FIXTURES, build  # noqa: E402
from run_eval import matches, score  # noqa: E402
from tools import RepoTools  # noqa: E402


def _v(category, file="", description="", evidence=""):
    return {"category": category, "file": file,
            "description": description, "evidence": evidence}


def test_matches_category_and_file_token():
    item = {"category": "security", "file": "config.py"}
    assert matches(item, _v("security", file="config.py"))
    assert not matches(item, _v("ci", file="config.py"))        # wrong category
    assert not matches(item, _v("security", file="app.py"))     # token absent


def test_matches_token_found_in_evidence_not_just_file():
    item = {"category": "security", "file": "pull_request_target"}
    v = _v("security", file=".github/workflows/ci.yml",
           evidence="uses pull_request_target trigger")
    assert matches(item, v)


def test_matches_accepts_category_list():
    item = {"category": ["testing", "ci"], "file": None}
    assert matches(item, _v("ci"))
    assert matches(item, _v("testing"))
    assert not matches(item, _v("security"))


def test_score_perfect():
    expected = [{"category": "security", "file": "config.py"}]
    reported = [_v("security", file="config.py", description="hardcoded key")]
    s = score(reported, expected)
    assert (s["tp"], s["fp"], s["fn"]) == (1, 0, 0)
    assert s["precision"] == 1.0 and s["recall"] == 1.0


def test_score_false_positive_and_miss():
    expected = [{"category": "security", "file": "config.py"},
                {"category": "ci", "file": "ci.yml"}]
    reported = [_v("security", file="config.py"),
                _v("dependency management", file="requirements.txt")]  # spurious
    s = score(reported, expected)
    assert s["tp"] == 1 and s["fp"] == 1 and s["fn"] == 1
    assert s["precision"] == 0.5 and s["recall"] == 0.5
    assert s["spurious"]


def test_score_each_expected_matched_once():
    # two reports of the same issue must not score as two true positives
    expected = [{"category": "security", "file": "config.py"}]
    reported = [_v("security", file="config.py"), _v("security", file="config.py")]
    s = score(reported, expected)
    assert s["tp"] == 1 and s["fp"] == 1  # the duplicate is a false positive


def test_score_wildcard_item_does_not_block_specific_match():
    # A (file:null) matches any hygiene report; B needs the ".gitignore" one.
    # Greedy order would let A steal r0 and miss B; max matching pairs both.
    expected = [{"category": "repository hygiene", "file": None},
                {"category": "repository hygiene", "file": ".gitignore"}]
    reported = [_v("repository hygiene", description="missing .gitignore file"),
                _v("repository hygiene", description="tracked __pycache__")]
    s = score(reported, expected)
    assert s["tp"] == 2 and s["fp"] == 0 and s["fn"] == 0


def test_score_healthy_repo_penalises_any_report():
    assert score([], [])["precision"] == 1.0            # clean pass
    s = score([_v("security", file="x.py")], [])        # invented issue
    assert s["fp"] == 1 and s["precision"] == 0.0


def test_build_fixture_creates_git_repo_with_tracked_junk(tmp_path):
    build("repo_a", tmp_path)
    assert (tmp_path / "config.py").is_file()
    assert not (tmp_path / "README.md").exists()         # deliberately missing
    log = RepoTools(tmp_path).git_log()
    assert log["count"] == 1                             # one seed commit
    status = RepoTools(tmp_path).git_status()
    assert ".env" in status["tracked_junk"]              # junk really committed
    assert any("__pycache__" in f for f in status["tracked_junk"])


def test_all_fixtures_build(tmp_path):
    for name in FIXTURES:
        build(name, tmp_path / name)
        assert (tmp_path / name / ".git").is_dir()
