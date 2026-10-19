"""Inspection tools the agent dispatches against a cloned repository.

Every tool is read-only and returns plain data (dict/list) on success or
``{"error": "..."}`` on failure. Tools never raise: the agent must be able to
*see* a failure as an observation and react (try another path), rather than
have the loop crash.
"""

import os
import re
from pathlib import Path

import git

# --- caps -------------------------------------------------------------------
# We pay tokens for every byte returned to the model, so each tool bounds its
# output. Truncation is always flagged so the model knows more exists.
MAX_FILES = 2000          # list_files: entries before truncation
MAX_READ_LINES = 500      # read_file: lines before truncation
MAX_READ_BYTES = 50_000   # read_file: bytes before truncation
MAX_MATCHES = 100         # search_code: matches before truncation
MAX_SNIPPET = 200         # search_code: chars per matched line
MAX_COMMITS = 50          # git_log: commits returned

# Directories we never descend into regardless of the tool.
SKIP_DIRS = {".git"}


class RepoTools:
    """Read-only inspection tools bound to one clone root."""

    def __init__(self, root):
        self.root = Path(root).resolve()

    # -- helpers -------------------------------------------------------------
    def _resolve_inside(self, path):
        """Resolve ``path`` relative to the clone root and ensure it stays
        inside it. Returns the resolved Path, or raises ValueError on escape.

        This is the cheapest safety win here: read_file("../../.ssh/id_rsa")
        must not read outside the repo.
        """
        candidate = (self.root / path).resolve()
        if candidate != self.root and not candidate.is_relative_to(self.root):
            raise ValueError(f"path escapes repository root: {path}")
        return candidate

    # -- tools ---------------------------------------------------------------
    def list_files(self):
        """Discover repository files as root-relative paths, skipping .git/."""
        try:
            files = []
            truncated = False
            for dirpath, dirnames, filenames in os.walk(self.root):
                # prune skip dirs in place so os.walk doesn't descend
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
                for name in filenames:
                    rel = os.path.relpath(os.path.join(dirpath, name), self.root)
                    files.append(rel.replace(os.sep, "/"))
                    if len(files) >= MAX_FILES:
                        truncated = True
                        break
                if truncated:
                    break
            files.sort()
            result = {"files": files, "count": len(files)}
            if truncated:
                result["truncated"] = True
                result["note"] = f"listing capped at {MAX_FILES} files"
            return result
        except Exception as exc:  # noqa: BLE001 - errors are data here
            return {"error": str(exc)}

    def read_file(self, path):
        """Return a file's text, capped at ~500 lines / ~50KB, flagging when
        truncated. Rejects paths that escape the clone root."""
        try:
            resolved = self._resolve_inside(path)
        except ValueError as exc:
            return {"error": str(exc)}

        try:
            if not resolved.exists():
                return {"error": f"file not found: {path}"}
            if not resolved.is_file():
                return {"error": f"not a regular file: {path}"}

            raw = resolved.read_bytes()
            truncated_bytes = len(raw) > MAX_READ_BYTES
            raw = raw[:MAX_READ_BYTES]
            text = raw.decode("utf-8", errors="replace")

            lines = text.splitlines()
            truncated_lines = len(lines) > MAX_READ_LINES
            lines = lines[:MAX_READ_LINES]

            result = {
                "path": path,
                "content": "\n".join(lines),
                "lines": len(lines),
            }
            if truncated_bytes or truncated_lines:
                result["truncated"] = True
                result["note"] = (
                    f"truncated to first {len(lines)} lines / {MAX_READ_BYTES} "
                    "bytes; request specific ranges if more is needed"
                )
            return result
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def search_code(self, pattern):
        """Search text files for a regex pattern. Returns matches as
        ``path:line: snippet`` entries, capped at MAX_MATCHES."""
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return {"error": f"invalid regex: {exc}"}

        try:
            matches = []
            truncated = False
            for dirpath, dirnames, filenames in os.walk(self.root):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
                for name in filenames:
                    full = os.path.join(dirpath, name)
                    rel = os.path.relpath(full, self.root).replace(os.sep, "/")
                    try:
                        with open(full, "r", encoding="utf-8", errors="strict") as f:
                            for lineno, line in enumerate(f, start=1):
                                if regex.search(line):
                                    snippet = line.strip()[:MAX_SNIPPET]
                                    matches.append(
                                        {"path": rel, "line": lineno, "snippet": snippet}
                                    )
                                    if len(matches) >= MAX_MATCHES:
                                        truncated = True
                                        break
                    except (UnicodeDecodeError, OSError):
                        # skip binary / unreadable files
                        continue
                    if truncated:
                        break
                if truncated:
                    break

            result = {"matches": matches, "count": len(matches)}
            if truncated:
                result["truncated"] = True
                result["note"] = f"search capped at {MAX_MATCHES} matches"
            return result
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def git_log(self):
        """Return recent commits (hash, author, date, subject)."""
        try:
            repo = git.Repo(self.root)
        except Exception as exc:  # noqa: BLE001 (InvalidGitRepositoryError etc.)
            return {"error": f"not a git repository: {exc}"}

        try:
            commits = []
            for commit in repo.iter_commits(max_count=MAX_COMMITS):
                commits.append(
                    {
                        "hash": commit.hexsha[:10],
                        "author": f"{commit.author.name} <{commit.author.email}>",
                        "date": commit.committed_datetime.isoformat(),
                        "subject": commit.summary,
                    }
                )
            return {"commits": commits, "count": len(commits)}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
