#!/usr/bin/env python3
"""Root entry point so the assignment's exact command works from the repo root:

    python agent.py --repo <github_repository_url> --output report.json

The implementation lives in src/; this puts src on the path and delegates.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from agent import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
