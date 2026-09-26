#!/usr/bin/env bash
# Runs at the start of every Claude Code session in this repo.
# Makes sure commits are authored by the repo owner and AI trailers get stripped.
set -euo pipefail
cd "${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel)}"

git config user.name "Harkirat Singh Sandhu"
git config user.email "u7810361@anu.edu.au"
git config core.hooksPath .githooks
chmod +x .githooks/* 2>/dev/null || true
