#!/usr/bin/env bash
# One-time GitHub setup on RunPod: gh CLI + git push auth.
#
# Usage (paste your token when prompted, or export first):
#   export GITHUB_TOKEN=ghp_xxxxxxxx
#   ./demos/realtime_motion_graph_web/runpod/setup_github.sh
#
# Token needs at least: repo, read:org (fork/private), workflow (optional).

set -euo pipefail

TOKEN="${GITHUB_TOKEN:-${GH_TOKEN:-}}"

if [[ -z "${TOKEN}" ]]; then
  echo "Paste a GitHub Personal Access Token (ghp_... or github_pat_...):"
  read -rs TOKEN
  echo
fi

if [[ -z "${TOKEN}" ]]; then
  echo "No token provided. Set GITHUB_TOKEN or GH_TOKEN and re-run." >&2
  exit 1
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "Installing gh..."
  apt-get update -qq
  apt-get install -y -qq gh
fi

# gh auth
echo "${TOKEN}" | gh auth login --with-token

# Persist for this shell + child processes (add to ~/.bashrc if missing)
MARKER="# DEMON runpod github"
if ! grep -qF "${MARKER}" ~/.bashrc 2>/dev/null; then
  {
    echo ""
    echo "${MARKER}"
    echo "export GITHUB_TOKEN=\"\${GITHUB_TOKEN:-}\""
    echo "export GH_TOKEN=\"\${GH_TOKEN:-\${GITHUB_TOKEN}}\""
  } >> ~/.bashrc
fi
export GITHUB_TOKEN="${TOKEN}"
export GH_TOKEN="${TOKEN}"

# Git identity (override via env before running)
GIT_NAME="${GIT_USER_NAME:-M1nd 3xpand3r}"
GIT_EMAIL="${GIT_USER_EMAIL:-}"
if [[ -z "${GIT_EMAIL}" ]]; then
  GIT_EMAIL="$(gh api user -q .email 2>/dev/null || true)"
fi
if [[ -z "${GIT_EMAIL}" ]]; then
  GIT_EMAIL="$(gh api user -q .login)@users.noreply.github.com"
fi
git config --global user.name "${GIT_NAME}"
git config --global user.email "${GIT_EMAIL}"

# HTTPS push via gh credential helper
gh auth setup-git

echo ""
echo "GitHub setup OK:"
gh auth status
echo ""
echo "Logged in as: $(gh api user -q .login)"
echo "Git user: $(git config --global user.name) <$(git config --global user.email)>"
echo ""
echo "Fork example:"
echo "  gh repo fork daydreamlive/DEMON --clone=false"
echo ""
echo "MCP (Grok): grok_com_github is separate — already uses your linked GitHub account in Cursor/Grok."