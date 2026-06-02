# GitHub quick setup (RunPod + Grok)

Two ways to use GitHub from this environment:

| Method | What it’s for | Status |
|--------|----------------|--------|
| **MCP `grok_com_github`** | Fork, PRs, issues, search — from the AI chat | Usually already linked to your Grok/Cursor account |
| **`gh` CLI + token** | Terminal: fork, clone, push, PR from shell | Run `setup_github.sh` once with a PAT |

## 1. MCP (chat) — already working

The agent can call tools like `fork_repository`, `create_pull_request`, `get_file_contents` when the **grok_com_github** MCP server is enabled in Grok settings.

Verified account in this session: **TheMindExpansionNetwork**.

No extra token in the pod is required for MCP if you’re logged into GitHub in Grok/Cursor.

## 2. CLI + git push — needs a token on the pod

### Create a token

1. GitHub → **Settings** → **Developer settings** → **Personal access tokens**
2. **Fine-grained** or **Classic** with:
   - `repo` (full control of private repositories)
   - `read:org` (if forking org repos)
3. Copy the token (`ghp_...` or `github_pat_...`)

### Run setup on the pod

```bash
cd /workspace/DEMON
export GITHUB_TOKEN=ghp_YOUR_TOKEN_HERE
chmod +x demos/realtime_motion_graph_web/runpod/setup_github.sh
./demos/realtime_motion_graph_web/runpod/setup_github.sh
```

Or paste interactively when prompted.

### Quick commands after setup

```bash
# Fork (no local clone)
gh repo fork daydreamlive/DEMON --clone=false

# Fork and clone to ~/DEMON-fork
gh repo fork daydreamlive/DEMON --clone=true --remote=true

# Open PR from current branch
gh pr create --fill

# Status
gh auth status
```

## 3. This repo remote

```text
origin  https://github.com/daydreamlive/DEMON
```

To push your RunPod fixes to **your fork**:

```bash
gh repo fork daydreamlive/DEMON --remote-name origin-fork
git remote add fork https://github.com/TheMindExpansionNetwork/DEMON.git  # or use gh output
git push fork your-branch
```

## Security

- Do **not** commit tokens to git.
- Prefer `export GITHUB_TOKEN=...` in the terminal only, or RunPod secrets.
- Revoke the token on GitHub when the pod is destroyed if it was a one-off.