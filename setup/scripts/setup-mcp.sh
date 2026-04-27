#!/usr/bin/env bash
# Re-register the `mem0` MCP entry at user scope.
#
# Reads the MCP server's runtime env from the repo root .env (or .env.example
# as a fallback) and forwards every uncommented variable to `claude mcp add`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

ENV_FILE="$REPO_ROOT/.env"
if [[ ! -f "$ENV_FILE" ]]; then
  ENV_FILE="$REPO_ROOT/.env.example"
  echo "[setup-mcp] $REPO_ROOT/.env not found — sourcing .env.example."
  echo "[setup-mcp] copy .env.example to .env and fill in real values for production use."
fi

set -a; source "$ENV_FILE"; set +a

FORK="${MEM0_FORK_PATH:-$REPO_ROOT}"
if [[ ! -f "$FORK/pyproject.toml" ]]; then
  echo "[setup-mcp] no pyproject.toml at $FORK — set MEM0_FORK_PATH or run from repo root" >&2
  exit 1
fi

# Forward every uncommented variable from the env file (skips comments and
# blank lines). Vars with empty values are skipped so `claude mcp add` doesn't
# clutter its config with ineffective entries.
ENV_ARGS=()
while IFS='=' read -r name _; do
  val="${!name:-}"
  if [[ -n "$val" ]]; then
    ENV_ARGS+=(--env "$name=$val")
  fi
done < <(grep -E '^[A-Z_][A-Z0-9_]*=' "$ENV_FILE")

echo "[setup-mcp] removing existing 'mem0' entry (user scope) if present"
claude mcp remove mem0 -s user >/dev/null 2>&1 || true

echo "[setup-mcp] adding 'mem0' (fork: $FORK)"
claude mcp add --scope user --transport stdio mem0 \
  "${ENV_ARGS[@]}" \
  -- uvx --from "$FORK" mem0-mcp-selfhosted

echo "[setup-mcp] done. Restart Claude Code so the stdio child re-spawns."
claude mcp get mem0
