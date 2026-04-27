# `setup/` — Qdrant + Neo4j compose stack

Docker Compose stack that runs Qdrant (vectors) and Neo4j 5+ (graph) for the
mem0 MCP server defined in the repo root. The mem0 server itself runs on
demand as an MCP stdio child launched by Claude Code (via `uvx`) — it is not
a long-running container.

Persistent data lives outside the repo at `$MEM0_DATA_DIR` (default
recommendation: `~/.mem0-stack/`) so it survives clones, branch switches,
moves, and `git clean`.

## Stack

| Service | Image | Ports | Role |
|---|---|---|---|
| qdrant | `qdrant/qdrant:latest` | 6333 (REST), 6334 (gRPC) | vector store |
| neo4j  | `neo4j:5.26-community` (LTS) + APOC | 7474 (HTTP), 7687 (Bolt) | graph store |
| ollama | host-native, **not in compose** | 11434 | embeddings (`bge-m3`) and optionally local LLM |

## First-time setup

```bash
# 1. Configure data location + ports/passwords
cp .env.example .env
# Edit .env and set MEM0_DATA_DIR to an absolute path
# (Docker Compose does not expand ~ or $HOME). Example:
#   MEM0_DATA_DIR=/Users/you/.mem0-stack

# 2. Bring up the stack (creates $MEM0_DATA_DIR/{qdrant,neo4j}/* on first run)
docker compose up -d

# 3. (Optional) Register the MCP server in Claude Code with values from <repo>/.env
./scripts/setup-mcp.sh

# 4. Restart Claude Code so the MCP stdio child re-spawns with new env.
```

Embeddings need Ollama running natively on the host:

```bash
brew install ollama        # macOS; for other OSes see https://ollama.com
ollama pull bge-m3
ollama serve               # runs on :11434
```

## Day-to-day

```bash
docker compose up -d            # start
docker compose ps               # status
docker compose logs -f neo4j    # tail
docker compose down             # stop (data persists in $MEM0_DATA_DIR)

./scripts/backup.sh             # snapshot both stores -> $MEM0_DATA_DIR/backups/<ts>/
```

Neo4j browser: http://localhost:7474 — login with values from `.env`
(`neo4j` / `mem0graph` by default).

## Files

```
docker-compose.yml         Qdrant + Neo4j 5.26 with APOC
.env                       MEM0_DATA_DIR + ports + Neo4j password (gitignored)
.env.example               template for .env
scripts/
  setup-mcp.sh             claude mcp remove + add, forwards <repo>/.env
  backup.sh                snapshot Qdrant + dump Neo4j -> $MEM0_DATA_DIR/backups/<ts>/
  restore-qdrant.sh        restore a Qdrant .snapshot into the running stack
```

## MCP server config vs. compose config

There are **two** env files in this repo, and they serve different purposes:

| File | What's in it | Who reads it |
|------|-------------|--------------|
| `<repo>/.env` | MCP server runtime (LLM provider, API keys, Qdrant URL, Neo4j creds for the *client*) | the mem0 MCP server (the Python package); `setup/scripts/setup-mcp.sh` |
| `setup/.env` | Compose stack runtime (data dir, host ports, Neo4j *server* heap/password) | `docker compose` |

Some values legitimately appear in both — e.g. the Neo4j password. The compose
file uses it to *create* the user; the MCP server uses it to *connect*. Keep
them in sync if you change one.

## Bringing your own infra

If you already run Qdrant and/or Neo4j elsewhere, skip `docker compose up`
entirely. Just set `MEM0_QDRANT_URL` / `MEM0_NEO4J_URL` (and creds) in the
repo-root `.env` so the MCP server points at your existing services. The
stack here is convenience scaffolding, not a hard dependency.
