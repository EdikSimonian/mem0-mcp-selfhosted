# mem0-mcp-selfhosted

Self-hosted [mem0](https://github.com/mem0ai/mem0) MCP server for Claude Code. Run a complete memory server against self-hosted Qdrant + Neo4j + Ollama, with your choice of Anthropic (Claude) or Ollama as the main LLM.

Uses the `mem0ai` package directly as a library, supports both Claude's OAT token and fully local Ollama setups, and exposes 11 MCP tools for full memory management.

## Prerequisites

You need these services running:

| Service | Required | Purpose |
|---------|----------|---------|
| **Qdrant** | Yes | Vector memory storage and search |
| **Ollama** | Yes | Embedding generation (bge-m3 or similar) |
| **Neo4j 5+** | Optional | Knowledge graph (entity relationships) |
| **Anthropic API** | Conditional | LLM for fact extraction and memory updates. Required when `MEM0_LLM_PROVIDER=anthropic` (default). Not needed for fully local Ollama setups. Auto-authenticates via Claude Code's OAT token. |
| **Google API** | Optional | Graph LLM for entity extraction (`gemini`/`gemini_split` providers) |

Python >= 3.10.

## Quick Start

### Default (Anthropic)

Add the MCP server globally (available across all projects):

```bash
claude mcp add --scope user --transport stdio mem0 \
  --env MEM0_QDRANT_URL=http://localhost:6333 \
  --env MEM0_EMBED_URL=http://localhost:11434 \
  --env MEM0_EMBED_MODEL=bge-m3 \
  --env MEM0_EMBED_DIMS=1024 \
  --env MEM0_USER_ID=your-user-id \
  -- uvx --from git+https://github.com/elvismdev/mem0-mcp-selfhosted.git mem0-mcp-selfhosted
```

`uvx` automatically downloads, installs, and runs the server in an isolated environment — no manual installation needed. Claude Code launches it on demand when the MCP connection starts.

The server auto-reads your OAT token from `~/.claude/.credentials.json` — no manual token configuration needed.

### Fully Local (Ollama)

For a fully local setup with no cloud dependencies, use Ollama for both the main LLM and embeddings:

```bash
claude mcp add --scope user --transport stdio mem0 \
  --env MEM0_LLM_PROVIDER=ollama \
  --env MEM0_LLM_MODEL=qwen3:14b \
  --env MEM0_LLM_URL=http://localhost:11434 \
  --env MEM0_QDRANT_URL=http://localhost:6333 \
  --env MEM0_EMBED_URL=http://localhost:11434 \
  --env MEM0_EMBED_MODEL=bge-m3 \
  --env MEM0_EMBED_DIMS=1024 \
  --env MEM0_USER_ID=your-user-id \
  -- uvx --from git+https://github.com/elvismdev/mem0-mcp-selfhosted.git mem0-mcp-selfhosted
```

Or add it to a single project by creating `.mcp.json` in the project root:

```json
{
  "mcpServers": {
    "mem0": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/elvismdev/mem0-mcp-selfhosted.git", "mem0-mcp-selfhosted"],
      "env": {
        "MEM0_LLM_PROVIDER": "ollama",
        "MEM0_LLM_MODEL": "qwen3:14b",
        "MEM0_LLM_URL": "http://localhost:11434",
        "MEM0_QDRANT_URL": "http://localhost:6333",
        "MEM0_EMBED_URL": "http://localhost:11434",
        "MEM0_EMBED_MODEL": "bge-m3",
        "MEM0_EMBED_DIMS": "1024",
        "MEM0_USER_ID": "your-user-id"
      }
    }
  }
}
```

### Try It

Restart Claude Code, then:

```
> Search my memories for TypeScript preferences
> Remember that I prefer Hatch for Python packaging
> Show me all entities in my knowledge graph
```

## CLAUDE.md Integration

Add these rules to your project's `CLAUDE.md` (or `~/.claude/CLAUDE.md` for global use) so Claude Code automatically uses memory across sessions:

```markdown
# MCP Servers

- **mem0**: Persistent memory across sessions. At the start of each session, `search_memories` for relevant context before asking the user to re-explain anything. Use `add_memory` whenever you discover project architecture, coding conventions, debugging insights, key decisions, or user preferences. Use `update_memory` when prior context changes. Save information like: "This project uses PostgreSQL with Prisma", "Tests run with pytest -v", "Auth uses JWT validated in middleware". When in doubt, save it — future sessions benefit from over-remembering.
```

This gives Claude Code persistent memory across sessions. Instead of re-exploring the codebase every time, it retrieves what it already knows and starts productive work immediately.

## Authentication

The server resolves an Anthropic token using a prioritized fallback chain:

| Priority | Source | Details |
|----------|--------|---------|
| 1 | `MEM0_ANTHROPIC_TOKEN` env var | Explicit, user-controlled |
| 2 | `~/.claude/.credentials.json` | Auto-reads Claude Code's OAT token (zero-config) |
| 3 | `ANTHROPIC_API_KEY` env var | Standard pay-per-use API key |
| 4 | Disabled | Warns and disables Anthropic LLM features |

**OAT tokens** (`sk-ant-oat...`) use your Claude subscription. The server automatically detects the token type and configures the SDK accordingly.

**API keys** (`sk-ant-api...`) use standard pay-per-use billing.

## Tools

### Memory Tools (9 core)

| Tool | Description |
|------|-------------|
| `add_memory` | Store text or conversation history as memories. Supports `enable_graph`, `infer`, `metadata`. |
| `search_memories` | Semantic search with optional `filters`, `threshold`, `rerank`, `enable_graph`. |
| `get_memories` | List/filter memories (non-search). Supports `limit` and scope filters. |
| `get_memory` | Fetch a single memory by UUID. |
| `update_memory` | Replace memory text. Re-embeds and re-indexes in Qdrant. |
| `delete_memory` | Delete a single memory by UUID. |
| `delete_all_memories` | Bulk-delete all memories in a scope. |
| `list_entities` | List users/agents/runs with memory counts. Uses Qdrant Facet API. |
| `delete_entities` | Cascade-delete an entity and all its memories. |

### Graph Tools

| Tool | Description |
|------|-------------|
| `search_graph` | Search Neo4j entities by name substring. Returns entities + outgoing relationships. |
| `get_entity` | Get all relationships for an entity (bidirectional: incoming + outgoing). |

### Prompt

The server registers a `memory_assistant` MCP prompt that provides Claude with a quick-start guide for using the memory tools effectively.

### Parameters

All tools use Pydantic `Annotated[type, Field(description=...)]` for self-documenting parameter schemas. Common patterns:

- **`user_id`** defaults to `MEM0_USER_ID` env var when not provided
- **`enable_graph`** overrides the default `MEM0_ENABLE_GRAPH` per-call
- **`filters`** supports structured operators: `{"key": {"eq": "value"}}`, `{"AND": [...]}`
- All responses are JSON strings via `json.dumps(result, ensure_ascii=False)`

## Configuration

All configuration is via environment variables. Create a `.env` file or set them in your MCP config.

### Authentication

| Variable | Default | Description |
|----------|---------|-------------|
| `MEM0_ANTHROPIC_TOKEN` | -- | Anthropic OAT or API token (priority 1) |
| `ANTHROPIC_API_KEY` | -- | Standard Anthropic API key (priority 3) |
| `MEM0_OAT_HEADERS` | `auto` | OAT identity headers: `auto` or `none` |

### LLM

| Variable | Default | Description |
|----------|---------|-------------|
| `MEM0_LLM_PROVIDER` | `anthropic` | Main LLM provider: `anthropic` or `ollama` |
| `MEM0_LLM_MODEL` | _(per-provider)_ | Model for the selected LLM provider. Defaults to `claude-opus-4-6` for Anthropic, `qwen3:14b` for Ollama |
| `MEM0_LLM_URL` | `http://localhost:11434` | Ollama base URL for the main LLM. Only used when `MEM0_LLM_PROVIDER=ollama` |
| `MEM0_LLM_MAX_TOKENS` | `16384` | Max tokens for LLM responses (Anthropic only) |
| `MEM0_GRAPH_LLM_PROVIDER` | `anthropic` | Graph LLM provider (`anthropic`, `anthropic_oat`, `ollama`, `gemini`, `gemini_split`) |
| `MEM0_GRAPH_LLM_URL` | _(cascades)_ | Ollama base URL for graph LLM. Cascades: `MEM0_GRAPH_LLM_URL` → `MEM0_LLM_URL` → `http://localhost:11434` |
| `MEM0_GRAPH_LLM_MODEL` | _(varies)_ | Graph model. Inherits `MEM0_LLM_MODEL` for anthropic/ollama; defaults to `gemini-2.5-flash-lite` for gemini/gemini_split |
| `GOOGLE_API_KEY` | -- | Google API key (required for `gemini`/`gemini_split` graph providers) |
| `MEM0_GRAPH_CONTRADICTION_LLM_PROVIDER` | `anthropic` | Contradiction LLM provider in `gemini_split` mode (`anthropic`, `anthropic_oat`, `ollama`) |
| `MEM0_GRAPH_CONTRADICTION_LLM_MODEL` | _(inherits MEM0_LLM_MODEL)_ | Contradiction model in `gemini_split` mode |
| `MEM0_OLLAMA_KEEP_ALIVE` | `30m` | How long Ollama keeps the model in VRAM between calls (e.g., `1h`, `5m`). Prevents model unload during multi-call graph pipelines |
| `MEM0_OLLAMA_THINK` | `false` | Set to `true` to re-enable qwen3 thinking mode (disabled by default to prevent `<think>` + `format:"json"` collision) |

### Embedder

| Variable | Default | Description |
|----------|---------|-------------|
| `MEM0_EMBED_PROVIDER` | `ollama` | Embedding provider (`ollama` or `openai`) |
| `MEM0_EMBED_MODEL` | `bge-m3` | Embedding model name |
| `MEM0_EMBED_URL` | `http://localhost:11434` | Ollama URL for embeddings only. For LLM URLs, use `MEM0_LLM_URL` / `MEM0_GRAPH_LLM_URL` |
| `MEM0_EMBED_DIMS` | `1024` | Embedding vector dimensions |

### Vector Store (Qdrant)

| Variable | Default | Description |
|----------|---------|-------------|
| `MEM0_QDRANT_URL` | `http://localhost:6333` | Qdrant REST API URL |
| `MEM0_QDRANT_API_KEY` | -- | Qdrant API key (for Qdrant Cloud) |
| `MEM0_QDRANT_ON_DISK` | `false` | Store vectors on disk (reduces RAM, slower search) |
| `MEM0_QDRANT_TIMEOUT` | _(client default)_ | Qdrant REST API timeout in seconds (e.g., `30`). Only set if you hit `ReadTimeout` during collection operations |
| `MEM0_COLLECTION` | `mem0_mcp_selfhosted` | Qdrant collection name |

### Graph Store (Neo4j)

| Variable | Default | Description |
|----------|---------|-------------|
| `MEM0_ENABLE_GRAPH` | `false` | Enable graph memory (entity extraction to Neo4j) |
| `MEM0_NEO4J_URL` | `bolt://127.0.0.1:7687` | Neo4j Bolt endpoint |
| `MEM0_NEO4J_USER` | `neo4j` | Neo4j username |
| `MEM0_NEO4J_PASSWORD` | `mem0graph` | Neo4j password |
| `MEM0_NEO4J_DATABASE` | -- | Neo4j database name (multi-database setups) |
| `MEM0_NEO4J_BASE_LABEL` | -- | Custom Neo4j base label for node type grouping |
| `MEM0_GRAPH_THRESHOLD` | `0.7` | Embedding similarity threshold for node matching |

### Server

| Variable | Default | Description |
|----------|---------|-------------|
| `MEM0_TRANSPORT` | `stdio` | Transport: `stdio`, `sse`, or `streamable-http` |
| `MEM0_HOST` | `0.0.0.0` | Host for SSE/HTTP transports |
| `MEM0_PORT` | `8081` | Port for SSE/HTTP transports |
| `MEM0_USER_ID` | `user` | Default user ID for memory scoping |
| `MEM0_LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `MEM0_HISTORY_DB_PATH` | -- | SQLite path for memory change history |

## Architecture

```
Claude Code
  |
  └── MCP stdio/SSE/streamable-http
        |
        ├── auth.py              ← Hybrid token fallback chain
        ├── llm_anthropic.py     ← Custom Anthropic LLM provider (OAT + structured outputs)
        ├── llm_ollama.py        ← Custom Ollama LLM provider (restored tool-calling)
        ├── config.py            ← Env vars → MemoryConfig dict
        ├── helpers.py           ← Error wrapper, concurrency lock, safe bulk-delete
        ├── graph_tools.py       ← Direct Neo4j Cypher queries (lazy driver)
        ├── llm_router.py        ← Split-model graph LLM router (gemini_split)
        └── server.py            ← FastMCP orchestrator (11 tools + prompt)
              |
              ├── mem0ai Memory class
              │     ├── Vector: LLM fact extraction → Ollama embed → Qdrant
              │     └── Graph: LLM entity extraction (tool calls) → Neo4j
              |
              └── Infrastructure
                    ├── Qdrant          ← Vector store
                    ├── Ollama          ← Embeddings
                    ├── Neo4j           ← Knowledge graph (optional)
                    └── Anthropic/Ollama ← Main LLM (configurable)
```

## Graph Memory & Quota

Graph memory is **disabled by default** (`MEM0_ENABLE_GRAPH=false`) to protect your Claude quota. Each `add_memory` with graph enabled triggers 3 additional LLM calls for entity extraction, relationship generation, and conflict resolution.

### Using Ollama for Graph Operations

To eliminate Claude quota usage for graph ops, use a local Ollama model:

```env
MEM0_ENABLE_GRAPH=true
MEM0_GRAPH_LLM_PROVIDER=ollama
MEM0_GRAPH_LLM_MODEL=qwen3:14b
```

Qwen3:14b has 0.971 tool-calling F1 (nearly matching GPT-4's 0.974) and runs in ~7-8GB VRAM with Q4_K_M quantization.

### Using Gemini for Graph Operations

Google's Gemini 2.5 Flash Lite is the cheapest option for graph ops while maintaining strong entity extraction accuracy:

```env
MEM0_ENABLE_GRAPH=true
MEM0_GRAPH_LLM_PROVIDER=gemini
MEM0_GRAPH_LLM_MODEL=gemini-2.5-flash-lite
GOOGLE_API_KEY=your-google-api-key
```

### Using Split-Model for Best Accuracy

The `gemini_split` provider routes graph pipeline calls to different LLMs based on the operation. Entity extraction (Calls 1 & 2) goes to Gemini for speed and cost; contradiction detection (Call 3) goes to Claude for accuracy.

```env
MEM0_ENABLE_GRAPH=true
MEM0_GRAPH_LLM_PROVIDER=gemini_split
GOOGLE_API_KEY=your-google-api-key
MEM0_GRAPH_CONTRADICTION_LLM_PROVIDER=anthropic
MEM0_GRAPH_CONTRADICTION_LLM_MODEL=claude-opus-4-6
```

Benchmark results across 248 test cases: Gemini scores 85.4% on entity extraction (vs Claude's 79.1%), while Claude scores 100% on contradiction detection (vs Gemini's 80%). The split-model combines the best of both.

## Transport Modes

| Mode | Use Case | Config |
|------|----------|--------|
| `stdio` (default) | Claude Code integration | `MEM0_TRANSPORT=stdio` |
| `sse` | Legacy remote clients | `MEM0_TRANSPORT=sse` |
| `streamable-http` | Modern remote clients | `MEM0_TRANSPORT=streamable-http` |

For remote deployments, MCP SDK >= 1.23.0 enables DNS rebinding protection by default.

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run unit tests
python3 -m pytest tests/unit/ -v

# Run contract tests (validates mem0ai internal API assumptions)
python3 -m pytest tests/contract/ -v

# Run integration tests (requires live Qdrant + Neo4j + Ollama)
python3 -m pytest tests/integration/ -v

# Run all tests
python3 -m pytest tests/ -v
```

### Test Structure

- **`tests/unit/`** -- Pure unit tests with mocked dependencies (auth, config, helpers, LLM provider, graph tools, LLM router)
- **`tests/contract/`** -- Validates assumptions about mem0ai internals (schema detection invariant, `vector_store.client` access path, `LlmFactory` registration idempotency)
- **`tests/integration/`** -- Live infrastructure tests (memory lifecycle, graph ops, bulk operations) against real Qdrant + Neo4j + Ollama. Marked with `@pytest.mark.integration`.

Contract tests catch breaking changes in `mem0ai` upgrades before they reach production.

## Telemetry

All mem0ai telemetry is suppressed. `os.environ["MEM0_TELEMETRY"] = "false"` is set at package import time, before any `mem0` module is loaded. No PostHog events are sent.

## License

MIT
