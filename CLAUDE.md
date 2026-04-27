# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## MCP Servers

- **mem0**: Persistent memory across sessions. At the start of each session, `search_memories` for relevant context before asking the user to re-explain anything. Use `add_memory` whenever you discover project architecture, coding conventions, debugging insights, key decisions, or user preferences. Use `update_memory` when prior context changes. Save information like: "This project uses PostgreSQL with Prisma", "Tests run with pytest -v", "Auth uses JWT validated in middleware". When in doubt, save it — future sessions benefit from over-remembering.

## Build & Test Commands

```bash
pip install -e ".[dev]"              # Install with dev dependencies
python3 -m pytest tests/unit/ -v     # Unit tests (mocked, no infra needed)
python3 -m pytest tests/contract/ -v # Contract tests (validates mem0ai internals)
python3 -m pytest tests/integration/ -v  # Integration tests (requires live Qdrant + Neo4j + Ollama)
python3 -m pytest tests/ -v          # All tests
python3 -m pytest tests/ -m "not integration" -v  # Skip integration
python3 -m pytest tests/unit/test_auth.py::TestIsOatToken -v  # Single test class
python3 -m pytest tests/unit/test_auth.py::TestIsOatToken::test_oat_token_detected -v  # Single test
```

## Architecture

Self-hosted MCP server using `mem0ai` as a library. 11 tools (9 memory + 2 graph), FastMCP orchestrator.

**Module roles:**
- `server.py` — FastMCP orchestrator, registers all tools + `memory_assistant` prompt
- `config.py` — Env vars → mem0ai `MemoryConfig` dict, handles all 5 graph LLM provider configs
- `auth.py` — 3-tier token fallback: `MEM0_ANTHROPIC_TOKEN` → `~/.claude/.credentials.json` → `ANTHROPIC_API_KEY`
- `llm_anthropic.py` — Custom Anthropic provider registered with mem0ai's `LlmFactory`; handles OAT headers, structured outputs (JSON schema via `output_config`), and tool-call parsing
- `llm_router.py` — `SplitModelGraphLLM` routes by tool name: extraction tools → Gemini, contradiction tools → Claude
- `helpers.py` — `_mem0_call()` error wrapper, `call_with_graph()` threading lock for per-call graph toggle, `safe_bulk_delete()` iterates+deletes individually (never calls `memory.delete_all()`), `patch_graph_sanitizer()` monkey-patches mem0ai's relationship sanitizer for Neo4j compliance
- `graph_tools.py` — Direct Neo4j Cypher queries with lazy driver init
- `__init__.py` — Suppresses mem0ai telemetry before any imports

**Critical implementation details:**
- `memory.delete()` does NOT clean Neo4j nodes (mem0ai bug #3245) — `safe_bulk_delete()` explicitly calls `memory.graph.delete_all(filters)` after
- `memory.enable_graph` is mutable instance state — `call_with_graph()` holds a `threading.Lock` for the full duration of each Memory call (2-20s)
- Contract tests (`tests/contract/`) validate mem0ai internal API assumptions — if these fail after a mem0ai upgrade, the code needs updating
- `Memory.update()` uses `data=` parameter, not `text=`
- Structured output support requires claude-opus-4/sonnet-4/haiku-4 models; older models fall back to JSON extraction
- mem0ai's `sanitize_relationship_for_cypher()` has gaps (no hyphen handling, no leading-digit check) — `patch_graph_sanitizer()` wraps it at startup to ensure all relationship types match `^[a-zA-Z_][a-zA-Z0-9_]*$`
- `patch_extract_relations_prompt()` appends failure-mode rules (compound-entity preservation, direct org→thing edges, direct person→person via artifact, no self-loops, direction discipline) to mem0ai's `EXTRACT_RELATIONS_PROMPT`. Opt out with `MEM0_GRAPH_PROMPT_AUGMENT=false`. Patches both `mem0.graphs.utils` and the four importer modules (`graph_memory`, `memgraph_memory`, `kuzu_memory`, `apache_age_memory`).
- `delete_memory` (single) and `delete_all_memories` (bulk) accept `enable_graph` and route through `call_with_graph()` — required because `Memory.delete()` only cleans Neo4j when `self.enable_graph` is True at call time, and that's mutable instance state shared across threads.

## Post-restart MCP smoke test

After restarting Claude Code (which respawns the mem0 MCP with the latest `~/.claude.json` env), run these against the live MCP from any session. Each step has an expected outcome; deviations point to a specific subsystem.

### 1. Tool naming
Verify the tool list contains `find_entity` and `get_entity`. The pre-rename names `mcp_search_graph` and `mcp_get_entity` must be absent. (If they're still there, MCP did not restart — fully quit Claude Code.)

### 2. New LLM + augmented prompt (one combined check)
Call `add_memory` with text `"SmokeTest: TestService runs on AWS us-east-1. Maya leads TestService at TestCorp."`, `run_id="smoke-test"`, `enable_graph=true`. Then call `find_entity("aws")`. Expect:
- A node literally named `aws us-east-1` (proves compound-entity preservation rule fires AND the new LLM is in use — the old gemma4:e2b would have split this into `aws` + `us east 1`).
- An edge along the lines of `testservice --[owned_by]--> testcorp` (proves the org→thing rule).

If `aws` and `us east 1` come back as separate nodes: augmented prompt didn't fire. Check `MEM0_GRAPH_PROMPT_AUGMENT` is not set to `false` in `~/.claude.json` and restart again.

### 3. delete_memory cleans graph (regression check for the race-fix)
Take the `memory_id` returned by step 2 and call `delete_memory(memory_id=..., enable_graph=true)`. Then `find_entity("testservice")`. Expect empty `entities` — the per-memory cascade should remove both the vector entry and the graph nodes that came from this memory's text.

### 4. Cleanup
`delete_all_memories(run_id="smoke-test", enable_graph=true)`. Then `list_entities` should no longer show `smoke-test` under `runs`. If any nodes linger in Neo4j, run a direct cypher: `MATCH (n {run_id: 'smoke-test'}) DETACH DELETE n`.
