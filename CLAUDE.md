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
- `helpers.py` — `_mem0_call()` error wrapper, `call_with_graph()` threading lock for per-call graph toggle, `safe_bulk_delete()` iterates+deletes individually (never calls `memory.delete_all()`), `gc_orphan_graph_nodes()` hard-deletes Neo4j orphans left by `Memory.delete()`'s edge soft-delete, `patch_graph_sanitizer()` monkey-patches mem0ai's relationship sanitizer for Neo4j compliance
- `graph_tools.py` — Direct Neo4j Cypher queries with lazy driver init
- `__init__.py` — Suppresses mem0ai telemetry before any imports

**Critical implementation details:**
- `memory.delete()` does NOT clean Neo4j nodes (mem0ai bug #3245) — `safe_bulk_delete()` explicitly calls `memory.graph.delete_all(filters)` after
- `memory.enable_graph` is mutable instance state — `call_with_graph()` holds a `threading.Lock` for the full duration of each Memory call (2-20s)
- Contract tests (`tests/contract/`) validate mem0ai internal API assumptions — if these fail after a mem0ai upgrade, the code needs updating
- `Memory.update()` uses `data=` parameter, not `text=`
- Structured output support requires claude-opus-4/sonnet-4/haiku-4 models; older models fall back to JSON extraction
- mem0ai's `sanitize_relationship_for_cypher()` has gaps (no hyphen handling, no leading-digit check) — `patch_graph_sanitizer()` wraps it at startup to ensure all relationship types match `^[a-zA-Z_][a-zA-Z0-9_]*$`
- `patch_extract_relations_prompt()` PREPENDS failure-mode rules (compound-entity preservation, direct org→thing edges, direct person→person via artifact, no self-loops, direction discipline) to the HEAD of mem0ai's `EXTRACT_RELATIONS_PROMPT`. Tail-appended rules were silently ignored by Haiku-class extraction LLMs (e.g. self-loop on `owned_by` despite a literal counter-example in the rules). Opt out with `MEM0_GRAPH_PROMPT_AUGMENT=false`. Patches both `mem0.graphs.utils` and the four importer modules (`graph_memory`, `memgraph_memory`, `kuzu_memory`, `apache_age_memory`).
- `patch_graph_entity_extraction()` wraps `MemoryGraph._retrieve_nodes_from_data` to inject compound-entity preservation into its inline system prompt. Without it, `Memory.delete()`'s graph cascade silently leaks edges: at add time the LLM keeps "AWS us-east-1" whole, but at delete time the unaugmented entity prompt may split it into `aws` + `us-east-1`, so the relation re-extraction can't recreate the matching triple and `_delete_entities` finds nothing to soft-delete. Same opt-out flag as above.
- `delete_memory` (single) and `delete_all_memories` (bulk) accept `enable_graph` and route through `call_with_graph()` — required because `Memory.delete()` only cleans Neo4j when `self.enable_graph` is True at call time, and that's mutable instance state shared across threads.
- `gc_orphan_graph_nodes()` runs after `Memory.delete()` in the per-memory `delete_memory` path. mem0ai's `_delete_entities` only *soft*-deletes edges (`r.valid = false`) for temporal reasoning; the endpoint nodes linger and leak through `find_entity`/`get_entity`. The GC scopes a Cypher pass to the deleted memory's `user_id`/`agent_id`/`run_id`, hard-deletes any node that (a) has no remaining valid (`valid IS NULL OR valid = true`) incident edges AND (b) had at least one edge invalidated within the recency window (default 30s) — proving *this* delete just orphaned the node. The recency check is the safety rail that prevents wiping pre-existing nodes whose edges were soft-deleted earlier for unrelated temporal-reasoning reasons. The bulk path doesn't need this because `safe_bulk_delete` already calls `graph.delete_all(filters)`, which is HARD `DETACH DELETE` scoped by user/agent/run.

## Post-restart MCP smoke test

After restarting Claude Code (which respawns the mem0 MCP with the latest `~/.claude.json` env), run these against the live MCP from any session. Each step has an expected outcome; deviations point to a specific subsystem.

### 1. Tool naming
Verify the tool list contains `find_entity` and `get_entity`. The pre-rename names `mcp_search_graph` and `mcp_get_entity` must be absent. (If they're still there, MCP did not restart — fully quit Claude Code.)

### 2. New LLM + augmented prompt (one combined check)

The smoke-test surface forms must NOT appear verbatim in `helpers.py:_AUGMENT_RULES` or `_ENTITY_EXTRACT_AUGMENT` — otherwise we're testing whether the LLM regurgitates the prompt, not whether the rules generalize. Current safe names: BlueOcean, Acme, Priya, `cloud-region-7`. If you change `_AUGMENT_RULES`, audit this step.

Call `add_memory` with text `"SmokeTest: BlueOcean runs in Acme cloud-region-7. Priya leads BlueOcean at Acme."`, `run_id="smoke-test"`, `enable_graph=true`. Then call `find_entity("acme")`. Expect:
- A node literally named `acme cloud-region-7` (proves compound-entity preservation rule generalizes to a region string the prompt never mentions).
- An edge along the lines of `blueocean --[owned_by]--> acme` (proves the org→thing rule generalizes).
- No self-loop edge with source == target (proves rule A holds).

If `acme` and `cloud-region-7` come back as separate nodes: augmented prompt didn't fire. Check `MEM0_GRAPH_PROMPT_AUGMENT` is not set to `false` in `~/.claude.json` and restart again.

### 3. delete_memory cleans graph (regression check for the race-fix + orphan GC)
Step 2 returns two memory IDs (one per fact extracted). Call `delete_memory(memory_id=..., enable_graph=true)` for the first ID. After it returns, `find_entity("acme")` should still show `acme` and `blueocean` (the second memory still references them) but the orphan `acme_cloud-region-7` should be gone — proving the per-memory cascade soft-deleted the `runs_in` edge AND `gc_orphan_graph_nodes()` hard-deleted the now-orphaned region node. Call `delete_memory` for the second ID. Then `find_entity("blueocean")`, `find_entity("acme")`, `find_entity("priya")` should all return empty `entities`.

If after the second delete any of those entities remain, the GC's recency window (30s) likely didn't fire — check the helper logs for `"GC: hard-deleted N orphan graph node(s)"`. If it never logs, mem0ai's cascade didn't soft-delete the matching edges (i.e. delete-time re-extraction missed them), which is a prompt-augmentation regression, not a GC bug.

### 4. Cleanup (only needed if step 3 leaks)
`delete_all_memories(run_id="smoke-test", enable_graph=true)`. With step 3 passing this is a no-op (zero memories left, zero graph nodes to sweep). If anything lingered: `list_entities` should no longer show `smoke-test` under `runs`. Graph nodes from mem0ai's pipeline are scoped by `user_id`/`agent_id`/`run_id`, so the bulk path's `graph.delete_all(filters)` does `DETACH DELETE` everything in scope. As a last resort sweep by name: `MATCH (n) WHERE n.name IN ['blueocean','acme','acme_cloud-region-7','priya'] DETACH DELETE n` (note: mem0ai's sanitizer normalizes the space in `acme cloud-region-7` to an underscore).
