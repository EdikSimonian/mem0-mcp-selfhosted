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
- `helpers.py` — `_mem0_call()` error wrapper, `call_with_graph()` threading lock for per-call graph toggle, `safe_bulk_delete()` iterates+deletes individually (never calls `memory.delete_all()`), `add_with_batch_provenance()` + `delete_memory_with_batch()` + `hard_delete_batch()` implement the batch_uuid provenance design (primary single-memory delete cascade), `gc_orphan_graph_nodes()` is the legacy fallback for memories without a batch_uuid, `patch_graph_sanitizer()` monkey-patches mem0ai's relationship sanitizer for Neo4j compliance
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
- `delete_memory` (single) routes through `delete_memory_with_batch()`, which dispatches to the batch_uuid hard-delete path or the legacy `gc_orphan` path based on whether the vector payload carries a `batch_uuid`. `delete_all_memories` (bulk) routes through `safe_bulk_delete()` whose `graph.delete_all(filters)` is already HARD `DETACH DELETE` scoped by user/agent/run, so the batch_uuid layer is unnecessary there. Both paths handle the `enable_graph` toggle internally under `_graph_lock` (mutable `memory.enable_graph` is shared across threads — the lock spans the full call so concurrent requests can't flip it mid-cycle).
- **Batch_uuid provenance (primary delete cascade):** `add_with_batch_provenance()` mints a `batch_uuid` per `add_memory` call, stores it in every vector payload via `metadata`, and Cypher-tags all graph nodes/edges from that call (Pass A by triple identity + timestamp window + scope; Pass B backfills endpoint nodes because mem0ai's `_add_entities` only sets `created` on NEW nodes, not on endpoints of edges between existing nodes). On delete, `delete_memory_with_batch()` runs the inner `mem.delete()` with `enable_graph=False` to skip mem0ai's re-extraction cascade, counts remaining vector anchors carrying that `batch_uuid` in scope, and calls `hard_delete_batch()` only when anchors hit zero — so multi-memory batches preserve their graph until the last anchor is deleted. Orphan-from-birth (graph relations created but vector path produced no anchors) is reclaimed at add time. Replaces re-extraction-based cleanup, which was non-deterministic on abstract / code-heavy text. Surface contracts pinned in `tests/contract/test_provenance_invariants.py`.
- `gc_orphan_graph_nodes()` (legacy fallback) — runs only when `delete_memory_with_batch()` falls through to the legacy path because the deleted memory carries no `batch_uuid` (predates the rollout, or had `enable_graph=False` at add time). mem0ai's `_delete_entities` only *soft*-deletes edges (`r.valid = false`) for temporal reasoning; the endpoint nodes linger and leak through `find_entity`/`get_entity`. The GC scopes a Cypher pass to the deleted memory's `user_id`/`agent_id`/`run_id`, hard-deletes any node that (a) has no remaining valid (`valid IS NULL OR valid = true`) incident edges AND (b) had at least one edge invalidated within the recency window (default 30s) — proving *this* delete just orphaned the node. The recency check prevents wiping pre-existing nodes whose edges were soft-deleted earlier for unrelated temporal-reasoning reasons.

## Post-restart MCP smoke test

After restarting Claude Code (which respawns the mem0 MCP with the latest `~/.claude.json` env), run these against the live MCP from any session. Each step has an expected outcome; deviations point to a specific subsystem.

### 1. Tool naming
Verify the tool list contains `find_entity` and `get_entity`. The pre-rename names `mcp_search_graph` and `mcp_get_entity` must be absent. (If they're still there, MCP did not restart — fully quit Claude Code.)

### 2. New LLM + augmented prompt (one combined check)

The smoke-test surface forms must NOT appear verbatim in `helpers.py:_AUGMENT_RULES` or `_ENTITY_EXTRACT_AUGMENT` — otherwise we're testing whether the LLM regurgitates the prompt, not whether the rules generalize. Current safe names: BlueOcean, Acme, Priya, `cloud-region-7`. If you change `_AUGMENT_RULES`, audit this step.

Call `add_memory` with text `"BlueOcean runs in Acme cloud-region-7. Priya leads BlueOcean at Acme."`, `run_id="smoke-test"`, `enable_graph=true`. Then call `find_entity("acme")`. Expect:
- A node literally named `acme cloud-region-7` (proves compound-entity preservation rule generalizes to a region string the prompt never mentions).
- An edge along the lines of `blueocean --[owned_by]--> acme` (proves the org→thing rule generalizes).
- No self-loop edge with source == target (proves rule A holds).

If `acme` and `cloud-region-7` come back as separate nodes: augmented prompt didn't fire. Check `MEM0_GRAPH_PROMPT_AUGMENT` is not set to `false` in `~/.claude.json` and restart again.

### 2b. Single-memory delete cleans graph (batch_uuid hard-delete, regression check)

This is the original failure mode the batch_uuid design was built for: under the old `gc_orphan_graph_nodes` path, mem0ai's delete-time re-extraction would drift on abstract input and miss the soft-delete, leaking endpoint nodes. Under the new design, every node/edge `_add_entities` creates is Cypher-tagged with the call's `batch_uuid` (also stored in vector payload), and `delete_memory_with_batch` hard-deletes the entire tagged batch deterministically once the last anchor memory is gone — bypassing re-extraction entirely.

Call `add_memory` with text `"orion_service depends on polaris_module which is maintained by team_vega."`, `run_id="smoke-test-single"`, `enable_graph=true`. The fact extractor produces 3 vector memories; the graph extractor produces 2 relations (`orion_service --[depends_on]--> polaris_module` and `polaris_module --[maintained_by]--> team_vega`). Note every memory ID. Then call `delete_memory(memory_id=..., enable_graph=true)` for each ID in turn:

- After deletes #1 and #2: `find_entity("orion")` should still return `orion_service` (anchor count > 0).
- After delete #3 (last anchor): `hard_delete_batch` fires. `find_entity("orion")`, `find_entity("polaris")`, `find_entity("team_vega")` should all return empty `entities`. `list_entities` should not show `smoke-test-single` under `runs`.

The hard-delete is keyed by Cypher tag, not by re-extracted triple identity, so it works regardless of whether the delete-time entity prompt would match. Surface forms (`orion_service`, `polaris_module`, `team_vega`) deliberately avoid the `_AUGMENT_RULES` exemplars.

Variant — orphan-from-birth: text like `"helpers.py exposes rule_a which augments rule_b in graph_tools.py."` makes the fact extractor return zero memories while the graph extractor still creates relations. Surface forms here deliberately avoid colon-prefixed labels (`Note:`, `SmokeTest:`, `ProbeE3:`) — Haiku-class relation extractors parse short leading colon-prefix labels as the syntactic subject of the sentence body and emit them as edge sources, contaminating triples. Adding this exercises the *add-time* reclaim path: `add_with_batch_provenance` detects 0 anchors with that batch_uuid and calls `hard_delete_batch` immediately. Expect `add_memory` to return `"results": []`, and `find_entity("rule_a")` to return empty straight away.

### 3. Multi-memory batch defers cleanup to last-anchor delete (batch_uuid contract)

Step 2 stored two memories from a single `add_memory` call → both share one `batch_uuid`. The new contract: graph cleanup defers until every anchor memory in the batch is deleted, because the `batch_uuid` stays alive in any remaining vector payload.

Call `delete_memory(memory_id=..., enable_graph=true)` for the first ID. The graph should be **unchanged**: `find_entity("acme")` still returns `acme` AND `acme_cloud-region-7`, `find_entity("blueocean")` still shows `blueocean` with all three edges intact. (This is intentional and different from the legacy `gc_orphan` path — do not treat surviving nodes here as a regression.)

Call `delete_memory` for the second ID. The anchor count for that `batch_uuid` now hits zero and `hard_delete_batch` fires. After this delete, `find_entity("blueocean")`, `find_entity("acme")`, `find_entity("priya")` should all return empty `entities`.

If entities remain after the second delete, check helper logs for `"Hard-deleted batch <uuid>: N relation(s), M node(s)"`. If it never logs, either:
- **Add-time tagging missed**: Pass A returned 0 — likely a sanitizer/normalization mismatch between mem0ai's `added_entities` shape and stored Neo4j `type(r)`. Inspect `tests/contract/test_provenance_invariants.py` for the surface contract.
- **Delete fell through to legacy path**: `batch_uuid` was missing from the vector payload (memory predates the rollout, or `enable_graph=False` at add time, or `_create_memory` didn't deepcopy metadata as expected). The legacy `gc_orphan_graph_nodes` then runs and may not catch all leaks on its own.

### 4. Cleanup (only needed if 2b or 3 leaks)
`delete_all_memories(run_id="smoke-test", enable_graph=true)` and `delete_all_memories(run_id="smoke-test-single", enable_graph=true)`. With 2b and 3 passing these are no-ops. If anything lingered: `list_entities` should no longer show either run. The bulk path's `safe_bulk_delete` calls `graph.delete_all(filters)` which does `DETACH DELETE` for everything in scope, so it cleans both batch_uuid-tagged and legacy nodes. Last-resort sweep by name: `MATCH (n) WHERE n.name IN ['blueocean','acme','acme_cloud-region-7','priya','helpers.py','rule_a','rule_b','graph_tools.py'] DETACH DELETE n` (note: mem0ai's sanitizer normalizes spaces and dots to underscores, so the actual stored names use `acme_cloud-region-7`, `helpers_py`, `graph_tools_py`).
