"""Shared utilities for mem0-mcp-selfhosted.

- patch_graph_sanitizer(): Monkey-patches mem0ai's relationship sanitizer for Neo4j compliance
- _mem0_call(): Error wrapper for all mem0ai calls
- call_with_graph(): Concurrency-safe enable_graph toggle
- safe_bulk_delete(): Iterate + individual delete (never memory.delete_all())
- get_default_user_id(): Default user_id injection
- list_entities_facet(): Qdrant Facet API entity listing with scroll fallback
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, Callable

from mem0_mcp_selfhosted.env import env

logger = logging.getLogger(__name__)

# Valid Neo4j relationship type: must start with a letter or underscore,
# followed by letters, digits, or underscores.
_NEO4J_VALID_TYPE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _make_enhanced_sanitizer(original_fn: Callable[[str], str]) -> Callable[[str], str]:
    """Wrap mem0ai's sanitize_relationship_for_cypher with Neo4j compliance fixes.

    Fixes two gaps in the upstream sanitizer:
    1. Hyphens and other ASCII characters not in the char_map
    2. Leading digits (Neo4j types must start with a letter or underscore)

    The wrapper calls the original first (preserving its 26+ special character
    mappings), then applies additional fixes.
    """

    def enhanced(relationship: str) -> str:
        # Run the original sanitizer first
        sanitized = original_fn(relationship)

        # Fix: replace hyphens (not in upstream char_map) with underscores
        sanitized = sanitized.replace("-", "_")

        # Fix: strip any remaining non-alphanumeric/underscore characters
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", sanitized)

        # Collapse consecutive underscores and strip edges
        sanitized = re.sub(r"_+", "_", sanitized).strip("_")

        # Fix: leading digit → prepend 'rel_' prefix
        if sanitized and sanitized[0].isdigit():
            sanitized = "rel_" + sanitized

        # Fallback for empty result
        if not sanitized:
            sanitized = "related_to"

        return sanitized

    return enhanced


def patch_graph_sanitizer() -> None:
    """Monkey-patch mem0ai's relationship sanitizer for full Neo4j compliance.

    Must be called AFTER mem0 modules are imported but BEFORE Memory.from_config().
    Patches both the utils module and the already-imported references in
    graph_memory/memgraph_memory.
    """
    import mem0.memory.utils as utils_module

    original = utils_module.sanitize_relationship_for_cypher
    enhanced = _make_enhanced_sanitizer(original)

    # Patch the source module
    utils_module.sanitize_relationship_for_cypher = enhanced

    # Patch already-imported references (from ... import creates local bindings)
    try:
        import mem0.memory.graph_memory as graph_module

        graph_module.sanitize_relationship_for_cypher = enhanced
    except (ImportError, AttributeError):
        pass

    try:
        import mem0.memory.memgraph_memory as memgraph_module

        memgraph_module.sanitize_relationship_for_cypher = enhanced
    except (ImportError, AttributeError):
        pass

    logger.info("Patched mem0ai relationship sanitizer for Neo4j compliance")


# Augmentation rules appended to mem0ai's EXTRACT_RELATIONS_PROMPT to fix
# four failure modes measured against gemma4:e2b and gpt-5.4-nano-tier models
# in benchmarks/failure_modes_local.py:
#   F1 (compound entities): 0% → 100%
#   F2 (org→thing edges): 33% → 67-100% (model-dependent)
#   F3 (person→person via artifact): 50% → 75-100%
#   F4 (sub-unit distinct): already 100%
_AUGMENT_RULES = """

Additional extraction rules — read carefully:

A. Compound technical identifiers are ONE entity, never two:
   - "AWS us-east-1" → entity "aws us-east-1" (NOT "aws" + "us-east-1")
   - "GCP europe-west4" → entity "gcp europe-west4"
   - Cloud regions, fully-qualified service names, version numbers, model
     IDs, and dotted/dashed identifiers must stay intact as one node.

B. When a person performs an action on or with another person via an
   intermediary (artifact, team, group, project), create a DIRECT
   person-to-person edge in addition to any artifact-related edges:
   - "Maya's team handed off the bridge to David's group"
     → also emit (maya, handed_off_to, david)
   - "Sarah transferred the service to Tom"
     → emit (sarah, transferred_to, tom) directly
   The direct interpersonal edge is what queries about responsibility
   transfer will look for.

C. When an organization owns, runs, or hosts a platform/service/product,
   emit a DIRECT edge from the organization to the platform — not only
   via a person:
   - "Maya leads the Helios team at Northwind" → also emit
     (helios, owned_by, northwind) — the org→platform link is load-bearing.
   - "Stripe's checkout service" → emit (stripe, owns, checkout_service)
     with that exact direction (org → service, not service → org).

D. Sub-teams and organizational units (e.g. "Helios platform team",
   "infrastructure squad") are distinct entities from their parent.
   Both should appear as nodes, with an edge connecting them.

E. Never emit self-referential edges (source equal to destination).
   "X --[leads]--> X" is always wrong — drop the relation.

F. Direction matters. "A is head of B" means (a, head_of, b), not
   (b, head_of, a). "X reports to Y" means (x, reports_to, y).
"""


def patch_extract_relations_prompt() -> None:
    """Append augmentation rules to mem0ai's EXTRACT_RELATIONS_PROMPT.

    Opt-out via MEM0_GRAPH_PROMPT_AUGMENT=false (default: on). Safe to call
    multiple times — the augmentation is idempotent (sentinel-checked).

    Patches both the source module and all four importer modules
    (graph_memory, memgraph_memory, kuzu_memory, apache_age_memory) since
    `from ... import EXTRACT_RELATIONS_PROMPT` creates local bindings.

    Must be called AFTER mem0 modules are imported but BEFORE
    Memory.from_config().
    """
    if env("MEM0_GRAPH_PROMPT_AUGMENT", "true").lower() in ("false", "0", "no"):
        logger.info("Skipping EXTRACT_RELATIONS_PROMPT augmentation (opt-out)")
        return

    import mem0.graphs.utils as utils_module

    sentinel = "Additional extraction rules — read carefully:"
    if sentinel in utils_module.EXTRACT_RELATIONS_PROMPT:
        return  # Already patched (idempotent on re-imports)

    augmented = utils_module.EXTRACT_RELATIONS_PROMPT + _AUGMENT_RULES
    utils_module.EXTRACT_RELATIONS_PROMPT = augmented

    # Patch already-imported references in each graph backend module
    for mod_path in (
        "mem0.memory.graph_memory",
        "mem0.memory.memgraph_memory",
        "mem0.memory.kuzu_memory",
        "mem0.memory.apache_age_memory",
    ):
        try:
            import importlib

            mod = importlib.import_module(mod_path)
            if hasattr(mod, "EXTRACT_RELATIONS_PROMPT"):
                mod.EXTRACT_RELATIONS_PROMPT = augmented
        except (ImportError, AttributeError):
            pass

    logger.info("Patched mem0ai EXTRACT_RELATIONS_PROMPT with failure-mode augmentation")


def patch_gemini_parse_response() -> None:
    """Monkey-patch mem0ai's GeminiLLM to guard against null content responses.

    The upstream ``GeminiLLM._parse_response`` accesses
    ``response.candidates[0].content.parts`` without checking that ``.content``
    is not ``None``.  When the Gemini API returns a candidate with null content
    (safety block, empty response, transient error), this raises
    ``AttributeError: 'NoneType' object has no attribute 'parts'``.

    Must be called AFTER mem0 modules are imported but BEFORE Memory.from_config().
    """
    try:
        from mem0.llms.gemini import GeminiLLM
    except ImportError:
        logger.debug("mem0.llms.gemini not available — skipping Gemini null guard patch")
        return

    original = getattr(GeminiLLM, "_parse_response", None)
    if original is None:
        logger.debug("GeminiLLM._parse_response not found — skipping patch")
        return

    def _safe_parse_response(self, response, *args, **kwargs):  # noqa: ANN001
        """Guarded _parse_response that handles null content gracefully."""
        if (
            response.candidates
            and response.candidates[0].content is not None
            and response.candidates[0].content.parts
        ):
            return original(self, response, *args, **kwargs)
        logger.warning("[mem0] Gemini returned null content — returning empty string")
        return ""

    GeminiLLM._parse_response = _safe_parse_response
    logger.info("Patched GeminiLLM._parse_response for null content guard")


# Serializes enable_graph mutation + full Memory method execution.
# Lock hold time is 2-20 seconds (see PRD §2.4).
_graph_lock = threading.Lock()


def get_default_user_id() -> str:
    """Get the default user_id from MEM0_USER_ID env var."""
    return env("MEM0_USER_ID", "user")


def _mem0_call(func: Callable, *args: Any, **kwargs: Any) -> str:
    """Wrap a mem0ai call with structured error handling.

    Returns a JSON string in all cases (success or error).
    """
    try:
        result = func(*args, **kwargs)
    except Exception as exc:
        # Check if it's a MemoryError (imported lazily to avoid import issues)
        exc_type = type(exc).__name__
        is_memory_error = any(
            cls.__name__ == "MemoryError" for cls in type(exc).__mro__
        )
        if is_memory_error:
            logger.error("Mem0 call failed: %s", exc)
            return json.dumps(
                {
                    "error": str(exc),
                    "error_code": getattr(exc, "error_code", None),
                    "details": getattr(exc, "details", None),
                    "suggestion": getattr(exc, "suggestion", None),
                },
                ensure_ascii=False,
            )
        else:
            logger.error("Unexpected error: %s", exc)
            return json.dumps(
                {
                    "error": exc_type,
                    "detail": str(exc),
                },
                ensure_ascii=False,
            )
    return json.dumps(result, ensure_ascii=False)


def call_with_graph(
    memory: Any,
    enable_graph: bool | None,
    default_graph: bool,
    func: Callable,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute a Memory method with per-request enable_graph context.

    Each tool call resolves its own effective enable_graph value and passes
    it here. The lock ensures no concurrent request can observe a stale flag.

    IMPORTANT: The lock is held for the full duration of func() (2-20s),
    because Memory.add() blocks on concurrent.futures.wait() internally.
    """
    if memory is None:
        raise RuntimeError("Memory not initialized. Infrastructure may be unavailable.")
    effective = enable_graph if enable_graph is not None else default_graph
    with _graph_lock:
        memory.enable_graph = effective and memory.graph is not None
        return func(*args, **kwargs)


def safe_bulk_delete(memory: Any, filters: dict[str, Any], *, graph_enabled: bool = False) -> int:
    """Safely delete all memories matching filters.

    NEVER calls memory.delete_all() (which triggers vector_store.reset()).
    Instead: iterate + individual delete + mandatory graph cleanup.

    Args:
        graph_enabled: Explicit graph state from caller (avoids reading
            mutable ``memory.enable_graph`` which races with ``call_with_graph``).

    Returns the count of deleted memories.
    """
    # Get all memories matching the filters
    # Qdrant.list() returns raw scroll result: (records, next_page_offset)
    result = memory.vector_store.list(filters=filters)
    memories = result[0] if isinstance(result, tuple) else result

    count = 0
    for item in memories:
        # Extract memory_id from the Qdrant point
        memory_id = item.id if hasattr(item, "id") else item.get("id") if isinstance(item, dict) else str(item)
        try:
            memory.delete(memory_id)
            count += 1
        except Exception as exc:
            logger.warning("Failed to delete memory %s: %s", memory_id, exc)

    # Mandatory graph cleanup — memory.delete() does NOT clean Neo4j (GitHub #3245)
    if graph_enabled and hasattr(memory, "graph") and memory.graph is not None:
        try:
            memory.graph.delete_all(filters)
        except Exception as exc:
            logger.warning("Graph cleanup failed for filters %s: %s", filters, exc)

    return count


def list_entities_facet(memory: Any) -> dict[str, list[dict]]:
    """List entities using Qdrant Facet API with scroll fallback.

    Primary: Facet API (Qdrant v1.12+) — server-side distinct value aggregation.
    Fallback: scroll+dedupe for older Qdrant versions.

    Returns: {"users": [{"value": ..., "count": ...}], "agents": [...], "runs": [...]}
    """
    client = memory.vector_store.client
    collection = memory.vector_store.collection_name

    result: dict[str, list[dict]] = {"users": [], "agents": [], "runs": []}
    entity_keys = {"users": "user_id", "agents": "agent_id", "runs": "run_id"}

    try:
        for result_key, payload_key in entity_keys.items():
            facet_response = client.facet(
                collection_name=collection,
                key=payload_key,
            )
            result[result_key] = [
                {"value": hit.value, "count": hit.count}
                for hit in facet_response.hits
            ]
        return result
    except Exception as exc:
        # Facet API unavailable — fall back to scroll+dedupe
        logger.warning(
            "Qdrant Facet API unavailable (%s). Falling back to scroll+dedupe. "
            "Upgrade to Qdrant v1.12+ for better performance.",
            exc,
        )
        return _list_entities_scroll_fallback(memory)


def _list_entities_scroll_fallback(memory: Any) -> dict[str, list[dict]]:
    """Fallback entity listing via scroll+dedupe."""
    entities: dict[str, dict[str, int]] = {
        "user_id": {},
        "agent_id": {},
        "run_id": {},
    }

    # Scroll through all memories in batches
    # Qdrant.list() returns raw scroll result: (records, next_page_offset)
    result = memory.vector_store.list(filters={}, limit=500)
    all_memories = result[0] if isinstance(result, tuple) else result
    for item in all_memories:
        payload = item.payload if hasattr(item, "payload") else item
        if isinstance(payload, dict):
            for key in entities:
                val = payload.get(key)
                if val:
                    entities[key][val] = entities[key].get(val, 0) + 1

    return {
        "users": [{"value": v, "count": c} for v, c in entities["user_id"].items()],
        "agents": [{"value": v, "count": c} for v, c in entities["agent_id"].items()],
        "runs": [{"value": v, "count": c} for v, c in entities["run_id"].items()],
    }
