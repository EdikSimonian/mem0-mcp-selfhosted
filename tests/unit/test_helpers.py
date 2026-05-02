"""Tests for helpers.py — error wrapper, call_with_graph, bulk delete, user_id, sanitizer, Gemini patch."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from mem0_mcp_selfhosted.helpers import (
    _flatten_added_entities,
    _make_enhanced_sanitizer,
    _mem0_call,
    add_with_batch_provenance,
    call_with_graph,
    count_batch_anchors,
    gc_orphan_graph_nodes,
    get_default_user_id,
    hard_delete_batch,
    patch_extract_relations_prompt,
    patch_gemini_parse_response,
    safe_bulk_delete,
    tag_batch_provenance,
)


class TestMem0Call:
    def test_success_returns_json(self):
        result = _mem0_call(lambda: {"status": "ok"})
        parsed = json.loads(result)
        assert parsed == {"status": "ok"}

    def test_memory_error_caught(self):
        """MemoryError subclass returns structured error JSON."""

        # Create a mock MemoryError-like exception
        class FakeMemoryError(Exception):
            pass

        FakeMemoryError.__name__ = "MemoryError"

        exc = FakeMemoryError("something failed")
        exc.error_code = "VALIDATION_ERROR"
        exc.details = "missing field"
        exc.suggestion = "add user_id"

        # Patch the MRO check
        def _raise():
            raise exc

        result = _mem0_call(_raise)
        parsed = json.loads(result)
        assert "error" in parsed

    def test_generic_exception_caught(self):
        """Generic Exception returns type name and detail."""

        def _raise():
            raise ValueError("bad input")

        result = _mem0_call(_raise)
        parsed = json.loads(result)
        assert parsed["error"] == "ValueError"
        assert parsed["detail"] == "bad input"

    def test_ensure_ascii_false(self):
        """Non-ASCII characters preserved in output."""
        result = _mem0_call(lambda: {"text": "Alice prefiere TypeScript"})
        assert "prefiere" in result


class TestCallWithGraph:
    def test_sets_enable_graph_true(self):
        memory = MagicMock()
        memory.graph = MagicMock()

        def _check():
            assert memory.enable_graph is True
            return "ok"

        result = call_with_graph(memory, True, False, _check)
        assert result == "ok"

    def test_sets_enable_graph_false(self):
        memory = MagicMock()
        memory.graph = MagicMock()

        def _check():
            assert memory.enable_graph is False
            return "ok"

        result = call_with_graph(memory, False, True, _check)
        assert result == "ok"

    def test_uses_default_when_none(self):
        memory = MagicMock()
        memory.graph = MagicMock()

        def _check():
            assert memory.enable_graph is True
            return "ok"

        result = call_with_graph(memory, None, True, _check)
        assert result == "ok"

    def test_graph_none_forces_false(self):
        """If memory.graph is None, enable_graph stays False regardless."""
        memory = MagicMock()
        memory.graph = None

        def _check():
            assert memory.enable_graph is False
            return "ok"

        result = call_with_graph(memory, True, True, _check)
        assert result == "ok"

    def test_none_memory_raises_runtime_error(self):
        """call_with_graph raises RuntimeError when memory is None."""
        with pytest.raises(RuntimeError, match="Memory not initialized"):
            call_with_graph(None, False, False, lambda: "ok")


def _stub_scroll(memory, pages):
    """Configure memory.vector_store.client.scroll to yield the given pages.

    `pages` is a list of point lists; each call returns the next page paired
    with a non-None offset, except the last which pairs with None to terminate
    the loop in _scroll_all_points.
    """
    returns = []
    for idx, page in enumerate(pages):
        next_offset = f"cursor-{idx}" if idx < len(pages) - 1 else None
        returns.append((page, next_offset))
    if not returns:
        returns = [([], None)]
    memory.vector_store.client.scroll.side_effect = returns


class TestSafeBulkDelete:
    def test_iterates_and_deletes(self):
        memory = MagicMock()
        memory.enable_graph = False
        memory.graph = None

        item1 = MagicMock()
        item1.id = "id-1"
        item2 = MagicMock()
        item2.id = "id-2"
        _stub_scroll(memory, [[item1, item2]])

        count = safe_bulk_delete(memory, {"user_id": "testuser"})

        assert count == 2
        assert memory.delete.call_count == 2
        memory.delete.assert_any_call("id-1")
        memory.delete.assert_any_call("id-2")

    def test_paginates_until_cursor_exhausted(self):
        """Multi-page scroll: every page's points must be deleted, not just the first."""
        memory = MagicMock()
        memory.enable_graph = False
        memory.graph = None

        page1 = [MagicMock(id=f"id-{i}") for i in range(3)]
        page2 = [MagicMock(id=f"id-{i}") for i in range(3, 5)]
        _stub_scroll(memory, [page1, page2])

        count = safe_bulk_delete(memory, {"user_id": "u"})

        assert count == 5
        assert memory.delete.call_count == 5
        assert memory.vector_store.client.scroll.call_count == 2

    def test_graph_cleanup_when_graph_enabled_true(self):
        memory = MagicMock()
        memory.graph = MagicMock()
        _stub_scroll(memory, [])

        safe_bulk_delete(memory, {"user_id": "testuser"}, graph_enabled=True)

        memory.graph.delete_all.assert_called_once_with({"user_id": "testuser"})

    def test_no_graph_cleanup_when_graph_enabled_false(self):
        memory = MagicMock()
        memory.graph = MagicMock()
        _stub_scroll(memory, [])

        safe_bulk_delete(memory, {"user_id": "testuser"}, graph_enabled=False)

        memory.graph.delete_all.assert_not_called()

    def test_no_graph_cleanup_default(self):
        """Default graph_enabled=False skips graph cleanup."""
        memory = MagicMock()
        memory.graph = MagicMock()
        _stub_scroll(memory, [])

        safe_bulk_delete(memory, {"user_id": "testuser"})

        memory.graph.delete_all.assert_not_called()

    def test_sweeps_batch_uuids_before_delete_all(self):
        """Each distinct batch_uuid in payload triggers hard_delete_batch.

        Required because graph nodes can MERGE onto cross-scope existing
        nodes (different run_id/agent_id), so graph.delete_all(filters) by
        run_id misses them. The batch_uuid sweep cleans those up by tag.
        """
        memory = MagicMock()
        memory.graph = MagicMock()
        # Four points across two scroll pages: two share batch_uuid b1, one
        # has b2, one has none. Splitting across pages also proves the
        # sweep observes uuids from every page, not just the first.
        p1 = MagicMock(id="id-1", payload={"batch_uuid": "b1", "user_id": "u"})
        p2 = MagicMock(id="id-2", payload={"batch_uuid": "b1", "user_id": "u"})
        p3 = MagicMock(id="id-3", payload={"batch_uuid": "b2", "user_id": "u"})
        p4 = MagicMock(id="id-4", payload={"user_id": "u"})  # legacy, no batch_uuid
        _stub_scroll(memory, [[p1, p2], [p3, p4]])
        memory.graph.graph.query.return_value = [{"deleted": 0}]

        safe_bulk_delete(memory, {"user_id": "u"}, graph_enabled=True)

        # Each distinct batch_uuid swept exactly once. hard_delete_batch
        # issues 2 queries (relations + nodes) per call, so 2 distinct
        # batch_uuids → 4 queries.
        assert memory.graph.graph.query.call_count == 4
        observed_batch_uuids = {
            call.kwargs["params"]["batch_uuid"]
            for call in memory.graph.graph.query.call_args_list
        }
        assert observed_batch_uuids == {"b1", "b2"}
        # Filter-based fallback still runs.
        memory.graph.delete_all.assert_called_once_with({"user_id": "u"})

    def test_sweep_failure_does_not_block_filter_fallback(self):
        """An exception from hard_delete_batch must not skip graph.delete_all."""
        memory = MagicMock()
        memory.graph = MagicMock()
        p1 = MagicMock(id="id-1", payload={"batch_uuid": "b1", "user_id": "u"})
        _stub_scroll(memory, [[p1]])
        memory.graph.graph.query.side_effect = RuntimeError("neo4j hiccup")

        safe_bulk_delete(memory, {"user_id": "u"}, graph_enabled=True)

        memory.graph.delete_all.assert_called_once_with({"user_id": "u"})


class TestGcOrphanGraphNodes:
    def _build_memory(self, node_label: str = "", deleted: int = 0):
        """MagicMock memory whose graph.graph.query returns deleted-count rows."""
        memory = MagicMock()
        memory.graph = MagicMock()
        memory.graph.node_label = node_label
        memory.graph.graph.query.return_value = [{"deleted": deleted}]
        return memory

    def test_returns_zero_when_graph_unavailable(self):
        memory = MagicMock()
        memory.graph = None
        assert gc_orphan_graph_nodes(memory, {"user_id": "u1"}) == 0

    def test_returns_zero_without_user_id(self):
        memory = self._build_memory()
        # No user_id in filters → must not run query (avoids unbounded scan).
        assert gc_orphan_graph_nodes(memory, {"agent_id": "a1"}) == 0
        memory.graph.graph.query.assert_not_called()

    def test_basic_user_scope_query_and_count(self):
        memory = self._build_memory(deleted=2)
        deleted = gc_orphan_graph_nodes(memory, {"user_id": "eddie"})
        assert deleted == 2
        call = memory.graph.graph.query.call_args
        cypher = call.args[0]
        params = call.kwargs["params"]
        assert params == {"user_id": "eddie", "recency": 30}
        assert "n.user_id = $user_id" in cypher
        assert "DETACH DELETE x" in cypher
        # No agent/run scope when filters omit them
        assert "$agent_id" not in cypher
        assert "$run_id" not in cypher

    def test_agent_and_run_scope_propagated(self):
        memory = self._build_memory(deleted=0)
        gc_orphan_graph_nodes(
            memory,
            {"user_id": "eddie", "agent_id": "agentX", "run_id": "smoke-test"},
            recency_seconds=10,
        )
        call = memory.graph.graph.query.call_args
        cypher = call[0][0]
        params = call.kwargs["params"]
        assert params == {
            "user_id": "eddie",
            "agent_id": "agentX",
            "run_id": "smoke-test",
            "recency": 10,
        }
        assert "n.agent_id = $agent_id" in cypher
        assert "n.run_id = $run_id" in cypher

    def test_node_label_injected(self):
        memory = self._build_memory(node_label=":`__Entity__`", deleted=0)
        gc_orphan_graph_nodes(memory, {"user_id": "eddie"})
        cypher = memory.graph.graph.query.call_args[0][0]
        assert ":`__Entity__`" in cypher

    def test_recency_check_uses_invalidated_at(self):
        memory = self._build_memory(deleted=0)
        gc_orphan_graph_nodes(memory, {"user_id": "eddie"})
        cypher = memory.graph.graph.query.call_args[0][0]
        # Both legs of the WHERE: orphan-check (no valid edges)
        # AND recency-check (at least one edge invalidated within window).
        assert "r.valid IS NULL OR r.valid = true" in cypher
        assert "r2.valid = false" in cypher
        assert "r2.invalidated_at >= datetime() - duration" in cypher

    def test_query_failure_swallowed_returns_zero(self):
        memory = self._build_memory()
        memory.graph.graph.query.side_effect = RuntimeError("neo4j down")
        assert gc_orphan_graph_nodes(memory, {"user_id": "eddie"}) == 0

    def test_empty_result_returns_zero(self):
        memory = self._build_memory()
        memory.graph.graph.query.return_value = []
        assert gc_orphan_graph_nodes(memory, {"user_id": "eddie"}) == 0

    def test_null_deleted_count_returns_zero(self):
        memory = self._build_memory()
        memory.graph.graph.query.return_value = [{"deleted": None}]
        assert gc_orphan_graph_nodes(memory, {"user_id": "eddie"}) == 0


class TestGetDefaultUserId:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("MEM0_USER_ID", raising=False)
        assert get_default_user_id() == "user"

    def test_custom(self, monkeypatch):
        monkeypatch.setenv("MEM0_USER_ID", "bob")
        assert get_default_user_id() == "bob"


class TestEnhancedSanitizer:
    """Tests for the enhanced relationship name sanitizer."""

    @pytest.fixture()
    def sanitize(self):
        """Create enhanced sanitizer wrapping a passthrough original."""
        # Simulate mem0ai's original: returns input unchanged (our tests
        # focus on what the wrapper adds, not on the original's char_map).
        return _make_enhanced_sanitizer(lambda r: r)

    def test_leading_digit_gets_prefix(self, sanitize):
        """Neo4j types can't start with digits — should get rel_ prefix."""
        assert sanitize("3tier_fallback") == "rel_3tier_fallback"

    def test_leading_digit_with_hyphen(self, sanitize):
        """The exact error case: '3-tier_oat_token_fallback'."""
        result = sanitize("3-tier_oat_token_fallback")
        assert result == "rel_3_tier_oat_token_fallback"
        assert result[0].isalpha()

    def test_hyphens_replaced(self, sanitize):
        """Hyphens should become underscores."""
        assert sanitize("has-authored") == "has_authored"

    def test_multiple_hyphens(self, sanitize):
        """Multiple hyphens in a row collapse to single underscore."""
        assert sanitize("is--related--to") == "is_related_to"

    def test_spaces_replaced(self, sanitize):
        """Spaces should become underscores."""
        assert sanitize("author of") == "author_of"

    def test_mixed_special_chars(self, sanitize):
        """Mix of problematic characters."""
        assert sanitize("has.authored-by!user") == "has_authored_by_user"

    def test_already_valid(self, sanitize):
        """Valid relationship type passes through unchanged."""
        assert sanitize("WORKS_FOR") == "WORKS_FOR"

    def test_already_valid_lowercase(self, sanitize):
        """Valid lowercase type passes through unchanged."""
        assert sanitize("prefers") == "prefers"

    def test_empty_string_fallback(self, sanitize):
        """Empty string gets fallback name."""
        assert sanitize("") == "related_to"

    def test_only_special_chars_fallback(self, sanitize):
        """String of only special chars gets fallback name."""
        assert sanitize("---") == "related_to"

    def test_consecutive_underscores_collapsed(self, sanitize):
        """Multiple underscores collapse to one."""
        assert sanitize("foo___bar") == "foo_bar"

    def test_leading_trailing_underscores_stripped(self, sanitize):
        """Leading/trailing underscores are stripped."""
        assert sanitize("_foo_bar_") == "foo_bar"

    def test_pure_digits(self, sanitize):
        """Pure numeric string gets prefix."""
        assert sanitize("123") == "rel_123"

    def test_unicode_stripped(self, sanitize):
        """Non-ASCII characters become underscores then get collapsed/stripped."""
        result = sanitize("関係_type")
        # 関係 → __ → stripped, leaving just "type"
        assert result == "type"

    def test_wraps_original_function(self):
        """Enhanced sanitizer calls the original function first."""
        call_log = []

        def mock_original(r):
            call_log.append(r)
            return r.replace("&", "_ampersand_")

        enhanced = _make_enhanced_sanitizer(mock_original)
        result = enhanced("has&uses")
        assert call_log == ["has&uses"]
        assert result == "has_ampersand_uses"

    def test_valid_neo4j_pattern(self, sanitize):
        """All outputs must match Neo4j's identifier pattern."""
        import re

        pattern = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
        test_cases = [
            "3-tier_oat_token_fallback",
            "has-authored",
            "is professor of",
            "123",
            "---",
            "",
            "WORKS_FOR",
            "has.dot.notation",
            "with spaces and-hyphens",
            "5th_element",
        ]
        for case in test_cases:
            result = sanitize(case)
            assert pattern.match(result), (
                f"'{case}' → '{result}' is not a valid Neo4j type"
            )


class TestPatchGeminiParseResponse:
    """Tests for the Gemini null content guard monkey-patch."""

    def test_null_content_returns_empty_string(self):
        """When Gemini returns candidate with content=None, return empty string."""
        # Create a mock GeminiLLM class with a _parse_response method
        mock_module = MagicMock()
        mock_gemini_cls = MagicMock()
        original_parse = MagicMock(return_value="original result")
        mock_gemini_cls._parse_response = original_parse

        with patch.dict("sys.modules", {"mem0.llms.gemini": mock_module}):
            mock_module.GeminiLLM = mock_gemini_cls

            # Apply the patch
            patch_gemini_parse_response()

            # Verify the method was replaced
            patched_method = mock_gemini_cls._parse_response
            assert patched_method is not original_parse

            # Test with null content
            response = MagicMock()
            candidate = MagicMock()
            candidate.content = None
            response.candidates = [candidate]

            result = patched_method(MagicMock(), response)
            assert result == ""

    def test_normal_response_delegates_to_original(self):
        """Normal responses with valid content delegate to original method."""
        mock_module = MagicMock()
        mock_gemini_cls = MagicMock()
        original_parse = MagicMock(return_value="parsed content")
        mock_gemini_cls._parse_response = original_parse

        with patch.dict("sys.modules", {"mem0.llms.gemini": mock_module}):
            mock_module.GeminiLLM = mock_gemini_cls

            patch_gemini_parse_response()
            patched_method = mock_gemini_cls._parse_response

            # Test with valid content
            response = MagicMock()
            candidate = MagicMock()
            candidate.content = MagicMock()
            candidate.content.parts = [MagicMock()]
            response.candidates = [candidate]

            instance = MagicMock()
            patched_method(instance, response)
            original_parse.assert_called_once_with(instance, response)

    def test_empty_candidates_returns_empty_string(self):
        """Response with empty candidates list returns empty string."""
        mock_module = MagicMock()
        mock_gemini_cls = MagicMock()
        original_parse = MagicMock(return_value="original")
        mock_gemini_cls._parse_response = original_parse

        with patch.dict("sys.modules", {"mem0.llms.gemini": mock_module}):
            mock_module.GeminiLLM = mock_gemini_cls

            patch_gemini_parse_response()
            patched_method = mock_gemini_cls._parse_response

            response = MagicMock()
            response.candidates = []

            result = patched_method(MagicMock(), response)
            assert result == ""


class TestPatchExtractRelationsPrompt:
    """Tests for the EXTRACT_RELATIONS_PROMPT augmentation patch."""

    @pytest.fixture(autouse=True)
    def _restore_prompt(self):
        """Save and restore the original prompt around each test so patches
        from one test don't leak into others."""
        import mem0.graphs.utils as utils_module

        original = utils_module.EXTRACT_RELATIONS_PROMPT
        try:
            yield
        finally:
            utils_module.EXTRACT_RELATIONS_PROMPT = original
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
                        mod.EXTRACT_RELATIONS_PROMPT = original
                except (ImportError, AttributeError):
                    pass

    def test_appends_augmentation_to_source(self, monkeypatch):
        monkeypatch.delenv("MEM0_GRAPH_PROMPT_AUGMENT", raising=False)
        import mem0.graphs.utils as utils_module

        sentinel = "MANDATORY EXTRACTION RULES — apply these BEFORE"
        assert sentinel not in utils_module.EXTRACT_RELATIONS_PROMPT
        patch_extract_relations_prompt()
        assert sentinel in utils_module.EXTRACT_RELATIONS_PROMPT
        # Rules must precede the upstream prompt body (head, not tail)
        rules_idx = utils_module.EXTRACT_RELATIONS_PROMPT.index(sentinel)
        body_idx = utils_module.EXTRACT_RELATIONS_PROMPT.index(
            "You are an advanced algorithm"
        )
        assert rules_idx < body_idx

    def test_propagates_to_graph_memory_module(self, monkeypatch):
        """The bound reference in graph_memory must also be updated, since
        `from ... import` creates local bindings."""
        monkeypatch.delenv("MEM0_GRAPH_PROMPT_AUGMENT", raising=False)
        import mem0.memory.graph_memory as gm

        sentinel = "Compound technical identifiers are ONE entity"
        assert sentinel not in gm.EXTRACT_RELATIONS_PROMPT
        patch_extract_relations_prompt()
        assert sentinel in gm.EXTRACT_RELATIONS_PROMPT

    def test_idempotent_on_second_call(self, monkeypatch):
        """Calling twice must not double-append."""
        monkeypatch.delenv("MEM0_GRAPH_PROMPT_AUGMENT", raising=False)
        import mem0.graphs.utils as utils_module

        patch_extract_relations_prompt()
        once = utils_module.EXTRACT_RELATIONS_PROMPT
        patch_extract_relations_prompt()
        twice = utils_module.EXTRACT_RELATIONS_PROMPT
        assert once == twice

    def test_opt_out_via_env(self, monkeypatch):
        """MEM0_GRAPH_PROMPT_AUGMENT=false skips the patch entirely."""
        monkeypatch.setenv("MEM0_GRAPH_PROMPT_AUGMENT", "false")
        import mem0.graphs.utils as utils_module

        before = utils_module.EXTRACT_RELATIONS_PROMPT
        patch_extract_relations_prompt()
        assert utils_module.EXTRACT_RELATIONS_PROMPT == before


# ============================================================
# Batch-UUID provenance helpers (Phase 1)
# ============================================================


class TestFlattenAddedEntities:
    """_flatten_added_entities normalizes mem0ai's nested return shape."""

    def test_flattens_list_of_lists(self):
        nested = [
            [{"source": "a", "relationship": "R1", "target": "b"}],
            [
                {"source": "c", "relationship": "R2", "target": "d"},
                {"source": "e", "relationship": "R3", "target": "f"},
            ],
        ]
        flat = _flatten_added_entities(nested)
        assert len(flat) == 3
        assert flat[0] == {"source": "a", "relationship": "R1", "target": "b"}

    def test_handles_flat_dicts_in_outer_list(self):
        flat_input = [{"source": "a", "relationship": "R", "target": "b"}]
        out = _flatten_added_entities(flat_input)
        assert out == [{"source": "a", "relationship": "R", "target": "b"}]

    def test_drops_malformed_items(self):
        nested = [
            [{"source": "a", "relationship": "R", "target": "b"}],
            [{"source": "no_target", "relationship": "R"}],  # missing target
            ["string_garbage"],  # non-dict
        ]
        flat = _flatten_added_entities(nested)
        assert len(flat) == 1
        assert flat[0]["source"] == "a"

    def test_empty_input(self):
        assert _flatten_added_entities([]) == []
        assert _flatten_added_entities(None) == []


class TestTagBatchProvenance:
    """tag_batch_provenance runs Pass A + Pass B Cypher under the right conditions."""

    def test_skips_when_graph_is_none(self):
        memory = MagicMock()
        memory.graph = None
        rel, node = tag_batch_provenance(
            memory,
            batch_uuid="b1",
            start_ts=1000,
            user_id="u",
            added_entities=[[{"source": "x", "relationship": "R", "target": "y"}]],
        )
        assert (rel, node) == (0, 0)

    def test_skips_when_no_relations(self):
        memory = MagicMock()
        memory.graph = MagicMock()
        rel, node = tag_batch_provenance(
            memory,
            batch_uuid="b1",
            start_ts=1000,
            user_id="u",
            added_entities=[],
        )
        assert (rel, node) == (0, 0)
        memory.graph.graph.query.assert_not_called()

    def test_runs_pass_a_then_pass_b_when_relations_match(self):
        memory = MagicMock()
        # First call (Pass A) returns 2 tagged; second (Pass B) returns 3.
        memory.graph.graph.query.side_effect = [
            [{"tagged": 2}],
            [{"tagged": 3}],
        ]
        rel, node = tag_batch_provenance(
            memory,
            batch_uuid="b1",
            start_ts=1000,
            user_id="u",
            added_entities=[[{"source": "x", "relationship": "R", "target": "y"}]],
        )
        assert rel == 2
        assert node == 3
        assert memory.graph.graph.query.call_count == 2

        # Pass A params: triples + user_id + batch_uuid + start_ts. No
        # agent_id / run_id (deliberately dropped — see helpers.py docstring).
        pass_a_call = memory.graph.graph.query.call_args_list[0]
        params = pass_a_call.kwargs["params"]
        assert params == {
            "triples": [{"source": "x", "relationship": "R", "target": "y"}],
            "uid": "u",
            "start_ts": 1000,
            "batch_uuid": "b1",
        }

    def test_pass_a_cypher_does_not_filter_by_run_id_or_agent_id(self):
        """Regression: cross-scope MERGE'd nodes (existing nodes whose run_id
        was set by an earlier add) must still be tagged. Filtering Pass A by
        s.run_id silently misses every relation between such nodes.
        """
        memory = MagicMock()
        memory.graph.graph.query.side_effect = [
            [{"tagged": 1}],
            [{"tagged": 1}],
        ]
        tag_batch_provenance(
            memory,
            batch_uuid="b1",
            start_ts=1000,
            user_id="u",
            added_entities=[[{"source": "x", "relationship": "R", "target": "y"}]],
        )
        pass_a_cypher = memory.graph.graph.query.call_args_list[0][0][0]
        assert "$run_id" not in pass_a_cypher
        assert "$agent_id" not in pass_a_cypher
        assert "s.run_id" not in pass_a_cypher
        assert "s.agent_id" not in pass_a_cypher

    def test_skips_pass_b_when_pass_a_tags_zero(self):
        """If Pass A doesn't match any relations (e.g. sanitizer drift),
        Pass B would tag endpoints from prior batches' edges — skip it."""
        memory = MagicMock()
        memory.graph.graph.query.return_value = [{"tagged": 0}]
        rel, node = tag_batch_provenance(
            memory,
            batch_uuid="b1",
            start_ts=1000,
            user_id="u",
            added_entities=[[{"source": "x", "relationship": "R", "target": "y"}]],
        )
        assert (rel, node) == (0, 0)
        # Only Pass A should have been queried.
        assert memory.graph.graph.query.call_count == 1


class TestHardDeleteBatch:
    def test_skips_when_graph_is_none(self):
        memory = MagicMock()
        memory.graph = None
        assert hard_delete_batch(memory, "b1") == (0, 0)

    def test_runs_relations_then_nodes_cypher(self):
        memory = MagicMock()
        memory.graph.graph.query.side_effect = [
            [{"deleted": 4}],  # relations
            [{"deleted": 2}],  # nodes
        ]
        rel, node = hard_delete_batch(memory, "b1")
        assert (rel, node) == (4, 2)
        assert memory.graph.graph.query.call_count == 2
        for call in memory.graph.graph.query.call_args_list:
            assert call.kwargs["params"] == {"batch_uuid": "b1"}


class TestCountBatchAnchors:
    def test_counts_qdrant_points(self):
        memory = MagicMock()
        # Qdrant.scroll returns (records_list, next_offset)
        memory.vector_store.list.return_value = (
            [MagicMock(), MagicMock(), MagicMock()],
            None,
        )
        n = count_batch_anchors(
            memory,
            batch_uuid="b1",
            user_id="u",
            agent_id=None,
            run_id=None,
        )
        assert n == 3

        filters = memory.vector_store.list.call_args.kwargs["filters"]
        assert filters == {"user_id": "u", "batch_uuid": "b1"}

    def test_includes_agent_and_run_in_filters(self):
        memory = MagicMock()
        memory.vector_store.list.return_value = ([], None)
        count_batch_anchors(
            memory,
            batch_uuid="b1",
            user_id="u",
            agent_id="a",
            run_id="r",
        )
        filters = memory.vector_store.list.call_args.kwargs["filters"]
        assert filters == {
            "user_id": "u",
            "batch_uuid": "b1",
            "agent_id": "a",
            "run_id": "r",
        }


class TestAddWithBatchProvenance:
    """End-to-end orchestration: lock, metadata injection, reconciliation, orphan check."""

    def _make_memory(self, *, graph_active=True, anchors=1, added=None):
        """Build a Memory mock that simulates a graph-enabled add."""
        memory = MagicMock()
        memory.graph = MagicMock() if graph_active else None
        # _neo4j_now: first query (timestamp). Then Pass A (tagged>0). Then Pass B.
        memory.graph.graph.query.side_effect = [
            [{"now": 1000}],
            [{"tagged": 1}],
            [{"tagged": 2}],
        ]
        memory.add.return_value = {
            "results": [{"id": "mid", "memory": "fact", "event": "ADD"}],
            "relations": {
                "added_entities": added
                if added is not None
                else [[{"source": "x", "relationship": "R", "target": "y"}]],
            },
        }
        memory.vector_store.list.return_value = (
            [MagicMock() for _ in range(anchors)],
            None,
        )
        return memory

    def test_injects_batch_uuid_into_metadata(self):
        memory = self._make_memory()
        add_with_batch_provenance(
            memory,
            [{"role": "user", "content": "hi"}],
            user_id="u",
            metadata={"source": "chat"},
            enable_graph=True,
        )
        call_kwargs = memory.add.call_args.kwargs
        assert call_kwargs["metadata"]["source"] == "chat"
        assert "batch_uuid" in call_kwargs["metadata"]
        assert len(call_kwargs["metadata"]["batch_uuid"]) == 36  # uuid4

    def test_caller_metadata_is_not_mutated(self):
        """Defensive copy: caller's dict must not gain batch_uuid in-place."""
        memory = self._make_memory()
        caller_metadata = {"source": "chat"}
        add_with_batch_provenance(
            memory,
            [{"role": "user", "content": "hi"}],
            user_id="u",
            metadata=caller_metadata,
            enable_graph=True,
        )
        assert "batch_uuid" not in caller_metadata
        assert caller_metadata == {"source": "chat"}

    def test_skips_reconciliation_when_graph_disabled(self):
        memory = self._make_memory()
        add_with_batch_provenance(
            memory,
            [{"role": "user", "content": "hi"}],
            user_id="u",
            enable_graph=False,
        )
        # No graph queries should run.
        memory.graph.graph.query.assert_not_called()
        # mem.enable_graph should be False during the call.
        assert memory.add.called

    def test_runs_full_reconciliation_when_graph_enabled(self):
        memory = self._make_memory()
        add_with_batch_provenance(
            memory,
            [{"role": "user", "content": "hi"}],
            user_id="u",
            enable_graph=True,
        )
        # Three queries: timestamp anchor, Pass A, Pass B.
        assert memory.graph.graph.query.call_count == 3

    def test_orphan_from_birth_triggers_hard_delete(self):
        """When mem.add() reports added_entities but Qdrant has 0 anchors,
        the just-tagged batch must be hard-deleted immediately."""
        memory = self._make_memory(anchors=0)
        # Extend the side_effect to cover the hard-delete passes (rels + nodes).
        memory.graph.graph.query.side_effect = [
            [{"now": 1000}],
            [{"tagged": 1}],
            [{"tagged": 2}],
            [{"deleted": 1}],
            [{"deleted": 2}],
        ]
        add_with_batch_provenance(
            memory,
            [{"role": "user", "content": "hi"}],
            user_id="u",
            enable_graph=True,
        )
        # 5 queries: timestamp + Pass A + Pass B + hard_delete relations + nodes.
        assert memory.graph.graph.query.call_count == 5

    def test_no_orphan_check_when_added_is_empty(self):
        """If the LLM extracted no relations, orphan-from-birth is impossible
        — skip the Qdrant scroll entirely."""
        memory = self._make_memory(added=[], anchors=0)
        memory.graph.graph.query.side_effect = [[{"now": 1000}]]
        add_with_batch_provenance(
            memory,
            [{"role": "user", "content": "hi"}],
            user_id="u",
            enable_graph=True,
        )
        # Only timestamp anchor — no Pass A, no Pass B, no scroll.
        assert memory.graph.graph.query.call_count == 1
        memory.vector_store.list.assert_not_called()

    def test_reconciliation_failure_does_not_break_add(self):
        """If Cypher reconciliation throws after mem.add() succeeds, the
        add() result is still returned; vector store stays intact."""
        memory = self._make_memory()
        memory.graph.graph.query.side_effect = [
            [{"now": 1000}],
            RuntimeError("Neo4j down mid-reconciliation"),
        ]
        result = add_with_batch_provenance(
            memory,
            [{"role": "user", "content": "hi"}],
            user_id="u",
            enable_graph=True,
        )
        # Add result preserved.
        assert result["results"][0]["id"] == "mid"

    def test_neo4j_now_failure_falls_back_to_plain_add(self):
        """If we can't anchor the timestamp window, plain add (no provenance)
        runs instead — better than blocking the user."""
        memory = self._make_memory()
        memory.graph.graph.query.side_effect = RuntimeError("Neo4j unreachable")
        result = add_with_batch_provenance(
            memory,
            [{"role": "user", "content": "hi"}],
            user_id="u",
            enable_graph=True,
        )
        # mem.add was still called; result returned.
        assert memory.add.called
        assert result["results"][0]["id"] == "mid"


class TestDeleteMemoryWithBatch:
    """End-to-end orchestration of the delete-side cleanup."""

    def _make_memory(self, *, payload=None, graph=True, anchors=0):
        """Build a Memory mock for delete tests."""
        memory = MagicMock()
        memory.graph = MagicMock() if graph else None
        memory.vector_store.get.return_value.payload = (
            payload if payload is not None else {}
        )
        memory.vector_store.list.return_value = (
            [MagicMock() for _ in range(anchors)],
            None,
        )
        return memory

    def test_legacy_path_when_no_batch_uuid(self):
        """Memories without batch_uuid (added before rollout) use the legacy
        cascade + gc_orphan path via mem.delete()."""
        memory = self._make_memory(payload={"user_id": "alice"})
        with patch("mem0_mcp_selfhosted.helpers.gc_orphan_graph_nodes") as mock_gc:
            from mem0_mcp_selfhosted.helpers import delete_memory_with_batch

            result = delete_memory_with_batch(memory, "mid", enable_graph=True)
        assert result == {"message": "Memory deleted successfully!"}
        memory.delete.assert_called_once_with("mid")
        # Legacy path: enable_graph set True, gc_orphan_graph_nodes called
        # because graph was active and scope had user_id.
        assert memory.enable_graph is True
        mock_gc.assert_called_once_with(memory, {"user_id": "alice"})

    def test_legacy_path_when_graph_disabled(self):
        """When caller passes enable_graph=False, legacy path runs (no batch
        cleanup, no gc_orphan)."""
        memory = self._make_memory(payload={"user_id": "alice", "batch_uuid": "b1"})
        with patch("mem0_mcp_selfhosted.helpers.gc_orphan_graph_nodes") as mock_gc:
            from mem0_mcp_selfhosted.helpers import delete_memory_with_batch

            delete_memory_with_batch(memory, "mid", enable_graph=False)
        assert memory.enable_graph is False
        memory.delete.assert_called_once_with("mid")
        mock_gc.assert_not_called()

    def test_batch_path_bypasses_re_extraction(self):
        """When batch_uuid is present, mem.enable_graph is False during
        mem.delete() — that bypasses mem0ai's re-extraction cascade."""
        memory = self._make_memory(
            payload={"user_id": "alice", "batch_uuid": "b1"}, anchors=0
        )
        # Hard-delete inside the helper hits Cypher; stub it.
        memory.graph.graph.query.side_effect = [
            [{"deleted": 2}],  # hard_delete relations
            [{"deleted": 1}],  # hard_delete nodes
        ]
        # Capture enable_graph value at the moment mem.delete runs.
        captured = {}

        def _capture_delete(_memory_id):
            captured["enable_graph_during_delete"] = memory.enable_graph

        memory.delete.side_effect = _capture_delete
        memory.enable_graph = True  # prior request left it True

        from mem0_mcp_selfhosted.helpers import delete_memory_with_batch

        delete_memory_with_batch(memory, "mid", enable_graph=True)

        # During mem.delete, enable_graph must have been False.
        assert captured["enable_graph_during_delete"] is False
        # After return, we restored to the prior value.
        assert memory.enable_graph is True

    def test_hard_deletes_only_when_no_anchors_remain(self):
        """If other vector memories still carry the batch_uuid in scope,
        the graph batch must NOT be hard-deleted."""
        memory = self._make_memory(
            payload={"user_id": "alice", "batch_uuid": "b1"}, anchors=3
        )
        from mem0_mcp_selfhosted.helpers import delete_memory_with_batch

        delete_memory_with_batch(memory, "mid", enable_graph=True)
        # mem.delete ran, but no Cypher hard-delete (graph.query wasn't called).
        memory.delete.assert_called_once_with("mid")
        memory.graph.graph.query.assert_not_called()

    def test_qdrant_scroll_failure_leaves_graph_intact(self):
        """If the anchor count fails (Qdrant down mid-flow), we never
        hard-delete on partial state — the tagged graph stays as-is."""
        memory = self._make_memory(payload={"user_id": "alice", "batch_uuid": "b1"})
        memory.vector_store.list.side_effect = RuntimeError("Qdrant down")

        from mem0_mcp_selfhosted.helpers import delete_memory_with_batch

        result = delete_memory_with_batch(memory, "mid", enable_graph=True)
        # Vector delete still happened.
        memory.delete.assert_called_once_with("mid")
        # No hard-delete attempted.
        memory.graph.graph.query.assert_not_called()
        # Caller still gets a success message — vector store IS clean.
        assert result == {"message": "Memory deleted successfully!"}

    def test_payload_read_failure_falls_through_to_legacy(self):
        """If we can't read the memory's payload at all, fall through to the
        legacy mem.delete() path with no batch dispatch."""
        memory = self._make_memory(graph=True)
        memory.vector_store.get.side_effect = RuntimeError("payload missing")

        with patch("mem0_mcp_selfhosted.helpers.gc_orphan_graph_nodes") as mock_gc:
            from mem0_mcp_selfhosted.helpers import delete_memory_with_batch

            delete_memory_with_batch(memory, "mid", enable_graph=True)
        memory.delete.assert_called_once_with("mid")
        # No scope captured → gc_orphan not called.
        mock_gc.assert_not_called()

    def test_raises_when_memory_is_none(self):
        from mem0_mcp_selfhosted.helpers import delete_memory_with_batch

        with pytest.raises(RuntimeError, match="Memory not initialized"):
            delete_memory_with_batch(None, "mid", enable_graph=True)
