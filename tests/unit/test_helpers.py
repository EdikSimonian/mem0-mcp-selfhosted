"""Tests for helpers.py — error wrapper, call_with_graph, bulk delete, user_id, sanitizer, Gemini patch."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from mem0_mcp_selfhosted.helpers import (
    _make_enhanced_sanitizer,
    _mem0_call,
    call_with_graph,
    gc_orphan_graph_nodes,
    get_default_user_id,
    patch_extract_relations_prompt,
    patch_gemini_parse_response,
    safe_bulk_delete,
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


class TestSafeBulkDelete:
    def test_iterates_and_deletes(self):
        memory = MagicMock()
        memory.enable_graph = False
        memory.graph = None

        # Mock vector_store.list returning items with .id
        item1 = MagicMock()
        item1.id = "id-1"
        item2 = MagicMock()
        item2.id = "id-2"
        memory.vector_store.list.return_value = [item1, item2]

        count = safe_bulk_delete(memory, {"user_id": "testuser"})

        assert count == 2
        assert memory.delete.call_count == 2
        memory.delete.assert_any_call("id-1")
        memory.delete.assert_any_call("id-2")

    def test_graph_cleanup_when_graph_enabled_true(self):
        memory = MagicMock()
        memory.graph = MagicMock()
        memory.vector_store.list.return_value = []

        safe_bulk_delete(memory, {"user_id": "testuser"}, graph_enabled=True)

        memory.graph.delete_all.assert_called_once_with({"user_id": "testuser"})

    def test_no_graph_cleanup_when_graph_enabled_false(self):
        memory = MagicMock()
        memory.graph = MagicMock()
        memory.vector_store.list.return_value = []

        safe_bulk_delete(memory, {"user_id": "testuser"}, graph_enabled=False)

        memory.graph.delete_all.assert_not_called()

    def test_no_graph_cleanup_default(self):
        """Default graph_enabled=False skips graph cleanup."""
        memory = MagicMock()
        memory.graph = MagicMock()
        memory.vector_store.list.return_value = []

        safe_bulk_delete(memory, {"user_id": "testuser"})

        memory.graph.delete_all.assert_not_called()


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
