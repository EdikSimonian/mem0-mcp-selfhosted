"""Integration tests: Neo4j graph operations.

These tests require Neo4j in addition to the base infrastructure.
Skipped independently when Neo4j is unavailable.
"""

from __future__ import annotations

import json
import os

import pytest

from mem0_mcp_selfhosted.graph_tools import find_entity
from mem0_mcp_selfhosted.helpers import call_with_graph

pytestmark = pytest.mark.integration

GEMINI_TEST_COLLECTION = "mem0_gemini_graph_test"


@pytest.fixture(scope="module")
def gemini_api_key():
    """Skip Gemini graph tests if no GOOGLE_API_KEY available."""
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        pytest.skip("GOOGLE_API_KEY not set — skipping Gemini graph tests")
    return key


def _make_gemini_memory(qdrant_url, ollama_url, graph_llm_provider, gemini_api_key):
    """Create a Memory instance using the specified graph LLM provider."""
    original_collection = os.environ.get("MEM0_COLLECTION")
    original_graph = os.environ.get("MEM0_ENABLE_GRAPH")
    original_graph_provider = os.environ.get("MEM0_GRAPH_LLM_PROVIDER")
    original_graph_model = os.environ.get("MEM0_GRAPH_LLM_MODEL")

    os.environ["MEM0_COLLECTION"] = GEMINI_TEST_COLLECTION
    os.environ["MEM0_ENABLE_GRAPH"] = "true"
    os.environ["MEM0_GRAPH_LLM_PROVIDER"] = graph_llm_provider
    os.environ["MEM0_GRAPH_LLM_MODEL"] = "gemini-2.5-flash-lite"
    os.environ["GOOGLE_API_KEY"] = gemini_api_key

    from mem0_mcp_selfhosted.config import build_config
    from mem0_mcp_selfhosted.server import register_providers

    config_dict, providers_info, split_config = build_config()
    register_providers(providers_info)

    from mem0 import Memory
    memory = Memory.from_config(config_dict)

    # If split-model was requested, swap the graph LLM with the router
    if split_config and memory.graph is not None:
        from mem0_mcp_selfhosted.llm_router import SplitModelGraphLLM, SplitModelGraphLLMConfig
        memory.graph.llm = SplitModelGraphLLM(SplitModelGraphLLMConfig(**split_config))

    def cleanup():
        try:
            from qdrant_client import QdrantClient
            QdrantClient(url=qdrant_url).delete_collection(GEMINI_TEST_COLLECTION)
        except Exception:
            pass
        # Restore env
        for key, orig in [
            ("MEM0_COLLECTION", original_collection),
            ("MEM0_ENABLE_GRAPH", original_graph),
            ("MEM0_GRAPH_LLM_PROVIDER", original_graph_provider),
            ("MEM0_GRAPH_LLM_MODEL", original_graph_model),
        ]:
            if orig is not None:
                os.environ[key] = orig
            else:
                os.environ.pop(key, None)

    return memory, cleanup


class TestGraphOperations:
    def test_add_with_graph_enabled(
        self, memory_instance, neo4j_available, test_user_id
    ):
        """Add a memory with graph extraction enabled."""
        # Enable graph for this test
        original = os.environ.get("MEM0_ENABLE_GRAPH")
        os.environ["MEM0_ENABLE_GRAPH"] = "true"

        try:
            result = call_with_graph(
                memory_instance,
                True,
                True,
                memory_instance.add,
                [
                    {
                        "role": "user",
                        "content": "Alice prefers TypeScript over JavaScript for web development",
                    }
                ],
                user_id=test_user_id,
            )

            assert "results" in result
            assert len(result["results"]) >= 1
            assert "id" in result["results"][0]
        finally:
            if original is not None:
                os.environ["MEM0_ENABLE_GRAPH"] = original
            else:
                os.environ.pop("MEM0_ENABLE_GRAPH", None)

    def test_find_entity_finds_entity(
        self, memory_instance, neo4j_available, test_user_id
    ):
        """Add an entity-rich memory with graph, then search Neo4j for it."""
        original = os.environ.get("MEM0_ENABLE_GRAPH")
        os.environ["MEM0_ENABLE_GRAPH"] = "true"

        try:
            call_with_graph(
                memory_instance,
                True,
                True,
                memory_instance.add,
                [
                    {
                        "role": "user",
                        "content": "GraphTestUser loves using Kubernetes for container orchestration",
                    }
                ],
                user_id=test_user_id,
            )

            result_str = find_entity("GraphTestUser")
            result = json.loads(result_str)

            assert "error" not in result
            assert "entities" in result
            assert len(result["entities"]) >= 1
            entity_names = [e.get("entity", "").lower() for e in result["entities"]]
            assert any("graphtestuser" in name for name in entity_names)
        finally:
            if original is not None:
                os.environ["MEM0_ENABLE_GRAPH"] = original
            else:
                os.environ.pop("MEM0_ENABLE_GRAPH", None)


class TestGeminiGraphOperations:
    """Graph operations using Gemini as graph LLM provider."""

    def test_add_with_gemini_graph(
        self, qdrant_url, ollama_url, neo4j_available, gemini_api_key
    ):
        """Add memory with graph extraction using Gemini-only graph LLM."""
        memory, cleanup = _make_gemini_memory(
            qdrant_url, ollama_url, "gemini", gemini_api_key
        )
        try:
            result = call_with_graph(
                memory,
                True,
                True,
                memory.add,
                [
                    {
                        "role": "user",
                        "content": "GeminiTestUser prefers Python over Ruby for backend development",
                    }
                ],
                user_id="inttest-gemini-graph",
            )

            assert "results" in result
            assert len(result["results"]) >= 1
            assert "id" in result["results"][0]
        finally:
            cleanup()

    def test_add_with_gemini_split_graph(
        self, qdrant_url, ollama_url, neo4j_available, gemini_api_key
    ):
        """Add memory with graph extraction using split-model router (Gemini + Claude)."""
        memory, cleanup = _make_gemini_memory(
            qdrant_url, ollama_url, "gemini_split", gemini_api_key
        )
        try:
            result = call_with_graph(
                memory,
                True,
                True,
                memory.add,
                [
                    {
                        "role": "user",
                        "content": "SplitTestUser works at Anthropic and lives in San Francisco",
                    }
                ],
                user_id="inttest-gemini-split-graph",
            )

            assert "results" in result
            assert len(result["results"]) >= 1
            assert "id" in result["results"][0]
        finally:
            cleanup()
