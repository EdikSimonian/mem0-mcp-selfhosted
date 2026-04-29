"""Live contract tests for the batch-UUID provenance design.

These verify behaviors that source inspection alone cannot prove:
  - Custom metadata keys actually survive into the raw Qdrant payload.
  - Neo4j MERGE writes ms-precision timestamps on nodes/edges.
  - ON MATCH advances `updated_at` but not `created_at` on relationships.

Run with: .venv/bin/python -m pytest tests/integration/test_provenance_contract_live.py -v
Skipped automatically when infrastructure is unavailable.

Companion to tests/contract/test_provenance_invariants.py (fast,
source-inspection-only).
"""

from __future__ import annotations

import time
import uuid

import pytest

pytestmark = pytest.mark.integration


class TestMetadataPayloadRoundTrip:
    """Verify mem.add() preserves a custom metadata key into Qdrant raw payload.

    Uses infer=False to bypass the LLM — this isolates the metadata-passthrough
    invariant from extraction behavior. The provenance design's Pass 1
    (batch_uuid in metadata) depends on this.
    """

    def test_custom_key_survives_to_payload(self, memory_instance, test_user_id):
        sentinel = f"batch-test-{uuid.uuid4()}"
        result = memory_instance.add(
            f"provenance contract probe {uuid.uuid4()}",
            user_id=test_user_id,
            metadata={"batch_uuid": sentinel},
            infer=False,
        )

        # infer=False path returns memories directly under "results"
        memories = result.get("results", []) if isinstance(result, dict) else result
        assert memories, f"add() returned no memories: {result!r}"
        memory_id = memories[0]["id"]

        try:
            payload = memory_instance.vector_store.get(vector_id=memory_id).payload
            assert payload.get("batch_uuid") == sentinel, (
                "INVARIANT BROKEN: custom metadata key did not survive to "
                f"raw Qdrant payload. Got payload keys: {sorted(payload.keys())}"
            )
        finally:
            memory_instance.delete(memory_id=memory_id)


class TestGraphTimestampBehavior:
    """Verify Neo4j MERGE writes timestamps the provenance reconciliation depends on.

    Pass A's WHERE clause is `r.created_at >= $start_ts OR r.updated_at >= $start_ts`,
    so we must verify (a) both fields exist after creation, (b) ms precision,
    (c) ON MATCH advances updated_at but not created_at.
    """

    @pytest.fixture
    def graph_query(self, memory_instance, neo4j_available):
        """Yield a thin query helper bound to the live mem.graph."""
        if memory_instance.graph is None:
            pytest.skip("Graph backend not initialized on memory_instance")

        scope_uid = f"provenance-contract-{uuid.uuid4()}"
        scope = {"user_id": scope_uid}

        # mem0ai's MemoryGraph holds a Neo4jGraph at .graph; that has .query.
        def q(cypher, params=None):
            return memory_instance.graph.graph.query(cypher, params=params or {})

        yield q, scope_uid

        # Teardown: detach-delete everything tagged with our scope user_id
        q(
            "MATCH (n {user_id: $uid}) DETACH DELETE n",
            {"uid": scope_uid},
        )

    def test_node_created_property_is_ms_timestamp(self, graph_query):
        q, uid = graph_query
        before_ms = int(time.time() * 1000)
        q(
            """
            MERGE (n:Test {name: 'probe-node', user_id: $uid})
            ON CREATE SET n.created = timestamp()
            """,
            {"uid": uid},
        )
        rows = q(
            "MATCH (n {user_id: $uid, name: 'probe-node'}) RETURN n.created AS c",
            {"uid": uid},
        )
        assert rows, "node not found after MERGE"
        created = rows[0]["c"]
        assert isinstance(created, int), f"created must be int ms, got {type(created)}"
        # Sanity: within a 60-second envelope of wall clock (skew tolerance)
        now_ms = int(time.time() * 1000)
        assert before_ms - 60_000 <= created <= now_ms + 60_000, (
            f"created timestamp {created} not within wall-clock envelope "
            f"[{before_ms - 60_000}, {now_ms + 60_000}]"
        )

    def test_relation_has_created_at_and_updated_at_on_create(self, graph_query):
        q, uid = graph_query
        q(
            """
            MERGE (s:Test {name: 'src', user_id: $uid})
            MERGE (d:Test {name: 'dst', user_id: $uid})
            MERGE (s)-[r:TEST_REL]->(d)
            ON CREATE SET r.created_at = timestamp(),
                          r.updated_at = timestamp()
            ON MATCH SET r.updated_at = timestamp()
            """,
            {"uid": uid},
        )
        rows = q(
            """
            MATCH (s {user_id: $uid, name: 'src'})-[r:TEST_REL]->(d)
            RETURN r.created_at AS ca, r.updated_at AS ua
            """,
            {"uid": uid},
        )
        assert rows, "relationship not found after MERGE"
        ca, ua = rows[0]["ca"], rows[0]["ua"]
        assert isinstance(ca, int) and isinstance(ua, int), (
            f"created_at/updated_at must be int ms, got {type(ca)}/{type(ua)}"
        )

    def test_on_match_advances_updated_at_not_created_at(self, graph_query):
        """The load-bearing test for codex's hole #2: re-MERGEd edges must
        get updated_at bumped (so Pass A finds them) without created_at moving."""
        q, uid = graph_query

        # First MERGE — creates the relationship
        q(
            """
            MERGE (s:Test {name: 'src', user_id: $uid})
            MERGE (d:Test {name: 'dst', user_id: $uid})
            MERGE (s)-[r:TEST_REL]->(d)
            ON CREATE SET r.created_at = timestamp(),
                          r.updated_at = timestamp()
            ON MATCH SET r.updated_at = timestamp()
            """,
            {"uid": uid},
        )
        rows1 = q(
            """
            MATCH ({user_id: $uid, name: 'src'})-[r:TEST_REL]->()
            RETURN r.created_at AS ca, r.updated_at AS ua
            """,
            {"uid": uid},
        )
        ca1, ua1 = rows1[0]["ca"], rows1[0]["ua"]

        # Wait long enough for timestamp() to definitely advance (>1ms)
        time.sleep(0.05)

        # Second MERGE — must hit ON MATCH branch
        q(
            """
            MERGE (s:Test {name: 'src', user_id: $uid})
            MERGE (d:Test {name: 'dst', user_id: $uid})
            MERGE (s)-[r:TEST_REL]->(d)
            ON CREATE SET r.created_at = timestamp(),
                          r.updated_at = timestamp()
            ON MATCH SET r.updated_at = timestamp()
            """,
            {"uid": uid},
        )
        rows2 = q(
            """
            MATCH ({user_id: $uid, name: 'src'})-[r:TEST_REL]->()
            RETURN r.created_at AS ca, r.updated_at AS ua
            """,
            {"uid": uid},
        )
        ca2, ua2 = rows2[0]["ca"], rows2[0]["ua"]

        assert ca2 == ca1, (
            f"INVARIANT BROKEN: ON MATCH must NOT change created_at "
            f"(was {ca1}, became {ca2}). Provenance Pass A would falsely "
            f"tag edges from a prior batch as new."
        )
        assert ua2 > ua1, (
            f"INVARIANT BROKEN: ON MATCH must advance updated_at "
            f"(was {ua1}, became {ua2}). Without this, Pass A's "
            f"`r.updated_at >= $start_ts` filter cannot find re-merged edges."
        )
