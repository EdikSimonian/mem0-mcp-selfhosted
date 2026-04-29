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


class TestBatchProvenanceEndToEnd:
    """Full add → tag → delete → hard-delete round trip via the new helpers.

    These exercise the failure mode that motivated the whole design: text
    mentioning code identifiers that mem0ai's re-extraction-based delete
    cascade silently fails on. Provenance tagging makes the cleanup
    deterministic regardless of extraction drift.
    """

    @pytest.fixture
    def isolated_scope(self, memory_instance, neo4j_available):
        if memory_instance.graph is None:
            pytest.skip("Graph backend not initialized on memory_instance")
        run_id = f"prov-e2e-{uuid.uuid4()}"
        yield run_id
        # Belt-and-suspenders cleanup: detach-delete any leftover scoped nodes.
        try:
            memory_instance.graph.graph.query(
                "MATCH (n {run_id: $rid}) DETACH DELETE n",
                params={"rid": run_id},
            )
        except Exception:
            pass

    def test_add_then_delete_clears_graph_for_single_memory_batch(
        self, memory_instance, isolated_scope
    ):
        """Single memory in a batch: deleting it must clear the whole graph
        batch deterministically — not via re-extraction."""
        from mem0_mcp_selfhosted.helpers import (
            add_with_batch_provenance,
            delete_memory_with_batch,
        )

        run_id = isolated_scope
        text = (
            f"ProvenanceProbe: BluePort runs in Trantor cloud-region-{uuid.uuid4().hex[:6]}. "
            f"Anya leads BluePort at Trantor."
        )

        add_result = add_with_batch_provenance(
            memory_instance,
            [{"role": "user", "content": text}],
            user_id="prov-e2e-user",
            run_id=run_id,
            enable_graph=True,
        )
        memory_ids = [m["id"] for m in add_result.get("results", [])]
        assert memory_ids, f"Expected at least one memory; got {add_result!r}"

        # Sanity: graph picked up at least one tagged node in our scope.
        rows = memory_instance.graph.graph.query(
            "MATCH (n {run_id: $rid}) RETURN count(n) AS cnt",
            params={"rid": run_id},
        )
        assert rows[0]["cnt"] > 0, "no graph nodes created — extraction skipped?"

        # Delete every memory in the batch.
        for mid in memory_ids:
            delete_memory_with_batch(
                memory_instance, mid, enable_graph=True
            )

        # Graph must be fully clean for this scope after the LAST memory's delete.
        rows = memory_instance.graph.graph.query(
            "MATCH (n {run_id: $rid}) RETURN count(n) AS cnt",
            params={"rid": run_id},
        )
        assert rows[0]["cnt"] == 0, (
            f"Provenance cleanup left {rows[0]['cnt']} graph nodes in scope "
            f"after all batch memories deleted. Re-extraction would leak; "
            f"batch_uuid hard-delete should not."
        )

    def test_partial_batch_delete_preserves_remaining(
        self, memory_instance, isolated_scope
    ):
        """Multi-memory batch: deleting only ONE memory must NOT collapse the
        graph — other memories from the same batch still anchor it."""
        from mem0_mcp_selfhosted.helpers import (
            add_with_batch_provenance,
            delete_memory_with_batch,
        )

        run_id = isolated_scope
        text = (
            f"ProvenanceProbe2: BluePort runs in Trantor cloud-region-{uuid.uuid4().hex[:6]}. "
            f"Anya leads BluePort at Trantor."
        )

        add_result = add_with_batch_provenance(
            memory_instance,
            [{"role": "user", "content": text}],
            user_id="prov-e2e-user",
            run_id=run_id,
            enable_graph=True,
        )
        memory_ids = [m["id"] for m in add_result.get("results", [])]
        if len(memory_ids) < 2:
            pytest.skip(
                f"Extraction produced only {len(memory_ids)} fact(s); test "
                "needs >=2 memories in one batch. Re-run when LLM extracts more."
            )

        graph_before = memory_instance.graph.graph.query(
            "MATCH (n {run_id: $rid}) RETURN count(n) AS cnt",
            params={"rid": run_id},
        )[0]["cnt"]
        assert graph_before > 0

        # Delete just the first memory.
        delete_memory_with_batch(
            memory_instance, memory_ids[0], enable_graph=True
        )

        graph_after = memory_instance.graph.graph.query(
            "MATCH (n {run_id: $rid}) RETURN count(n) AS cnt",
            params={"rid": run_id},
        )[0]["cnt"]
        assert graph_after == graph_before, (
            f"Partial batch delete should preserve graph (other memories "
            f"still anchor it). Got {graph_after}, expected {graph_before}."
        )

        # Now delete the rest — graph should fully clear.
        for mid in memory_ids[1:]:
            delete_memory_with_batch(
                memory_instance, mid, enable_graph=True
            )
        rows = memory_instance.graph.graph.query(
            "MATCH (n {run_id: $rid}) RETURN count(n) AS cnt",
            params={"rid": run_id},
        )
        assert rows[0]["cnt"] == 0

    def test_abstract_config_note_no_leak(self, memory_instance, isolated_scope):
        """The failure mode that motivated the whole design.

        Memory `1df3d64f` flagged this as HIGH-severity: when memory text mentions
        code identifiers ("helpers.py", "_AUGMENT_RULES", "rule_a"), Haiku-class
        extraction over-produces graph nodes for the rule names and file paths.
        The pre-provenance gc_orphan path could not catch them on delete because
        re-extraction generates DIFFERENT triples than add-time, leaving orphans
        that lingered in find_entity / get_entity until manual Cypher swept them.

        Provenance design's contract: regardless of what extraction produced,
        delete must hard-delete every tagged node/edge in the batch.
        """
        from mem0_mcp_selfhosted.helpers import (
            add_with_batch_provenance,
            delete_memory_with_batch,
        )

        run_id = isolated_scope
        # Abstract text mirroring the kind of save that previously leaked.
        # Names randomized to dodge any LLM-side memorization of the shape.
        token = uuid.uuid4().hex[:8]
        text = (
            f"ConfigNote-{token}: helpers_module_{token} contains rule_a, rule_b, "
            f"and rule_c. Each rule is appended to extract_relations_prompt_{token}. "
            f"Sentinel string for idempotence is 'mandatory_rules_{token}'. "
            f"Verified by smoke_test_{token} in tests/contract."
        )

        add_with_batch_provenance(
            memory_instance,
            [{"role": "user", "content": text}],
            user_id="prov-e2e-user",
            run_id=run_id,
            enable_graph=True,
        )

        # Whatever extraction did — we don't assert on its quality, only that
        # cleanup is deterministic. Fetch the actual count first.
        nodes_after_add = memory_instance.graph.graph.query(
            "MATCH (n {run_id: $rid}) RETURN count(n) AS cnt",
            params={"rid": run_id},
        )[0]["cnt"]
        # If extraction produced nothing (Haiku skipped on this abstract input),
        # the test trivially holds. We still need at least the orphan-from-birth
        # path to have cleaned up, which is also a passing outcome.
        if nodes_after_add == 0:
            pytest.skip(
                f"Abstract text produced no graph extraction this run. "
                f"Re-run when extraction is more aggressive."
            )

        # Delete every memory in scope. Use safe_bulk_delete equivalent —
        # iterate vector memories carrying our run_id and delete each.
        result = memory_instance.vector_store.list(
            filters={"run_id": run_id}, limit=100
        )
        points = result[0] if isinstance(result, tuple) else result
        memory_ids = [p.id for p in points]
        for mid in memory_ids:
            delete_memory_with_batch(
                memory_instance, mid, enable_graph=True
            )

        # Hard contract: graph in scope is empty after all batch memories
        # delete, no matter what nodes extraction generated.
        rows = memory_instance.graph.graph.query(
            "MATCH (n {run_id: $rid}) RETURN count(n) AS cnt, collect(n.name) AS leaked",
            params={"rid": run_id},
        )
        cnt = rows[0]["cnt"]
        leaked = rows[0]["leaked"]
        assert cnt == 0, (
            f"Provenance cleanup left {cnt} graph node(s) in scope after "
            f"batch deletion. Leaked names: {leaked}. This is the failure "
            f"mode the design exists to fix — re-extraction would have left "
            f"these, batch_uuid hard-delete should not."
        )

    def test_shared_node_across_batches_keeps_per_batch_provenance(
        self, memory_instance, isolated_scope
    ):
        """Two batches mention the same entity; the shared node carries both
        batch_uuids. Deleting one batch must NOT delete the shared node —
        the other batch still anchors it. Deleting both batches removes it.

        This is the load-bearing test for codex's hole #1: edge `batch_uuid`
        being a list (not scalar) so two batches can co-own a graph element.
        """
        from mem0_mcp_selfhosted.helpers import (
            add_with_batch_provenance,
            delete_memory_with_batch,
        )

        run_id = isolated_scope
        # Same shared entity in both batches: a person + an org. Different
        # secondary entities to keep batch identity distinct.
        token = uuid.uuid4().hex[:6]
        shared_org = f"sharedco_{token}"
        shared_person = f"shareduser_{token}"

        batch_1_text = (
            f"{shared_person} is the founder of {shared_org}. "
            f"{shared_org} runs the analytics service."
        )
        batch_2_text = (
            f"{shared_person} also leads {shared_org}'s research team. "
            f"{shared_org} acquired BetaCorp last quarter."
        )

        # Add both batches.
        result1 = add_with_batch_provenance(
            memory_instance,
            [{"role": "user", "content": batch_1_text}],
            user_id="prov-e2e-user",
            run_id=run_id,
            enable_graph=True,
        )
        result2 = add_with_batch_provenance(
            memory_instance,
            [{"role": "user", "content": batch_2_text}],
            user_id="prov-e2e-user",
            run_id=run_id,
            enable_graph=True,
        )

        b1_ids = [m["id"] for m in result1.get("results", [])]
        b2_ids = [m["id"] for m in result2.get("results", [])]
        if not b1_ids or not b2_ids:
            pytest.skip(
                f"Both batches must produce memories; got "
                f"b1={len(b1_ids)}, b2={len(b2_ids)}. Re-run."
            )

        # Read the shared org's batch_uuids — should have at least 2 distinct uuids
        # if the LLM both batches' extraction produced an edge touching shared_org.
        rows = memory_instance.graph.graph.query(
            "MATCH (n {run_id: $rid, name: $name}) RETURN n.batch_uuids AS uids",
            params={"rid": run_id, "name": shared_org},
        )
        if not rows or rows[0]["uids"] is None or len(rows[0]["uids"]) < 2:
            pytest.skip(
                f"Shared node {shared_org} did not accumulate >=2 batch_uuids; "
                f"extraction may have skipped one batch. Got: {rows}"
            )

        # Delete only batch 1's memories. Shared node should survive.
        for mid in b1_ids:
            delete_memory_with_batch(
                memory_instance, mid, enable_graph=True
            )

        rows = memory_instance.graph.graph.query(
            "MATCH (n {run_id: $rid, name: $name}) RETURN n.batch_uuids AS uids",
            params={"rid": run_id, "name": shared_org},
        )
        assert rows, (
            f"Shared node {shared_org} was deleted after only batch 1's memories "
            f"were removed. Batch 2 should still anchor it."
        )
        remaining_uids = rows[0]["uids"] or []
        # batch 1's uuid should be gone, batch 2's uuid should still be present.
        assert len(remaining_uids) >= 1, (
            f"After batch 1 delete, shared node should still carry batch 2's uuid. "
            f"Got: {remaining_uids}"
        )

        # Now delete batch 2. Shared node should disappear entirely.
        for mid in b2_ids:
            delete_memory_with_batch(
                memory_instance, mid, enable_graph=True
            )
        rows = memory_instance.graph.graph.query(
            "MATCH (n {run_id: $rid}) RETURN count(n) AS cnt",
            params={"rid": run_id},
        )
        assert rows[0]["cnt"] == 0, (
            f"After both batches deleted, scope must be fully clean. "
            f"Got {rows[0]['cnt']} node(s) remaining."
        )
