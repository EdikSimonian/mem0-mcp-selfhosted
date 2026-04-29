"""Contract tests: invariants the batch-UUID provenance design depends on.

These are fast, source-inspection-only tests. They verify that mem0ai's
internals still behave as our planned add/delete wrappers expect, without
requiring live Neo4j or Qdrant. Live timestamp behavior is covered in
tests/integration/test_provenance_contract_live.py.

If these fail after a mem0ai upgrade, the provenance helpers in helpers.py
need updating before the upgrade can ship.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.contract


class TestMetadataPreservation:
    """Memory._create_memory must pass user-provided metadata through to vector_store.insert.

    The provenance design tags each add_memory call with a batch_uuid via the
    metadata channel. If mem0ai stops preserving metadata (or starts overwriting
    keys on the way down), batch lookups at delete time silently miss.
    """

    def test_create_memory_deepcopies_metadata(self):
        try:
            from mem0.memory.main import Memory
        except ImportError:
            pytest.skip("mem0ai not installed")

        source = inspect.getsource(Memory._create_memory)
        assert "deepcopy(metadata)" in source, (
            "INVARIANT BROKEN: _create_memory must deepcopy the metadata arg "
            "so caller-supplied keys (e.g. batch_uuid) survive into the vector "
            "store payload. Found: caller's metadata may be mutated or dropped."
        )

    def test_create_memory_inserts_metadata_as_payload(self):
        try:
            from mem0.memory.main import Memory
        except ImportError:
            pytest.skip("mem0ai not installed")

        source = inspect.getsource(Memory._create_memory)
        # The exact call form: vector_store.insert(vectors=..., ids=..., payloads=[new_metadata])
        assert "self.vector_store.insert(" in source, (
            "INVARIANT BROKEN: _create_memory must call self.vector_store.insert. "
            "Provenance design relies on metadata reaching the Qdrant payload."
        )
        assert "payloads=[new_metadata]" in source, (
            "INVARIANT BROKEN: _create_memory must pass new_metadata as the "
            "payload. Provenance design reads batch_uuid back via "
            "vector_store.get(memory_id).payload['batch_uuid']."
        )

    def test_create_memory_does_not_strip_unknown_keys(self):
        """new_metadata is built by adding mem0 keys to caller's dict, not by allowlist.

        If mem0ai ever switches to an allowlist (only known keys survive),
        batch_uuid would be silently dropped.
        """
        try:
            from mem0.memory.main import Memory
        except ImportError:
            pytest.skip("mem0ai not installed")

        source = inspect.getsource(Memory._create_memory)
        # The current shape: assigns new_metadata["data"] = ..., new_metadata["hash"] = ...
        # It augments the deepcopy, not filters it.
        assert 'new_metadata["data"] = data' in source, (
            "INVARIANT BROKEN: _create_memory must augment metadata in place "
            "(new_metadata['data'] = data). If this becomes an allowlist build, "
            "batch_uuid would be filtered out."
        )


class TestDeleteGraphGate:
    """Memory.delete must invoke graph cleanup conditionally on self.enable_graph.

    The provenance design's delete wrapper temporarily sets enable_graph=False
    around mem.delete() to bypass mem0ai's re-extraction-based cascade entirely.
    That bypass only works if delete() actually checks the flag at call time.
    """

    def test_delete_gates_graph_cleanup_on_enable_graph(self):
        try:
            from mem0.memory.main import Memory
        except ImportError:
            pytest.skip("mem0ai not installed")

        source = inspect.getsource(Memory.delete)
        assert "if self.enable_graph:" in source, (
            "INVARIANT BROKEN: Memory.delete must guard graph cleanup on "
            "self.enable_graph. Provenance design temporarily flips this flag "
            "to skip the re-extraction cascade — that bypass requires the "
            "guard to read the flag at call time."
        )

    def test_delete_calls_graph_delete_with_text_and_filters(self):
        """The conditional graph.delete(memory_text, filters) call we're bypassing."""
        try:
            from mem0.memory.main import Memory
        except ImportError:
            pytest.skip("mem0ai not installed")

        source = inspect.getsource(Memory.delete)
        assert "self.graph.delete(memory_text, filters)" in source, (
            "INVARIANT BROKEN: Memory.delete's graph cleanup branch must call "
            "self.graph.delete(memory_text, filters). If the call shape "
            "changes, our enable_graph=False bypass may need to evolve."
        )

    def test_delete_reads_payload_before_deleting(self):
        """delete reads the existing memory's payload BEFORE _delete_memory.

        Our delete wrapper depends on this ordering: we read the batch_uuid
        from the payload via mem.vector_store.get(memory_id) BEFORE invoking
        mem.delete(). If mem0ai ever moves the payload read after the delete,
        we'd race with our own deletion.
        """
        try:
            from mem0.memory.main import Memory
        except ImportError:
            pytest.skip("mem0ai not installed")

        source = inspect.getsource(Memory.delete)
        # Verify the structural ordering: vector_store.get(...) must appear
        # before _delete_memory in the source body.
        get_pos = source.find("self.vector_store.get(vector_id=memory_id)")
        delete_pos = source.find("self._delete_memory(")
        assert get_pos != -1, (
            "INVARIANT BROKEN: Memory.delete must read existing_memory via "
            "self.vector_store.get(vector_id=memory_id)."
        )
        assert delete_pos != -1, (
            "INVARIANT BROKEN: Memory.delete must call self._delete_memory."
        )
        assert get_pos < delete_pos, (
            "INVARIANT BROKEN: vector_store.get must occur before _delete_memory. "
            "Our wrapper reads payload['batch_uuid'] independently before "
            "mem.delete() runs; mem0ai's own ordering must match."
        )


class TestAddOrchestration:
    """Memory.add must submit vector and graph paths concurrently via ThreadPoolExecutor.

    The provenance design uses post-add Cypher reconciliation timed against a
    pre-add Neo4j-clock anchor. If mem0ai ever serializes graph after vector
    (or vice versa), the timing window assumptions hold — but the assumption
    that graph writes happen DURING mem.add() is load-bearing. If graph moves
    out-of-band (e.g. background queue), reconciliation would race.
    """

    def test_add_submits_vector_and_graph_concurrently(self):
        try:
            from mem0.memory.main import Memory
        except ImportError:
            pytest.skip("mem0ai not installed")

        source = inspect.getsource(Memory.add)
        # Both submissions must appear in the same ThreadPoolExecutor block
        assert "ThreadPoolExecutor" in source, (
            "INVARIANT BROKEN: Memory.add must use ThreadPoolExecutor to "
            "co-submit vector and graph paths. If graph moves out-of-band, "
            "post-add reconciliation needs redesign."
        )
        assert "_add_to_vector_store" in source, (
            "INVARIANT BROKEN: Memory.add must invoke _add_to_vector_store "
            "(or equivalent) inline."
        )
        assert "_add_to_graph" in source, (
            "INVARIANT BROKEN: Memory.add must invoke _add_to_graph "
            "(or equivalent) inline. Provenance reconciliation runs after "
            "this returns and assumes all graph writes are complete."
        )

    def test_add_to_graph_returns_added_entities(self):
        """_add_to_graph returns the list of (source, relationship, target) triples.

        Provenance Pass A narrows reconciliation by triple identity using this
        return value. If the shape changes (e.g. wrapped in another envelope),
        Pass A's UNWIND $triples query needs to track.
        """
        try:
            from mem0.memory.main import Memory
        except ImportError:
            pytest.skip("mem0ai not installed")

        source = inspect.getsource(Memory._add_to_graph)
        assert "self.graph.add(data, filters)" in source, (
            "INVARIANT BROKEN: _add_to_graph must call self.graph.add(data, filters) "
            "and return its result. Provenance design uses the returned "
            "added_entities list for triple-identity narrowing."
        )


class TestGraphAddEntitiesShape:
    """MemoryGraph._add_entities sets created/created_at/updated_at on MERGEd elements.

    Provenance Pass A's WHERE clause depends on these properties being set
    by mem0ai's own ON CREATE SET / ON MATCH SET clauses. We do NOT depend on
    the exact MERGE structure (branches, parameter names) — only on the
    timestamp property names existing.

    Live timestamp behavior (ms precision, ON MATCH advances updated_at)
    is verified in tests/integration/test_provenance_contract_live.py.
    """

    def test_add_entities_sets_node_created_timestamp(self):
        try:
            from mem0.memory.graph_memory import MemoryGraph
        except ImportError:
            pytest.skip("mem0ai not installed")

        source = inspect.getsource(MemoryGraph._add_entities)
        assert (
            "destination.created = timestamp()" in source
            or "source.created = timestamp()" in source
        ), (
            "INVARIANT BROKEN: _add_entities must set node.created = timestamp() "
            "on at least one MERGE branch. Provenance Pass B (endpoint backfill) "
            "depends on this property existing on freshly-created nodes."
        )

    def test_add_entities_sets_relation_created_at_timestamp(self):
        try:
            from mem0.memory.graph_memory import MemoryGraph
        except ImportError:
            pytest.skip("mem0ai not installed")

        source = inspect.getsource(MemoryGraph._add_entities)
        assert "r.created_at = timestamp()" in source, (
            "INVARIANT BROKEN: _add_entities must set r.created_at = timestamp() "
            "on relationship CREATE. Provenance Pass A's timestamp window "
            "depends on this."
        )

    def test_add_entities_sets_relation_updated_at_on_match(self):
        """The ON MATCH branch must set updated_at — Pass A filters on it."""
        try:
            from mem0.memory.graph_memory import MemoryGraph
        except ImportError:
            pytest.skip("mem0ai not installed")

        source = inspect.getsource(MemoryGraph._add_entities)
        assert "r.updated_at = timestamp()" in source, (
            "INVARIANT BROKEN: _add_entities must set r.updated_at on ON MATCH. "
            "Provenance Pass A's filter is "
            "(r.created_at >= $start_ts OR r.updated_at >= $start_ts) — "
            "without updated_at on match, re-merged edges would never get tagged."
        )
