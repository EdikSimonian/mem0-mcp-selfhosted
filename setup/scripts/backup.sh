#!/usr/bin/env bash
# Snapshot Qdrant + Neo4j into $MEM0_DATA_DIR/backups/<timestamp>/.
# Idempotent and safe to run while services are up.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SETUP_DIR"

# Load compose .env so MEM0_DATA_DIR + Neo4j creds are available
if [[ -f .env ]]; then set -a; source .env; set +a; fi

if [[ -z "${MEM0_DATA_DIR:-}" ]]; then
  echo "[backup] MEM0_DATA_DIR not set — copy setup/.env.example to setup/.env first." >&2
  exit 1
fi

QDRANT_URL="${QDRANT_URL:-http://localhost:${QDRANT_REST_PORT:-6333}}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-mem0graph}"
COLLECTION="${MEM0_COLLECTION:-mem0_mcp_selfhosted}"

TS="$(date +%Y%m%d-%H%M%S)"
DEST="$MEM0_DATA_DIR/backups/$TS"
mkdir -p "$DEST/qdrant" "$DEST/neo4j"

echo "[backup] target: $DEST"

# ---------- Qdrant ----------
echo "[backup] qdrant: capturing collection config + snapshot for '$COLLECTION'"
curl -fsS "$QDRANT_URL/collections/$COLLECTION" > "$DEST/qdrant/collection-config.json"

SNAP_NAME=$(curl -fsS -X POST "$QDRANT_URL/collections/$COLLECTION/snapshots" \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["result"]["name"])')

curl -fsS -o "$DEST/qdrant/$SNAP_NAME" \
  "$QDRANT_URL/collections/$COLLECTION/snapshots/$SNAP_NAME"

echo "[backup] qdrant: $DEST/qdrant/$SNAP_NAME"

# ---------- Neo4j ----------
if docker ps --format '{{.Names}}' | grep -q '^memory-neo4j$'; then
  echo "[backup] neo4j: dumping graph via APOC (online)"
  docker exec memory-neo4j cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" \
    "CALL apoc.export.cypher.all('/var/lib/neo4j/import/backup-$TS.cypher', {format:'cypher-shell', useOptimizations:{type:'UNWIND_BATCH', unwindBatchSize:100}})" \
    > /dev/null
  cp "$MEM0_DATA_DIR/neo4j/import/backup-$TS.cypher" "$DEST/neo4j/graph.cypher"
  docker exec memory-neo4j rm -f "/var/lib/neo4j/import/backup-$TS.cypher"
  echo "[backup] neo4j: $DEST/neo4j/graph.cypher"
else
  echo "[backup] neo4j: container not running, skipping"
fi

# ---------- Manifest ----------
cat > "$DEST/manifest.json" <<EOF
{
  "timestamp": "$TS",
  "qdrant": {
    "url": "$QDRANT_URL",
    "collection": "$COLLECTION",
    "snapshot": "qdrant/$SNAP_NAME"
  },
  "neo4j": {
    "user": "$NEO4J_USER",
    "dump": "neo4j/graph.cypher"
  }
}
EOF

echo "[backup] done -> $DEST"
