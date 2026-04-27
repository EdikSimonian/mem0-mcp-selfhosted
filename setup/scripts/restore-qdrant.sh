#!/usr/bin/env bash
# Restore a Qdrant snapshot into the running compose-managed Qdrant.
# Usage: setup/scripts/restore-qdrant.sh <path-to-snapshot-file>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SETUP_DIR"

if [[ -f .env ]]; then set -a; source .env; set +a; fi

if [[ -z "${MEM0_DATA_DIR:-}" ]]; then
  echo "[restore] MEM0_DATA_DIR not set — copy setup/.env.example to setup/.env first." >&2
  exit 1
fi

QDRANT_URL="${QDRANT_URL:-http://localhost:${QDRANT_REST_PORT:-6333}}"
COLLECTION="${MEM0_COLLECTION:-mem0_mcp_selfhosted}"

SNAP_PATH="${1:-}"
if [[ -z "$SNAP_PATH" || ! -f "$SNAP_PATH" ]]; then
  echo "usage: $0 <path-to-.snapshot-file>" >&2
  exit 1
fi

# Place snapshot inside the bind-mounted snapshots dir so Qdrant can see it
SNAPSHOTS_DIR="$MEM0_DATA_DIR/qdrant/snapshots"
mkdir -p "$SNAPSHOTS_DIR"
SNAP_NAME="$(basename "$SNAP_PATH")"
cp "$SNAP_PATH" "$SNAPSHOTS_DIR/$SNAP_NAME"

echo "[restore] uploading snapshot via recover endpoint"
curl -fsS -X PUT \
  "$QDRANT_URL/collections/$COLLECTION/snapshots/recover" \
  -H "Content-Type: application/json" \
  -d "{\"location\": \"file:///qdrant/snapshots/$SNAP_NAME\", \"priority\": \"snapshot\"}"
echo

echo "[restore] verifying"
curl -fsS "$QDRANT_URL/collections/$COLLECTION" | python3 -m json.tool | head -20
