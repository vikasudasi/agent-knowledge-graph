#!/usr/bin/env bash
# Helper to manage Neo4j lifecycle
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$PROJECT_DIR/docker-compose.yml"

cmd="${1:-help}"

case "$cmd" in
  up)
    echo "Starting Neo4j..."
    docker compose -f "$COMPOSE_FILE" up -d
    echo "Waiting for Neo4j to be ready..."
    for i in $(seq 1 30); do
      if docker compose -f "$COMPOSE_FILE" exec neo4j cypher-shell -u "${KG_NEO4J_USER:-neo4j}" -p "${KG_NEO4J_PASSWORD:-password}" "RETURN 1" >/dev/null 2>&1; then
        echo "Neo4j is ready!"
        exit 0
      fi
      sleep 2
    done
    echo "Timed out waiting for Neo4j"
    exit 1
    ;;
  down)
    echo "Stopping Neo4j..."
    docker compose -f "$COMPOSE_FILE" down
    ;;
  status)
    docker compose -f "$COMPOSE_FILE" ps
    ;;
  logs)
    docker compose -f "$COMPOSE_FILE" logs "${2:--f}"
    ;;
  reset)
    echo "Removing Neo4j data volumes..."
    docker compose -f "$COMPOSE_FILE" down -v
    echo "Starting fresh..."
    "$0" up
    ;;
  *)
    echo "Usage: $0 {up|down|status|logs|reset}"
    exit 1
    ;;
esac
