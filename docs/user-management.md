# User Management & Multi-Tenant Dashboard

This document describes the web dashboard, multi-user isolation model, and how to migrate an existing local graph to user ownership.

## Overview

The web layer adds:

- Email/password accounts with a browser dashboard
- Per-user **graphs** identified by a UUID `graph_id`
- **Agent API keys** (`kg_` + 40 hex chars) for authenticated MCP access
- An HTTP MCP endpoint at `/mcp` scoped to each agent's permitted graphs

The existing local CLI and stdio MCP server remain unchanged. When `graph_id` is not passed to `Neo4jClient`, behavior matches the original single-user local graph (including nodes that have no `graph_id` property).

## Architecture

| Component | Role |
|---|---|
| `web/db.py` | SQLite metadata: users, agents, graphs, agent↔graph scope |
| `web/auth.py` | bcrypt passwords, signed session cookies, agent key hashing |
| `web/app.py` | FastAPI dashboard routes |
| `web/mcp_http.py` | Authenticated streamable-HTTP + SSE MCP |
| `core/graph.py` | Optional `graph_id` / `graph_ids` scoping on Neo4j operations |

Metadata lives in `{storage.data_dir}/metadata.db` (default `~/.local/share/agent-knowledge-graph/metadata.db`).

## Graph isolation in Neo4j

All tenants share one Neo4j database. Isolation is enforced by a `graph_id` property on:

- `(:Resource)` nodes
- `(:PipelineCheckpoint)` nodes

Web-created graphs use a UUID hex string as `graph_id`. Legacy local data has **no** `graph_id` and is treated as the implicit **`default`** graph when explicitly querying with `graph_id="default"`.

### Vector search tradeoff (v1)

For semantic search, Neo4j returns the top-K vector hits first, then applies `WHERE n.graph_id = $graph_id`. That means fewer than `top_k` results may be returned when many globally similar nodes belong to other graphs. This is acceptable for v1.

## Running the web dashboard

```bash
pip install -e ".[web]"
kg web serve --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000`, sign up, create a graph, create an agent key, and follow **Setup** for MCP client snippets.

## Agent keys

- Raw keys are shown **once** at creation.
- Only a SHA-256 hash is stored in SQLite.
- Revoked keys are rejected by `/mcp`.
- Agents may be scoped to specific graphs; if none are selected, the agent can access all graphs owned by the user.

## MCP authentication

Clients send the agent key via either:

- `Authorization: Bearer kg_...`
- `X-Agent-Key: kg_...`

Invalid, missing, or revoked keys receive an error response from the MCP layer.

## Migration path (manual — do not auto-run)

To assign an existing local graph (nodes without `graph_id`) to a specific user graph:

1. **Create the user and graph** in the web dashboard. Note the new graph's UUID (`graph_id`).

2. **Back up Neo4j** before any bulk update.

3. **Assign `graph_id` to existing Resource nodes** (run in Neo4j Browser or `cypher-shell`):

   ```cypher
   MATCH (r:Resource)
   WHERE r.graph_id IS NULL
   SET r.graph_id = $target_graph_id
   RETURN count(r) AS updated_nodes;
   ```

   Replace `$target_graph_id` with the UUID from step 1.

4. **Assign checkpoints** if you use pipeline checkpoints:

   ```cypher
   MATCH (c:PipelineCheckpoint)
   WHERE c.graph_id IS NULL
   SET c.graph_id = $target_graph_id
   RETURN count(c) AS updated_checkpoints;
   ```

5. **Verify** scoped stats from the dashboard and via MCP using an agent key scoped to that graph.

6. **Optional cleanup**: After verifying, local CLI/MCP without `graph_id` will still see all nodes (including migrated ones). To keep CLI on a separate default graph, do not migrate nodes you want to remain local-only.

### Rollback

If needed, remove `graph_id` from affected nodes:

```cypher
MATCH (n)
WHERE n.graph_id = $target_graph_id
REMOVE n.graph_id
RETURN count(n) AS reverted;
```

Always restore from backup if the migration result is unexpected.

## Local default graph preservation

The stdio MCP entrypoint (`adapters/mcp_server_entry.py`) and CLI commands continue to use `Neo4jClient` without `graph_id` filtering. No changes are required for existing local workflows.
