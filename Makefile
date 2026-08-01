.PHONY: docker-up docker-down docker-status docker-reset

# Docker Compose file
COMPOSE_FILE ?= docker-compose.yml
COMPOSE_PROJECT ?= agent-knowledge-graph

docker-up:
	@echo "Starting Neo4j..."
	docker compose -f $(COMPOSE_FILE) -p $(COMPOSE_PROJECT) up -d
	@echo "Waiting for Neo4j to become healthy..."
	@docker compose -f $(COMPOSE_FILE) -p $(COMPOSE_PROJECT) wait neo4j 2>/dev/null || \
		echo "Neo4j health check — check with 'make docker-status'"

docker-down:
	@echo "Stopping Neo4j..."
	docker compose -f $(COMPOSE_FILE) -p $(COMPOSE_PROJECT) down

docker-status:
	@docker compose -f $(COMPOSE_FILE) -p $(COMPOSE_PROJECT) ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"

docker-reset:
	@echo "Stopping and removing Neo4j containers and volumes..."
	docker compose -f $(COMPOSE_FILE) -p $(COMPOSE_PROJECT) down -v
	@echo "Done. Run 'make docker-up' to start fresh."