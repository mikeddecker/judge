# -----------------------------
# Variables
# -----------------------------
COMPOSE_DEV = docker compose -f docker-compose.yaml -f docker-compose.dev.yaml
COMPOSE_PROD = docker compose -f docker-compose.yaml -f docker-compose.prod.yaml
SERVICE = api
SSD_MOUNT=/mnt/judge-drive

# Check if SSD is mounted
check-ssd:
	@mount | grep -q $(SSD_MOUNT) || (echo "Error: SSD is not mounted at $(SSD_MOUNT)!" && exit 1)

docker-pull:
	docker pull python:3.12-slim
	docker pull node:20-alpine
	docker pull nginx:alpine
	docker pull mysql:8.4

# -----------------------------
# Development Commands
# -----------------------------
dev: check-ssd ## Run development environment (with volume hot reload)
	$(COMPOSE_DEV) --profile dev up

ci-dev-test: ## Run the tests
	$(COMPOSE_DEV) --profile test up --abort-on-container-exit --exit-code-from api-test

dev-detached: check-ssd ## Run development environment in detached mode
	$(COMPOSE_DEV) --profile dev up -d

dev-down: check-ssd ## Stop dev environment
	$(COMPOSE_DEV) --profile dev down --remove-orphans -v

dev-force-recreate: check-ssd
	$(COMPOSE_DEV) --profile dev up --build --force-recreate

dev-logs: ## Show logs
	$(COMPOSE_DEV) --profile dev logs -f $(SERVICE)

dev-bash: ## Open shell inside API container
	$(COMPOSE_DEV) exec -it $(SERVICE) bash

# -----------------------------
# Production Commands
# -----------------------------
prod: check-ssd ## Run production environment
	$(COMPOSE_PROD) --profile prod up --build

prod-force-recreate: check-ssd
	$(COMPOSE_PROD) --profile prod up --build --force-recreate

prod-down: check-ssd ## Stop production
	$(COMPOSE_PROD) --profile prod down -v

prod-logs: check-ssd ## Production logs
	$(COMPOSE_PROD) logs -f $(SERVICE)

prod-bash: ## Shell inside API prod container
	$(COMPOSE_PROD) exec -it $(SERVICE) bash

# -----------------------------
# Utility Commands
# -----------------------------
rebuild-dev: ## Rebuild dev without cache
	$(COMPOSE_DEV) --profile dev build --no-cache

rebuild-prod: ## Rebuild prod without cache
	$(COMPOSE_PROD) --profile prod build --no-cache

restart-cv-dev: ## Restart the computervision service
	$(COMPOSE_DEV) restart computervision

restart-api-dev: ## Restart the API service
	$(COMPOSE_DEV) restart api

restart-cv-prod: ## Restart the computervision service
	$(COMPOSE_PROD) restart computervision

prune: ## Clean all docker trash
	docker system prune -f
	docker volume prune

help: ## Show available commands
	@grep -E '^[a-zA-Z0-9_-]+:.*?##' Makefile | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

