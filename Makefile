# -----------------------------
# Variables
# -----------------------------
COMPOSE_DEV = docker compose -f docker-compose.yaml -f docker-compose.dev.yaml
COMPOSE_PROD = docker compose -f docker-compose.yaml
SERVICE = api
SSD_MOUNT=/mnt/judge-drive

# Check if SSD is mounted
check-ssd:
	@mount | grep -q $(SSD_MOUNT) || (echo "Error: SSD is not mounted at $(SSD_MOUNT)!" && exit 1)

# -----------------------------
# Development Commands
# -----------------------------
dev: check-ssd ## Run development environment (with volume hot reload)
	$(COMPOSE_DEV) up --build

dev-detached: check-ssd ## Run development environment in detached mode
	$(COMPOSE_DEV) up --build -d

dev-down: check-ssd ## Stop dev environment
	$(COMPOSE_DEV) down --remove-orphans

dev-logs: ## Show logs
	$(COMPOSE_DEV) logs -f $(SERVICE)

dev-shell: ## Open shell inside API container
	$(COMPOSE_DEV) exec $(SERVICE) sh

# -----------------------------
# Production Commands
# -----------------------------
prod: check-ssd ## Run production environment
	$(COMPOSE_PROD) up --build

prod-down: check-ssd ## Stop production
	$(COMPOSE_PROD) down

prod-logs: check-ssd ## Production logs
	$(COMPOSE_PROD) logs -f $(SERVICE)

prod-shell: ## Shell inside API prod container
	$(COMPOSE_PROD) exec $(SERVICE) sh

# -----------------------------
# Utility Commands
# -----------------------------
rebuild-dev: ## Rebuild dev without cache
	$(COMPOSE_DEV) build --no-cache

rebuild-prod: ## Rebuild prod without cache
	$(COMPOSE_PROD) build --no-cache

restart-cv: ## Restart the computervision service
	$(COMPOSE_DEV) restart computervision

prune: ## Clean all docker trash
	docker system prune -f

help: ## Show available commands
	@grep -E '^[a-zA-Z0-9_-]+:.*?##' Makefile | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

