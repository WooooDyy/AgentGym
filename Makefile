# AgentGym Environment Makefile
# Provides commands for running and visualizing AgentGym environments
#
# Quick Start:
#   make env ENV=sciworld          # Start SciWorld environment server
#   make viz ENV=sciworld          # Start visualization with SciWorld
#   make viz-down                  # Stop all visualization services
#
# Available Environments:
#   sciworld, babyai, textcraft, searchqa, webarena

.PHONY: help env env-build env-down viz viz-build viz-down status clean

# Default environment configuration
ENV ?= sciworld
ENV_PORT ?= 36003

# Port mapping for environments
PORT_sciworld = 36003
PORT_babyai = 36002
PORT_textcraft = 36001
PORT_searchqa = 36005
PORT_webarena = 36004

# Get the port for the current environment
CURRENT_PORT = $(or $(PORT_$(ENV)),$(ENV_PORT))

help:
	@echo "AgentGym Environment Makefile"
	@echo ""
	@echo "Docker Commands:"
	@echo "  make env ENV=<name>     Start environment in Docker"
	@echo "  make viz ENV=<name>     Start visualization + environment"
	@echo "  make env-down           Stop all environments"
	@echo "  make viz-down           Stop all services"
	@echo ""
	@echo "Local Commands (no Docker):"
	@echo "  make local-env ENV=<name>         Start single environment"
	@echo "  make local-envs ENVS=\"sci bai\"    Start multiple environments"
	@echo "  make local-viz                    Start visualization"
	@echo "  make local-all ENVS=\"sci bai\"     Start everything"
	@echo ""
	@echo "Environments: sciworld(36003) babyai(36002) textcraft(36001) searchqa(36005) webarena(36004)"
	@echo ""
	@echo "Examples:"
	@echo "  make local-env ENV=sciworld"
	@echo "  make local-envs ENVS=\"sciworld babyai\" && make local-viz"

# Build environment Docker image
env-build:
	@echo "Building $(ENV) environment..."
	docker build -t agentgym/$(ENV):latest ./agentenv-$(ENV)

# Start environment server
env:
	@echo "Starting $(ENV) environment on port $(CURRENT_PORT)..."
	@docker run -d --rm \
		--name agentgym-$(ENV) \
		-p $(CURRENT_PORT):$(CURRENT_PORT) \
		-e VISUAL=true \
		agentgym/$(ENV):latest \
		|| docker start agentgym-$(ENV) 2>/dev/null \
		|| (echo "Image not found. Building..." && $(MAKE) env-build && $(MAKE) env)
	@echo ""
	@echo "$(ENV) server running at: http://localhost:$(CURRENT_PORT)"

# Stop environment server
env-down:
	@echo "Stopping environment servers..."
	-@docker stop agentgym-sciworld agentgym-babyai agentgym-textcraft agentgym-searchqa agentgym-webarena 2>/dev/null || true
	@echo "Environment servers stopped"

# Build visualization Docker image
viz-build:
	@echo "Building visualization..."
	docker build -t agentgym/visualization:latest ./env-visualization

# Start visualization with environment
viz: env-build viz-build
	@echo "Starting $(ENV) environment and visualization..."
	@docker run -d --rm \
		--name agentgym-$(ENV) \
		-p $(CURRENT_PORT):$(CURRENT_PORT) \
		-e VISUAL=true \
		agentgym/$(ENV):latest 2>/dev/null \
		|| docker start agentgym-$(ENV) 2>/dev/null \
		|| true
	@echo "Waiting for $(ENV) to start..."
	@sleep 3
	@docker run -d --rm \
		--name agentgym-visualization \
		--add-host=host.docker.internal:host-gateway \
		-p 5173:5173 \
		-e DOCKER_ENV=true \
		-e ENV=$(ENV) \
		-e ENV_HOST=host.docker.internal \
		-e ENV_PORT=$(CURRENT_PORT) \
		agentgym/visualization:latest 2>/dev/null \
		|| docker start agentgym-visualization 2>/dev/null \
		|| true
	@echo ""
	@echo "Services started:"
	@echo "  $(ENV) API:      http://localhost:$(CURRENT_PORT)"
	@echo "  Visualization:   http://localhost:5173"
	@echo ""
	@echo "Run 'make status' to check container status"
	@echo "Run 'make viz-down' to stop services"

# Stop visualization services
viz-down:
	@echo "Stopping visualization services..."
	-@docker stop agentgym-visualization 2>/dev/null || true
	-@docker stop agentgym-sciworld agentgym-babyai agentgym-textcraft agentgym-searchqa agentgym-webarena 2>/dev/null || true
	@echo "Visualization services stopped"

# Show container status
status:
	@echo "Running AgentGym containers:"
	@docker ps --filter "name=agentgym" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "No containers running"
	@echo ""
	@echo "AgentGym images:"
	@docker images --filter "reference=agentgym/*" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" 2>/dev/null || echo "No images found"

# Show environment logs
logs:
	@docker logs -f agentgym-$(ENV) 2>/dev/null || echo "Container agentgym-$(ENV) not running"

# Clean up Docker images
clean:
	@echo "Removing AgentGym Docker images..."
	-@docker rmi agentgym/visualization:latest 2>/dev/null || true
	-@docker rmi agentgym/sciworld:latest 2>/dev/null || true
	-@docker rmi agentgym/babyai:latest 2>/dev/null || true
	-@docker rmi agentgym/textcraft:latest 2>/dev/null || true
	-@docker rmi agentgym/searchqa:latest 2>/dev/null || true
	-@docker rmi agentgym/webarena:latest 2>/dev/null || true
	@echo "Images removed"

# Local development (without Docker)
.PHONY: local-env local-viz local-envs local-all

# Helper to get port for an environment
define get_port
$(or $(PORT_$(1)),36000)
endef

local-env:
	@echo "Starting $(ENV) on port $(CURRENT_PORT)..."
	@cd agentenv-$(ENV) && \
		(test -d .venv || uv venv .venv) && \
		uv pip install -e . && \
		VISUAL=true uv run $(ENV) --host 0.0.0.0 --port $(CURRENT_PORT)

local-viz:
	@echo "Starting visualization at http://localhost:5173"
	@cd env-visualization && npm install && npm run dev -- --host 0.0.0.0

# Start multiple environments in background
# Usage: make local-envs ENVS="sciworld babyai"
ENVS ?= sciworld

local-envs:
	@echo "Starting environments: $(ENVS)"
	@for env in $(ENVS); do \
		$(MAKE) -s _start-env-bg ENV=$$env; \
	done
	@echo "\nEnvironments running. Start visualization: make local-viz"

_start-env-bg:
	@echo "  → $(ENV) on port $(CURRENT_PORT)"
	@(cd agentenv-$(ENV) && \
		(test -d .venv || uv venv .venv) && \
		uv pip install -e . >/dev/null 2>&1 && \
		VISUAL=true uv run $(ENV) --host 0.0.0.0 --port $(CURRENT_PORT) &) 2>/dev/null

# Start environments + visualization together
local-all: local-envs
	@sleep 3
