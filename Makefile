# Development tasks for laya-server.
# Run `make` or `make help` to list targets.

HOST ?= 127.0.0.1
PORT ?= 8000
# Checkpoints to load at startup; empty means all three.
PRELOAD ?=

# Everything this project owns; all of it is lint-clean.
SOURCES := src tests scripts

IMAGE ?= laya-server
TAG   ?= dev
# Named volume for checkpoints, so the download survives container restarts.
CACHE_VOLUME ?= laya-checkpoints

# Docker Hub publishing. Override any of these on the command line.
REGISTRY  ?= docker.io
NAMESPACE ?= pambrose
PLATFORMS ?= linux/amd64,linux/arm64
VERSION   := $(shell grep -m1 '^version' pyproject.toml | cut -d'"' -f2)
IMAGE_REF := $(REGISTRY)/$(NAMESPACE)/$(IMAGE)

UVICORN := uv run uvicorn laya_server.app:app --host $(HOST) --port $(PORT)
ENV := LAYA_SERVER_PRELOAD=$(PRELOAD)

.DEFAULT_GOAL := help
.PHONY: help install lock check-lock lint format test test-slow test-all \
        run dev smoke docker-build docker-run docker-buildx docker-login \
        docker-push build clean ci

help: ## List available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk -F':.*?## ' '{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies into .venv
	uv sync

lock: ## Refresh uv.lock
	uv lock

check-lock: ## Fail if uv.lock is stale (what CI enforces)
	uv sync --locked

lint: ## Check style and formatting without changing anything
	uv run ruff check $(SOURCES)
	uv run ruff format --check $(SOURCES)

format: ## Apply autofixes and reformat
	uv run ruff check --fix $(SOURCES)
	uv run ruff format $(SOURCES)

test: ## Run the fast suite (stubs Laya, no model download)
	uv run pytest -m "not slow"

test-slow: ## Run the suite that loads a real checkpoint (~843MB on first run)
	uv run pytest -m slow

test-all: ## Run every test
	uv run pytest

run: ## Serve the API (override with HOST=, PORT=, PRELOAD=)
	$(ENV) $(UVICORN)

dev: ## Serve with auto-reload
	$(ENV) $(UVICORN) --reload

smoke: ## POST a sample request to an already-running server
	@resp=$$(curl -sf -X POST http://$(HOST):$(PORT)/v1/systemone \
		-H 'Content-Type: application/json' \
		-d '{"model":"jev-latest", \
		     "state":{"body":"We were billed twice for March. Refund it or we cancel."}, \
		     "questions":{"refund":{"type":"noul","instructions":"Is a refund requested?"}}}') \
		|| { echo "no server on $(HOST):$(PORT) -- start one with 'make run'"; exit 1; }; \
	echo "$$resp" | uv run python -m json.tool

docker-build: ## Build the container image
	docker build -t $(IMAGE):$(TAG) .

docker-run: ## Run the image, reusing a named checkpoint cache volume
	docker run --rm -p $(PORT):8000 \
		-v $(CACHE_VOLUME):/home/app/.cache/huggingface \
		-e LAYA_SERVER_PRELOAD=$(PRELOAD) \
		$(IMAGE):$(TAG)

docker-buildx: ## Cross-build for every target platform without pushing
	docker buildx build --platform $(PLATFORMS) -t $(IMAGE_REF):$(VERSION) .

docker-login: ## Log in to the registry (needed once before docker-push)
	docker login $(REGISTRY)

docker-push: ## Build multi-arch and PUBLISH to the registry (public!)
	@echo "About to publish:"
	@echo "  $(IMAGE_REF):$(VERSION)"
	@echo "  $(IMAGE_REF):latest"
	@echo "  platforms: $(PLATFORMS)"
	@printf 'Continue? [y/N] ' && read ans && [ "$$ans" = "y" ]
	docker buildx build --platform $(PLATFORMS) \
		-t $(IMAGE_REF):$(VERSION) \
		-t $(IMAGE_REF):latest \
		--push .

build: ## Build the sdist and wheel
	uv build

clean: ## Remove build and test artifacts
	rm -rf dist build .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -not -path './.venv/*' -exec rm -rf {} +
	find . -name '*.egg-info' -type d -not -path './.venv/*' -exec rm -rf {} +

ci: check-lock lint test ## Run what CI runs on a pull request
