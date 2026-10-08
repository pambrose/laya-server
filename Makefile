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

# `make release` only publishes from an up-to-date checkout of this branch.
RELEASE_BRANCH ?= master
# The release smoke test runs the image here, clear of a dev server on PORT.
SMOKE_PORT    ?= 8001
# Seconds to wait for /ready; the first run downloads a checkpoint (~843MB).
SMOKE_TIMEOUT ?= 900

UVICORN := uv run uvicorn laya_server.app:app --host $(HOST) --port $(PORT)
ENV := LAYA_SERVER_PRELOAD=$(PRELOAD)

.DEFAULT_GOAL := help
.PHONY: help install lock check-lock lint typecheck zizmor format toc test test-slow test-all \
        run dev smoke docker-build docker-run docker-smoke docker-login docker-buildx \
        docker-push docker-verify release release-check build clean ci

help: ## List available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk -F':.*?## ' '{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies into .venv
	uv sync

lock: ## Refresh uv.lock
	uv lock

check-lock: ## Fail if uv.lock is stale (what CI enforces)
	uv sync --locked

lint: ## Check style, formatting, and that the README TOC is current
	uv run ruff check $(SOURCES)
	uv run ruff format --check $(SOURCES)
	uv run python scripts/toc.py --check

typecheck: ## Type-check src/ with mypy (strict)
	uv run mypy

# With GH_TOKEN set (e.g. GH_TOKEN=$$(gh auth token) make zizmor) it also runs the
# online audits CI runs; without one it runs only the offline audits.
zizmor: ## Audit GitHub Actions workflows and dependabot.yml for security issues
	uv run zizmor .github/

format: ## Apply autofixes, reformat, and regenerate the README TOC
	uv run ruff check --fix $(SOURCES)
	uv run ruff format $(SOURCES)
	uv run python scripts/toc.py

toc: ## Regenerate the README table of contents
	uv run python scripts/toc.py

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

docker-smoke: ## Run the local image and check it answers a real request
	@name=laya-smoke-$$$$; \
	docker run -d --rm --name $$name -p $(SMOKE_PORT):8000 \
		-v $(CACHE_VOLUME):/home/app/.cache/huggingface \
		-e LAYA_SERVER_PRELOAD=english \
		$(IMAGE):$(TAG) >/dev/null || exit 1; \
	trap 'docker stop $$name >/dev/null 2>&1' EXIT; \
	echo "waiting up to $(SMOKE_TIMEOUT)s for $(IMAGE):$(TAG) to become ready..."; \
	for i in $$(seq 1 $(SMOKE_TIMEOUT)); do \
		curl -sf http://127.0.0.1:$(SMOKE_PORT)/ready >/dev/null && break; sleep 1; \
	done; \
	curl -sf http://127.0.0.1:$(SMOKE_PORT)/ready \
		|| { echo "not ready after $(SMOKE_TIMEOUT)s"; docker logs --tail 30 $$name; exit 1; }; \
	echo; \
	$(MAKE) --no-print-directory smoke HOST=127.0.0.1 PORT=$(SMOKE_PORT)

docker-login: ## Log in to the registry (needed once before docker-push)
	docker login $(REGISTRY)

docker-buildx: ## Cross-build for every target platform without pushing
	docker buildx build --platform $(PLATFORMS) -t $(IMAGE_REF):$(VERSION) .

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

docker-verify: ## Check the published VERSION carries every platform in PLATFORMS
	@out=$$(docker buildx imagetools inspect $(IMAGE_REF):$(VERSION)) || exit 1; \
	for platform in $$(echo $(PLATFORMS) | tr ',' ' '); do \
		echo "$$out" | grep -q "Platform: *$$platform" \
			|| { echo "$(IMAGE_REF):$(VERSION) is missing $$platform"; exit 1; }; \
	done; \
	echo "$(IMAGE_REF):$(VERSION) published for $(PLATFORMS)"

release-check: ## Fail unless this checkout is fit to publish VERSION
	@test -z "$$(git status --porcelain)" \
		|| { echo "working tree is dirty; commit or stash first"; exit 1; }
	@branch=$$(git rev-parse --abbrev-ref HEAD); [ "$$branch" = "$(RELEASE_BRANCH)" ] \
		|| { echo "releases are cut from $(RELEASE_BRANCH), not $$branch"; exit 1; }
	@git fetch -q origin $(RELEASE_BRANCH) \
		&& [ "$$(git rev-parse HEAD)" = "$$(git rev-parse origin/$(RELEASE_BRANCH))" ] \
		|| { echo "HEAD is not origin/$(RELEASE_BRANCH); pull or push first"; exit 1; }
	@grep -q '^## \[$(VERSION)\]' CHANGELOG.md \
		|| { echo "CHANGELOG.md has no [$(VERSION)] section"; exit 1; }
	@tagged=$$(git rev-parse -q --verify "refs/tags/$(VERSION)^{commit}"); \
	[ -z "$$tagged" ] || [ "$$tagged" = "$$(git rev-parse HEAD)" ] \
		|| { echo "tag $(VERSION) points at another commit"; exit 1; }
	@! docker buildx imagetools inspect $(IMAGE_REF):$(VERSION) >/dev/null 2>&1 \
		|| { echo "$(IMAGE_REF):$(VERSION) is already published; bump the version"; exit 1; }
	@echo "ready to release $(VERSION) from $$(git rev-parse --short HEAD)"

# Each step runs only if the one before it passed, so nothing is published
# until the code passes CI and the slow suite and the image has served a real
# request. docker-push still asks before publishing.
release: ## Gate, test, smoke-test the image, then publish VERSION and latest
	$(MAKE) release-check
	$(MAKE) ci
	$(MAKE) test-slow
	$(MAKE) docker-build TAG=$(VERSION)
	$(MAKE) docker-smoke TAG=$(VERSION)
	$(MAKE) docker-push
	$(MAKE) docker-verify
	@echo
	@echo "Published $(IMAGE_REF):$(VERSION) and :latest. Next: tag $(VERSION) and"
	@echo "create the GitHub release v$(VERSION)."

build: ## Build the sdist and wheel
	uv build

clean: ## Remove build and test artifacts
	rm -rf dist build .pytest_cache .ruff_cache .mypy_cache
	find . -name __pycache__ -type d -not -path './.venv/*' -exec rm -rf {} +
	find . -name '*.egg-info' -type d -not -path './.venv/*' -exec rm -rf {} +

ci: check-lock lint typecheck zizmor test ## Run what CI runs on a pull request
