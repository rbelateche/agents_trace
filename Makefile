.DEFAULT_GOAL := help

PYTHON_DIRS := sdk backend demo
FRONTEND_DIR := frontend

.PHONY: help install lint typecheck test ci dev down db-migrate frontend-lint

help:
	@echo "agents-trace — available targets:"
	@echo ""
	@echo "  install        Install all Python dev dependencies via uv"
	@echo "  lint           Run ruff check + ruff format --check"
	@echo "  typecheck      Run mypy"
	@echo "  test           Run pytest with coverage"
	@echo "  ci             lint + typecheck + test  (pre-commit gate)"
	@echo ""
	@echo "  dev            docker compose up --build"
	@echo "  down           docker compose down"
	@echo "  db-migrate     alembic upgrade head"
	@echo "  frontend-lint  eslint + tsc inside frontend/"

install:
	uv pip install --system -e ".[dev]" -e backend/ -e sdk/

lint:
	uv run ruff check $(PYTHON_DIRS)
	uv run ruff format --check $(PYTHON_DIRS)

typecheck:
	uv run mypy $(PYTHON_DIRS)

test:
	uv run pytest tests/ -v --tb=short \
		--cov=. --cov-report=xml --cov-report=term-missing \
		--ignore=$(FRONTEND_DIR) \
		|| { e=$$?; [ $$e -eq 5 ]; }

ci: lint typecheck test

dev:
	docker compose up --build

down:
	docker compose down

db-migrate:
	cd backend && uv run alembic upgrade head

frontend-lint:
	@[ -f $(FRONTEND_DIR)/package.json ] || { echo "frontend not scaffolded yet (PR 5)"; exit 0; }
	cd $(FRONTEND_DIR) && npm run lint && npx tsc --noEmit
