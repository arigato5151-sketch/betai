PYTHON ?= python
NPM ?= npm
DOCKER_COMPOSE ?= docker compose

.PHONY: dev dev-backend dev-frontend test lint typecheck format docker-up docker-down openapi migrate migration-check migration-check-docker create-admin create-user install

install:
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt
	$(NPM) --prefix frontend install

dev:
	$(MAKE) -j2 dev-backend dev-frontend

dev-backend:
	$(PYTHON) run.py --reload

dev-frontend:
	$(NPM) --prefix frontend run dev

test:
	$(PYTHON) -m pytest tests -q

migrate:
	$(PYTHON) -m alembic -c backend/alembic.ini upgrade head

migration-check:
	$(PYTHON) -m alembic -c backend/alembic.ini check

migration-check-docker:
	$(DOCKER_COMPOSE) run --rm migration alembic -c alembic.ini check

create-admin:
	$(PYTHON) scripts/bootstrap_admin.py

create-user:
	$(PYTHON) scripts/create_user.py $(ARGS)

lint:
	$(PYTHON) -m ruff check backend/app backend/migrations tests scripts
	$(PYTHON) -m black --check backend/app backend/migrations tests scripts
	$(MAKE) typecheck
	$(NPM) --prefix frontend run lint

typecheck:
	$(PYTHON) -m mypy backend/app

format:
	$(PYTHON) -m ruff check backend/app backend/migrations tests scripts --fix
	$(PYTHON) -m black backend/app backend/migrations tests scripts

docker-up:
	$(DOCKER_COMPOSE) up --build -d

docker-down:
	$(DOCKER_COMPOSE) down

openapi:
	$(PYTHON) scripts/export_openapi.py
