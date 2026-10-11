# Alea — developer shortcuts.
#
#   make setup     install backend deps, start the database, migrate, index
#   make api       run the FastAPI server
#   make web       run the Next.js dev server
#   make test      run the whole backend test suite
#
# Windows note: the venv interpreter lives in .venv/Scripts; on macOS/Linux in
# .venv/bin. PY is resolved below so both work.

ifeq ($(OS),Windows_NT)
PY := backend/.venv/Scripts/python.exe
else
PY := backend/.venv/bin/python
endif

.PHONY: help setup venv db db-stop migrate ingest api web test test-unit test-llm lint fmt typecheck clean

help:
	@echo "setup      create venv, install deps, start db, migrate, index knowledge"
	@echo "db         start local PostgreSQL + pgvector (no Docker required)"
	@echo "db-stop    stop it"
	@echo "migrate    alembic upgrade head"
	@echo "ingest     index the knowledge base"
	@echo "api        run the backend on :8000"
	@echo "web        run the frontend on :3000"
	@echo "test       full backend suite (requires db; set ALEA_REQUIRE_DB=1 to fail loudly)"
	@echo "test-unit  tests that need neither database nor network"
	@echo "test-llm   tests against the real model endpoint (costs tokens)"
	@echo "lint       ruff + mypy + tsc"

venv:
	python -m venv backend/.venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e "backend[dev]"

db:
	cd backend && ../$(PY) -m scripts.local_postgres start

db-stop:
	cd backend && ../$(PY) -m scripts.local_postgres stop

migrate:
	cd backend && ../$(PY) -m alembic upgrade head

ingest:
	cd backend && ../$(PY) -m scripts.ingest_knowledge

setup: venv db migrate ingest

api:
	cd backend && ../$(PY) -m scripts.serve --reload

web:
	cd frontend && npm run dev

test:
	cd backend && ../$(PY) -m pytest -v

test-unit:
	cd backend && ../$(PY) -m pytest -m "not db and not llm" -v

test-llm:
	cd backend && ALEA_REQUIRE_DB=1 ../$(PY) -m pytest -m llm -v

lint:
	cd backend && ../$(PY) -m ruff check app tests scripts
	cd backend && ../$(PY) -m mypy app
	cd frontend && npm run typecheck

fmt:
	cd backend && ../$(PY) -m ruff check --fix app tests scripts
	cd backend && ../$(PY) -m ruff format app tests scripts

clean:
	rm -rf backend/.pytest_cache backend/.ruff_cache backend/.mypy_cache
	rm -rf frontend/.next
