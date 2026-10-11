#!/bin/sh
set -e

echo "[alea] waiting for database…"
python - <<'PY'
import time

import psycopg

from app.core.config import get_settings

settings = get_settings()
for attempt in range(60):
    try:
        with psycopg.connect(settings.checkpoint_database_url, connect_timeout=5):
            print("[alea] database is up")
            break
    except Exception as exc:  # noqa: BLE001
        if attempt == 59:
            raise SystemExit(f"[alea] database unreachable: {exc}") from exc
        time.sleep(2)
PY

echo "[alea] applying migrations…"
alembic upgrade head

if [ "${SKIP_KNOWLEDGE_INGEST:-false}" != "true" ]; then
  echo "[alea] indexing knowledge base…"
  python -m scripts.ingest_knowledge || echo "[alea] knowledge ingest skipped (check EMBEDDING_* config)"
fi

echo "[alea] starting: $*"
exec "$@"
