"""Start a real PostgreSQL + pgvector instance for local development.

Docker Compose is the supported way to run the stack (see ``docker-compose.yml``).
This script exists for machines without Docker: it drives the PostgreSQL
binaries that the optional ``pgserver`` wheel ships, on a *fixed* port, so the
same ``.env`` works for both paths.

    python -m scripts.local_postgres start     # start (idempotent)
    python -m scripts.local_postgres status
    python -m scripts.local_postgres stop
    python -m scripts.local_postgres delete    # stop and remove the data dir

Unlike ``pgserver.get_server`` this pins the port and keeps the cluster across
process restarts, which is what a development database needs.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / ".data"
PGDATA = DATA_DIR / "pgdata"
LOGFILE = DATA_DIR / "postgres.log"

DEFAULT_PORT = int(os.environ.get("LOCAL_PG_PORT", "55432"))
HOST = "127.0.0.1"
SUPERUSER = "alea"
DATABASE = os.environ.get("LOCAL_PG_DATABASE", "alea")


def _bin_dir() -> Path:
    try:
        from pgserver._commands import POSTGRES_BIN_PATH  # type: ignore[attr-defined]

        return Path(POSTGRES_BIN_PATH)
    except Exception:  # noqa: BLE001 - fall back to a system install
        found = shutil.which("initdb")
        if not found:
            sys.exit(
                "找不到 PostgreSQL 二进制。请安装可选依赖 `pip install pgserver`，或改用 `docker compose up -d db`。"
            )
        return Path(found).parent


def _exe(name: str) -> str:
    candidate = _bin_dir() / (name + (".exe" if os.name == "nt" else ""))
    if not candidate.exists():
        sys.exit(f"缺少可执行文件：{candidate}")
    return str(candidate)


def _run(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=check, capture_output=True, text=True)


def _run_detached(args: list[str]) -> subprocess.CompletedProcess[bytes]:
    """Run a command that spawns a daemon.

    ``pg_ctl start`` hands its stdio to the postgres process it launches, so a
    capturing pipe would never see EOF and the call would hang forever on
    Windows. DEVNULL avoids that entirely.
    """
    return subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def is_running() -> bool:
    if not (PGDATA / "postmaster.pid").exists():
        return False
    result = _run([_exe("pg_ctl"), "-D", str(PGDATA), "status"], check=False)
    return result.returncode == 0


def init_cluster() -> None:
    if (PGDATA / "PG_VERSION").exists():
        return
    PGDATA.parent.mkdir(parents=True, exist_ok=True)
    print(f"初始化数据目录 {PGDATA} …")
    result = _run(
        [
            _exe("initdb"),
            "-D",
            str(PGDATA),
            "-U",
            SUPERUSER,
            "--auth=trust",
            "--auth-local=trust",
            "--encoding=utf8",
            "--locale=C",
        ]
    )
    print(result.stdout.strip().splitlines()[-1] if result.stdout else "initdb ok")


def ensure_database(port: int) -> None:
    psql = _exe("psql")
    base = [psql, "-h", HOST, "-p", str(port), "-U", SUPERUSER, "-d", "postgres"]
    exists = _run(
        [*base, "-tAc", f"SELECT 1 FROM pg_database WHERE datname = '{DATABASE}'"],
        check=False,
    )
    if exists.stdout.strip() != "1":
        _run([*base, "-c", f'CREATE DATABASE "{DATABASE}" OWNER "{SUPERUSER}"'])
        print(f"已创建数据库 {DATABASE}")


def start(port: int) -> int:
    init_cluster()
    if is_running():
        print(f"PostgreSQL 已在运行（数据目录 {PGDATA}）")
    else:
        PGDATA.parent.mkdir(parents=True, exist_ok=True)
        result = _run_detached(
            [
                _exe("pg_ctl"),
                "-D",
                str(PGDATA),
                "-l",
                str(LOGFILE),
                "-o",
                f"-p {port} -h {HOST}",
                "-w",
                "-t",
                "30",
                "start",
            ]
        )
        if result.returncode != 0:
            print(LOGFILE.read_text(encoding="utf-8", errors="replace")[-2000:])
            sys.exit("PostgreSQL 启动失败，详见 " + str(LOGFILE))
        print(f"PostgreSQL 已启动，端口 {port}")
        time.sleep(0.5)
    ensure_database(port)
    print()
    print(f"DATABASE_URL=postgresql+asyncpg://{SUPERUSER}:{SUPERUSER}@{HOST}:{port}/{DATABASE}")
    print(f"CHECKPOINT_DATABASE_URL=postgresql://{SUPERUSER}:{SUPERUSER}@{HOST}:{port}/{DATABASE}")
    return 0


def stop() -> int:
    if not (PGDATA / "PG_VERSION").exists():
        print("尚无本地数据目录")
        return 0
    result = _run([_exe("pg_ctl"), "-D", str(PGDATA), "-m", "fast", "-w", "stop"], check=False)
    print("已停止" if result.returncode == 0 else result.stderr.strip())
    return 0


def status(port: int) -> int:
    running = is_running()
    print("running" if running else "stopped")
    if running:
        print(f"psql -h {HOST} -p {port} -U {SUPERUSER} -d {DATABASE}")
    return 0 if running else 1


def delete() -> int:
    stop()
    if PGDATA.exists():
        shutil.rmtree(PGDATA, ignore_errors=True)
        print(f"已删除 {PGDATA}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Local PostgreSQL + pgvector helper")
    parser.add_argument("action", choices=["start", "stop", "status", "delete"])
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    match args.action:
        case "start":
            return start(args.port)
        case "stop":
            return stop()
        case "status":
            return status(args.port)
        case "delete":
            return delete()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
