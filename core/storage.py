"""Vahid saxlama qatı: DATABASE_URL varsa Postgres (Neon), yoxdursa lokal fayllar.

Üç növ:
  cədvəl  — write_table / read_table           (Postgres cədvəli  | storage/processed/<ad>.parquet)
  açar    — kv_get / kv_set                    (kv cədvəli, JSONB | storage/runtime/<açar>.json)
  fayl    — blob_put / blob_get / blob_list    (blobs, BYTEA      | storage/blobs/<açar>)

Bulud konteyneri (Render) diskə güvənmir: bütün vəziyyət burada saxlanılır.
"""
from __future__ import annotations

import io
import json
import logging
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd

from core.config import ROOT

log = logging.getLogger(__name__)
LOCAL = ROOT / "storage"


def db_url() -> str | None:
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        return None
    for p in ("postgres://", "postgresql://"):
        if url.startswith(p):
            return "postgresql+psycopg://" + url[len(p):]
    return url


def using_db() -> bool:
    return db_url() is not None


@lru_cache(maxsize=1)
def engine():
    from sqlalchemy import create_engine, text
    eng = create_engine(db_url(), pool_pre_ping=True, pool_size=3, max_overflow=2, pool_recycle=300)
    with eng.begin() as c:
        c.execute(text("CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value JSONB NOT NULL, "
                       "updated_at TIMESTAMP NOT NULL DEFAULT now())"))
        c.execute(text("CREATE TABLE IF NOT EXISTS blobs (key TEXT PRIMARY KEY, data BYTEA NOT NULL, "
                       "updated_at TIMESTAMP NOT NULL DEFAULT now())"))
    return eng


def _local(sub: str, name: str) -> Path:
    p = LOCAL / sub / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------------------------ cədvəllər
def write_table(name: str, df: pd.DataFrame) -> None:
    """Cədvəli tam əvəz edir. Postgres-də COPY ilə (sürətli)."""
    if not using_db():
        df.to_parquet(_local("processed", f"{name}.parquet"), index=False)
        return
    from sqlalchemy import text
    eng = engine()
    tmp = f"{name}__new"
    with eng.begin() as c:
        c.execute(text(f'DROP TABLE IF EXISTS "{tmp}"'))
    df.head(0).to_sql(tmp, eng, index=False)
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, na_rep="\\N")
    buf.seek(0)
    cols = ", ".join(f'"{c}"' for c in df.columns)
    raw = eng.raw_connection()
    try:
        with raw.cursor() as cur:
            with cur.copy(f'COPY "{tmp}" ({cols}) FROM STDIN WITH (FORMAT csv, NULL \'\\N\')') as cp:
                while chunk := buf.read(1 << 20):
                    cp.write(chunk)
        raw.commit()
    finally:
        raw.close()
    with eng.begin() as c:  # atomik əvəzləmə: oxuyanlar heç vaxt yarımçıq cədvəl görmür
        c.execute(text(f'DROP TABLE IF EXISTS "{name}"'))
        c.execute(text(f'ALTER TABLE "{tmp}" RENAME TO "{name}"'))
    log.info("Postgres: %s ← %d sətir", name, len(df))


def read_table(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    if not using_db():
        p = _local("processed", f"{name}.parquet")
        if not p.exists():
            raise FileNotFoundError(name)
        return pd.read_parquet(p, columns=columns)
    from sqlalchemy import inspect
    eng = engine()
    if not inspect(eng).has_table(name):
        raise FileNotFoundError(name)
    cols = "*" if not columns else ", ".join(f'"{c}"' for c in columns)
    with eng.connect() as c:
        return pd.read_sql(f'SELECT {cols} FROM "{name}"', c)


def has_table(name: str) -> bool:
    if not using_db():
        return _local("processed", f"{name}.parquet").exists()
    from sqlalchemy import inspect
    return inspect(engine()).has_table(name)


# ------------------------------------------------------------------ açar–dəyər
def kv_get(key: str, default=None):
    if not using_db():
        p = _local("runtime", f"{key.replace('/', '__')}.json")
        if not p.exists():
            return default
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return default
    from sqlalchemy import text
    with engine().connect() as c:
        r = c.execute(text("SELECT value FROM kv WHERE key = :k"), {"k": key}).fetchone()
    return default if r is None else r[0]


def kv_set(key: str, value) -> None:
    payload = json.dumps(value, ensure_ascii=False, default=str)
    if not using_db():
        _local("runtime", f"{key.replace('/', '__')}.json").write_text(payload, encoding="utf-8")
        return
    from sqlalchemy import text
    with engine().begin() as c:
        c.execute(text("INSERT INTO kv (key, value, updated_at) VALUES (:k, CAST(:v AS JSONB), now()) "
                       "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()"),
                  {"k": key, "v": payload})


# ------------------------------------------------------------------ fayllar
def blob_put(key: str, data: bytes) -> None:
    if not using_db():
        _local("blobs", key).write_bytes(data)
        return
    from sqlalchemy import text
    with engine().begin() as c:
        c.execute(text("INSERT INTO blobs (key, data, updated_at) VALUES (:k, :d, now()) "
                       "ON CONFLICT (key) DO UPDATE SET data = EXCLUDED.data, updated_at = now()"),
                  {"k": key, "d": data})


def blob_get(key: str) -> bytes | None:
    if not using_db():
        p = _local("blobs", key)
        return p.read_bytes() if p.exists() else None
    from sqlalchemy import text
    with engine().connect() as c:
        r = c.execute(text("SELECT data FROM blobs WHERE key = :k"), {"k": key}).fetchone()
    return None if r is None else bytes(r[0])


def blob_list(prefix: str = "") -> list[dict]:
    if not using_db():
        base = LOCAL / "blobs"
        if not base.exists():
            return []
        return [{"key": p.relative_to(base).as_posix(), "size": p.stat().st_size,
                 "updated_at": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")}
                for p in base.rglob("*") if p.is_file() and p.relative_to(base).as_posix().startswith(prefix)]
    from sqlalchemy import text
    with engine().connect() as c:
        rows = c.execute(text("SELECT key, octet_length(data), updated_at FROM blobs WHERE key LIKE :p ORDER BY key"),
                         {"p": prefix + "%"}).fetchall()
    return [{"key": k, "size": s, "updated_at": str(u)} for k, s, u in rows]


def describe() -> str:
    if not using_db():
        return f"lokal fayllar ({LOCAL})"
    u = db_url()
    host = u.split("@")[-1].split("/")[0] if "@" in u else "?"
    return f"Postgres ({host})"
