"""Konfiqurasiya və yolların vahid nöqtəsi."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

ROOT = Path(__file__).resolve().parents[1]
if load_dotenv:
    load_dotenv(ROOT / ".env")


@lru_cache(maxsize=1)
def load_config() -> dict:
    path = Path(os.environ.get("GASFC_CONFIG", ROOT / "config.yaml"))
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    """`paths` bölməsindəki açarı mütləq yola çevirir (qovluğu yaradır)."""
    p = ROOT / load_config()["paths"][key]
    if not p.suffix:
        p.mkdir(parents=True, exist_ok=True)
    return p


def stations() -> dict:
    return load_config()["stations"]


def load_yaml(key: str) -> dict:
    p = path(key)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
