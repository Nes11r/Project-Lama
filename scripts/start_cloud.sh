#!/bin/sh
# Bulud (Render) API başlanğıcı: təlim YOXDUR — data və model Neon-dan oxunur
# (əvvəlcədən lokal maşında: python scripts/publish.py).
set -e
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
