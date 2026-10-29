#!/bin/sh
# API konteynerinin başlanğıcı: model yoxdursa əvvəlcə data hazırlanır və model öyrədilir.
set -e
if [ ! -f storage/models/LATEST ]; then
  echo ">> Model tapılmadı: təmizləmə + hava + qiymətləndirmə + təlim (ilk dəfə ~30 dəq)"
  python -m data.pipeline
  python -m external.weather "${WEATHER_END:-2024-08-31}" || echo ">> Hava API əlçatmazdır, keş istifadə olunur"
  if [ "${SKIP_EVAL:-0}" != "1" ]; then python -m models.evaluate --quick; fi
  python -m models.train --skip-data
fi
exec uvicorn api.main:app --host 0.0.0.0 --port 8000
