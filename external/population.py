"""Əhali — WorldPop (CC BY 4.0) stats API: verilmiş poliqon daxilində ümumi əhali.

Xidmət əraziləri məlum olmadığından məntəqə ətrafında `radius_km` radiuslu dairə istifadə olunur
(config.yaml → stations.*.area). Dəqiq poliqonlar verildikdə `area.polygon` ilə əvəz edilə bilər.
Böyük ərazilər üçün halqalar (0–r/2, r/2–r) üzrə paylanma da saxlanılır.

Qeyd (yanaşma seçimi): bu göstəricilər zamanla demək olar dəyişmir. 3 məntəqəli modeldə onlar
faktiki olaraq məntəqə identifikatorunu təkrarlayır; faydası çox məntəqəli (global) modeldə
yeni məntəqəni tarixçəsi az olduqda belə ölçüləndirməkdir.

Alternativlər (yoxlanıb/qiymətləndirilib):
  - GHSL (JRC) — açıq, lakin yalnız böyük raster (GeoTIFF) yükləmə; API yoxdur
  - Kontur Population — H3 heksaqonlar, ölkə faylı (gpkg) yükləmə; API yoxdur
  - stat.gov.az — rayon səviyyəsində cədvəllər (yüklənən fayl), qəsəbə səviyyəsində dəqiq deyil
"""
from __future__ import annotations

import json
import logging
import math
import re

from core.config import load_config, stations
from external.common import cache_dir, http_get, set_status

log = logging.getLogger(__name__)


def circle(lat: float, lon: float, r_km: float, n: int = 32) -> list[list[float]]:
    pts = []
    for i in range(n + 1):
        a = 2 * math.pi * i / n
        dlat = (r_km / 111.32) * math.cos(a)
        dlon = (r_km / (111.32 * math.cos(math.radians(lat)))) * math.sin(a)
        pts.append([round(lon + dlon, 6), round(lat + dlat, 6)])
    return pts


def _worldpop_total(ring: list[list[float]]) -> float:
    cfg = load_config()["population"]
    gj = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [ring]}}]}
    r = http_get(cfg["worldpop_url"], {"dataset": cfg["worldpop_dataset"], "year": cfg["worldpop_year"],
                                       "geojson": json.dumps(gj, separators=(",", ":")), "runasync": "false"},
                 source="worldpop")
    # API JSON-dan əvvəl və sonra PHP xəbərdarlıqları qaytarır → yalnız JSON obyektini oxu
    m = re.search(r'\{\s*"status"', r.text)
    if not m:
        raise RuntimeError("WorldPop cavabında JSON tapılmadı")
    js, _ = json.JSONDecoder().raw_decode(r.text[m.start():])
    if js.get("error"):
        raise RuntimeError(js.get("error_message"))
    return float(js["data"]["total_population"])


def station_population(station_id: str, refresh: bool = False) -> dict:
    f = cache_dir("population") / f"{station_id}.json"
    if f.exists() and not refresh:
        return json.loads(f.read_text(encoding="utf-8"))
    a = stations()[station_id]["area"]
    r = a["radius_km"]
    try:
        inner = _worldpop_total(circle(a["lat"], a["lon"], r / 2))
        total = _worldpop_total(circle(a["lat"], a["lon"], r))
        res = {"pop_total": round(total), "pop_inner_share": round(inner / total, 3) if total else None,
               "area_km2": round(math.pi * r * r, 2), "pop_density": round(total / (math.pi * r * r), 1),
               "source": f"WorldPop {load_config()['population']['worldpop_year']}", "radius_km": r}
        f.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        set_status("worldpop", True)
        return res
    except Exception as e:
        set_status("worldpop", False, str(e))
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
        log.error("WorldPop %s: %s", station_id, e)
        return {}
