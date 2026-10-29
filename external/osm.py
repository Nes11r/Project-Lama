"""OpenStreetMap Overpass API (ODbL): xidmət ərazisində bina/obyekt tərkibi.

Hər məntəqə üçün: çoxmərtəbəli (apartments), fərdi ev (house/detached/semidetached), ümumi yaşayış,
sənaye, kommersiya, məktəb, xəstəxana sayları və payları. Overpass User-Agent tələb edir.

Qeyd: Bakı ətrafı qəsəbələrdə OSM-də binaların çoxu `building=yes` (tipsiz) kimi qeyd olunur —
payları ehtiyatla şərh edin (hesabatda `untyped_share` göstərilir).
"""
from __future__ import annotations

import json
import logging

from core.config import load_config, stations
from external.common import cache_dir, http_get, set_status

log = logging.getLogger(__name__)

QUERIES = {
    "buildings_total": 'way["building"]',
    "apartments": 'way["building"="apartments"]',
    "houses": 'way["building"~"^(house|detached|semidetached_house|terrace|bungalow)$"]',
    "residential_generic": 'way["building"="residential"]',
    "untyped": 'way["building"="yes"]',
    "industrial": 'way["building"~"^(industrial|warehouse|factory)$"]',
    "commercial": 'way["building"~"^(commercial|retail|office|supermarket)$"]',
    "schools": 'nwr["amenity"~"^(school|kindergarten)$"]',
    "hospitals": 'nwr["amenity"~"^(hospital|clinic)$"]',
}


def station_osm(station_id: str, refresh: bool = False) -> dict:
    f = cache_dir("osm") / f"{station_id}.json"
    if f.exists() and not refresh:
        return json.loads(f.read_text(encoding="utf-8"))
    a = stations()[station_id]["area"]
    r_m = int(a["radius_km"] * 1000)
    body = "".join(f'{q}(around:{r_m},{a["lat"]},{a["lon"]});out count;' for q in QUERIES.values())
    try:
        resp = http_get(load_config()["population"]["overpass_url"],
                        data={"data": f"[out:json][timeout:120];{body}"}, source="overpass")
        counts = [int(e["tags"].get("ways", 0)) + int(e["tags"].get("nodes", 0)) + int(e["tags"].get("relations", 0))
                  for e in resp.json()["elements"] if e.get("type") == "count"]
        c = dict(zip(QUERIES, counts))
        tot = max(c["buildings_total"], 1)
        typed_res = c["apartments"] + c["houses"]
        res = {**{f"osm_{k}": v for k, v in c.items()},
               "osm_apartment_share": round(c["apartments"] / typed_res, 3) if typed_res else None,
               "osm_industrial_share": round(c["industrial"] / tot, 4),
               "osm_commercial_share": round(c["commercial"] / tot, 4),
               "osm_untyped_share": round(c["untyped"] / tot, 3),
               "radius_km": a["radius_km"]}
        f.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        set_status("overpass", True)
        return res
    except Exception as e:
        set_status("overpass", False, str(e))
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
        log.error("Overpass %s: %s", station_id, e)
        return {}


def static_features() -> dict[str, dict]:
    """Məntəqə → statik xüsusiyyətlər (əhali + OSM)."""
    from external.population import station_population
    return {sid: {**station_population(sid), **station_osm(sid)} for sid in stations()}


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(static_features(), ensure_ascii=False, indent=2))
