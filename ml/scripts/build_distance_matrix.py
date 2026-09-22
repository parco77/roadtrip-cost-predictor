"""
Build the road-distance reference table that the site's lookup section serves.

WHY A CURATED CITY LIST AND NOT "THE TOP N BY POPULATION"
---------------------------------------------------------
data/india_cities.csv is GeoNames-derived, and GeoNames ranks administrative units, not the
places people name when they describe a drive. Taking its top 100 rows verbatim puts
"Rasapudipalem" above Visakhapatnam, lists Pimpri twice (once as Pimpri, once as
Pimpri-Chinchwad), and fills the table with Mumbai->Dharavi at 8.5 km. A city autocomplete hides
that, because nobody searches for a suburb they have never heard of. A browsable table does not.

So the cities below are written out by hand. Every name is one a traveller would recognise, the
list spans every large state, and the coordinates still come from india_cities.csv - the names
are curated, the geography is not.

A proximity pass runs afterwards as a safety net: walking in population order, any city within
MIN_SEPARATION_KM of one already accepted is dropped. That catches twin cities and satellites
that slipped through by name (Howrah beside Kolkata, Navi Mumbai beside Mumbai).

WHY /table AND NOT /route
-------------------------
scripts/fetch_real_distances.py asks OSRM for one route at a time, which is right when the
routes are unrelated. Here every city is wanted against every other, and OSRM's /table endpoint
answers that shape directly: one request returns a full N x N distance matrix. The limit is
exactly 100 coordinates (101 returns HTTP 400 "TooBig"), so larger lists are covered by pairing
blocks of 50 and slicing the cross sub-matrix out of each response.

Measured against the 840 /route distances already in data/real_distances.csv, the two endpoints
agree to 0.008% on average - they are the same road network, asked a different way.

WHY THE TABLE IS FILTERED TO 50-1500 km
---------------------------------------
That is the range the cost model was trained on. Random pairs drawn from a national city list
are mostly long-haul - the median separation is over 1200 km - so an unfiltered table would
invite the user to type a number the estimator then has to flag as extrapolation. Routes outside
the band are fetched, counted and reported, then left out of the shipped file.

Run:  python scripts/build_distance_matrix.py
Out:  frontend/public/city_distances.json   (served statically by Vite and by FastAPI)
      data/curated_distances.csv            (raw fetch cache, so re-runs cost no requests)
"""
import csv
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

BACKEND = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "backend"
)
# roadtrip_features.py is the server's feature contract - training imports the very file
# the API imports, so a column added for the model cannot go missing at serve time.
sys.path.insert(0, BACKEND)
import roadtrip_features as rf                                          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CITIES = os.path.join(BACKEND, "data", "india_cities.csv")
CACHE = os.path.join(ROOT, "data", "curated_distances.csv")
OUT = os.path.join(os.path.dirname(ROOT), "frontend", "public", "city_distances.json")

TABLE_URL = "http://router.project-osrm.org/table/v1/driving/{}?annotations=distance"
MAX_COORDS = 100           # hard OSRM limit; 101 is a 400
BLOCK = MAX_COORDS // 2    # so any two blocks fit in one request
PAUSE = 0.5                # politeness on a free public demo server
TIMEOUT = 30.0

MIN_KM, MAX_KM = 50.0, 1500.0   # the cost model's trained range
MIN_SEPARATION_KM = 40.0        # twin-city / satellite guard

# Hand-written. Modern spellings, because that is what india_cities.csv uses.
CURATED = """
Mumbai Delhi Bengaluru Hyderabad Ahmedabad Chennai Kolkata Surat Pune Jaipur Kanpur Lucknow
Nagpur Coimbatore Indore Vadodara Bhopal Patna Ludhiana Nashik Madurai Tirunelveli Agra Rajkot
Jamshedpur Meerut Srinagar Dhanbad Aurangabad Varanasi Amritsar Vijayawada Ranchi Jabalpur
Prayagraj Visakhapatnam Jodhpur Gwalior Raipur Tiruchirappalli Kota Sholapur Chandigarh Tiruppur
Guwahati Mysuru Bareilly Aligarh Jalandhar Bhubaneswar Salem Warangal Guntur Bhiwandi Saharanpur
Gorakhpur Bikaner Amravati Noida Jamnagar Bhilai Cuttack Firozabad Dehradun Durgapur Asansol
Nanded Kolhapur Ajmer Kalaburagi Ujjain Siliguri Jhansi Mangaluru Erode Belagavi
Rajamahendravaram Kakinada Bokaro Ballari Patiala Agartala Bhagalpur Muzaffarnagar Latur Udaipur
Davangere Kozhikode Akola Kurnool Rajapalayam Bathinda Jalgaon Gaya Udupi Thrissur Alwar Bilaspur
Shimla Puducherry Thiruvananthapuram Kochi Kollam Nellore Tirupati Anantapur Hubballi Shivamogga
Tumkur Vellore Thanjavur Dindigul Karur Nagercoil Panipat Rohtak Hisar Karnal Sonipat Ratlam
Satna Rewa Saugor Korba Durg Rourkela Sambalpur Brahmapur Dibrugarh Silchar Imphal Shillong
Aizawl Jammu Pathankot Hoshiarpur Moradabad Rampur Mathura Etawah Farrukhabad Sitapur Azamgarh
Chapra Darbhanga Muzaffarpur Purnia Katihar Begusarai Arrah Hajipur Deoghar Hazaribagh Giridih
""".split()


def load_cities():
    """Curated names -> rows from india_cities.csv, deduplicated by proximity."""
    with open(CITIES, encoding="utf-8") as fh:
        rows = {r["city"]: r for r in csv.DictReader(fh)
                if int(r["population"]) >= 100_000}

    wanted, missing = [], []
    for name in dict.fromkeys(CURATED):          # dict.fromkeys = dedupe, keep order
        if name in rows:
            r = rows[name]
            wanted.append({"city": r["city"], "state": r["state"],
                           "lat": float(r["lat"]), "lon": float(r["lon"]),
                           "population": int(r["population"])})
        else:
            missing.append(name)
    if missing:
        print(f"  not in india_cities.csv, skipped: {', '.join(missing)}")

    wanted.sort(key=lambda c: -c["population"])
    kept = []
    for c in wanted:
        near = next((k for k in kept
                     if rf.haversine(c["lat"], c["lon"], k["lat"], k["lon"])
                     < MIN_SEPARATION_KM), None)
        if near:
            print(f"  proximity drop: {c['city']:<22} within {MIN_SEPARATION_KM:.0f} km "
                  f"of {near['city']}")
        else:
            kept.append(c)
    kept.sort(key=lambda c: c["city"])           # alphabetical is what a lookup table wants
    return kept


def osrm_table(coords):
    """coords = [(lon, lat), ...] (max 100) -> distances matrix in metres."""
    joined = ";".join(f"{lon},{lat}" for lon, lat in coords)
    req = urllib.request.Request(TABLE_URL.format(joined),
                                 headers={"User-Agent": "roadtrip-predictor/1.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        payload = json.load(resp)
    if payload.get("code") != "Ok":
        raise RuntimeError(f"OSRM returned {payload.get('code')}: {payload.get('message')}")
    return payload["distances"]


def load_cache():
    if not os.path.exists(CACHE):
        return {}
    with open(CACHE, encoding="utf-8") as fh:
        return {(r["a"], r["b"]): float(r["road_km"]) for r in csv.DictReader(fh)}


def save_cache(pairs):
    with open(CACHE, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["a", "b", "road_km"])
        for (a, b), km in sorted(pairs.items()):
            w.writerow([a, b, f"{km:.3f}"])


def fetch_all(cities, cache):
    """Fill in every missing city pair, one /table request per pair of blocks."""
    n = len(cities)
    blocks = [list(range(i, min(i + BLOCK, n))) for i in range(0, n, BLOCK)]
    print(f"  {n} cities -> {n * (n - 1) // 2} pairs, "
          f"{len(blocks)} blocks, {len(blocks) * (len(blocks) + 1) // 2} requests max")

    requests = 0
    for bi in range(len(blocks)):
        for bj in range(bi, len(blocks)):
            left, right = blocks[bi], blocks[bj]
            idx = left if bi == bj else left + right
            # Skip the whole request when every pair it would cover is already cached.
            need = [(a, b) for a in left for b in (left if bi == bj else right)
                    if a < b and (cities[a]["city"], cities[b]["city"]) not in cache]
            if not need:
                continue

            coords = [(cities[k]["lon"], cities[k]["lat"]) for k in idx]
            matrix = osrm_table(coords)
            requests += 1
            pos = {k: p for p, k in enumerate(idx)}
            for a, b in need:
                metres = matrix[pos[a]][pos[b]]
                if metres is not None:
                    cache[(cities[a]["city"], cities[b]["city"])] = metres / 1000.0
            print(f"    block {bi}x{bj}: +{len(need)} pairs  ({len(cache)} cached)")
            time.sleep(PAUSE)
    return requests


def main():
    print("CURATED CITY LIST")
    cities = load_cities()
    print(f"  {len(cities)} cities across {len({c['state'] for c in cities})} states\n")

    print("FETCHING ROAD DISTANCES (OSRM /table)")
    cache = load_cache()
    print(f"  {len(cache)} pairs already cached")
    try:
        made = fetch_all(cities, cache)
        print(f"  {made} request(s) made")
    except (urllib.error.URLError, TimeoutError, RuntimeError, OSError) as exc:
        print(f"  STOPPED: {exc}")
        print("  Writing what was fetched; re-run to continue from the cache.")
    save_cache(cache)
    print(f"  cache -> {CACHE}\n")

    print("FILTERING TO THE MODEL'S TRAINED RANGE")
    index = {c["city"]: i for i, c in enumerate(cities)}
    routes, short, long_ = [], 0, 0
    for (a, b), km in cache.items():
        if a not in index or b not in index:
            continue                             # city dropped since the cache was written
        if km < MIN_KM:
            short += 1
        elif km > MAX_KM:
            long_ += 1
        else:
            routes.append([index[a], index[b], round(km)])
    routes.sort(key=lambda r: (r[0], r[1]))
    total = len(routes) + short + long_
    print(f"  {total} fetched -> {len(routes)} kept")
    print(f"  dropped {short} under {MIN_KM:.0f} km and {long_} over {MAX_KM:.0f} km "
          f"({(short + long_) / total * 100:.1f}% of the matrix)")
    print("  Those are real distances; they are left out because the estimator would have to")
    print("  flag them as extrapolating, and a reference table should not hand the user a")
    print("  number its own calculator distrusts.\n")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({
            "cities": [{"city": c["city"], "state": c["state"]} for c in cities],
            "routes": routes,
            "range_km": [MIN_KM, MAX_KM],
            "source": "OSRM driving profile, road distance rounded to the nearest kilometre",
        }, fh, separators=(",", ":"))
    size = os.path.getsize(OUT)
    print(f"saved -> {OUT}  ({size / 1024:.1f} KB, {len(routes)} routes)")


if __name__ == "__main__":
    main()
