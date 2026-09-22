"""
Fetch REAL road distances for a stratified sample of Indian city pairs.

WHY THIS EXISTS
---------------
Both scripts/build_wide_dataset.py and app.py convert straight-line distance to road distance
with a single hardcoded constant:

    road_km = haversine_km * 1.2355        # "winding factor"

That constant was measured on this project's own nine original routes. On routes it was NOT
measured on it can be badly wrong. Checked by hand against the OSRM routing engine:

    Jaipur -> Agra    haversine 221.9 km
                      constant x 1.2355 -> 274.1 km
                      real road distance   238.0 km      <- constant overshoots by 15.2%

One constant cannot describe Indian road geometry: a coastal route bends, a Deccan highway runs
straight, a Himalayan road switchbacks. The winding factor is a *function of where you are*, and
a function of the inputs is exactly what a model is for.

This script collects the training data for that model. It is the only genuinely OBSERVED data in
the project - everything in data/road_trip_wide.csv is generated.

OUTPUT
------
data/real_distances.csv  -  one row per city pair:
    start_city,start_state,start_lat,start_lon,
    dest_city,dest_state,dest_lat,dest_lon,
    haversine_km,road_km,factor,source

RESUMABILITY
------------
OSRM's public demo server is rate-limited and makes no uptime promise. Every successful fetch is
appended to the CSV immediately, and re-running skips pairs already present. If the server starts
refusing, stop the script and re-run it later - nothing is lost.

Run:  python scripts/fetch_real_distances.py
      python scripts/fetch_real_distances.py --target 1200     (collect more)
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(os.path.dirname(ROOT), "backend")   # what deploys; owns the
                                                           # data and models it serves
CITIES = os.path.join(BACKEND, "data", "india_cities.csv")
OUT = os.path.join(ROOT, "data", "real_distances.csv")

SEED = 20260910
MIN_POPULATION = 100_000          # same city domain as the wide dataset (537 cities)

# Straight-line distance bands, in km. Sampling evenly across these stops the model from being
# fitted almost entirely on short trips: random city pairs in India are overwhelmingly long-haul,
# but real road trips are not.
BANDS = [(40, 120), (120, 250), (250, 400), (400, 600), (600, 900), (900, 1250)]
TARGET_PER_BAND = 140             # 6 x 140 = 840 pairs

OSRM_URL = "http://router.project-osrm.org/route/v1/driving/{},{};{},{}?overview=false"
OSRM_TIMEOUT = 12.0
PAUSE = 0.40                      # be a polite client on a free public server
MAX_RETRIES = 3

FIELDS = ["start_city", "start_state", "start_lat", "start_lon",
          "dest_city", "dest_state", "dest_lat", "dest_lon",
          "haversine_km", "road_km", "factor", "source"]


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def load_cities():
    with open(CITIES, encoding="utf-8") as fh:
        cities = [c for c in csv.DictReader(fh) if int(c["population"]) >= MIN_POPULATION]
    for c in cities:
        c["lat"], c["lon"] = float(c["lat"]), float(c["lon"])
    return cities


def load_done():
    """Pairs already fetched, as a set of (start_city, dest_city)."""
    if not os.path.exists(OUT):
        return set(), 0
    with open(OUT, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {(r["start_city"], r["dest_city"]) for r in rows}, len(rows)


def plan_pairs(cities, done):
    """Stratified sample of city pairs, evenly spread over the distance bands."""
    rng = random.Random(SEED)
    buckets = {b: [] for b in BANDS}
    seen = set()
    attempts = 0
    # Draw random pairs and file each into its band until every band is full. Rejection sampling
    # is fine here: the bands are wide and 537 cities give ~144k possible pairs.
    while attempts < 400_000 and any(len(v) < TARGET_PER_BAND for v in buckets.values()):
        attempts += 1
        a, b = rng.sample(cities, 2)
        key = tuple(sorted((a["city"], b["city"])))
        if key in seen:
            continue
        d = haversine(a["lat"], a["lon"], b["lat"], b["lon"])
        for band in BANDS:
            if band[0] <= d < band[1] and len(buckets[band]) < TARGET_PER_BAND:
                seen.add(key)
                buckets[band].append((a, b, d))
                break

    planned = [p for band in BANDS for p in buckets[band]]
    rng.shuffle(planned)   # interleave bands so an interrupted run still covers the whole range
    todo = [p for p in planned if (p[0]["city"], p[1]["city"]) not in done
            and (p[1]["city"], p[0]["city"]) not in done]
    for band in BANDS:
        print(f"  band {band[0]:>4}-{band[1]:<4} km : {len(buckets[band]):>3} pairs")
    return todo


def fetch_road_km(a, b):
    """Real driving distance in km from OSRM, or None if the server will not answer."""
    url = OSRM_URL.format(a["lon"], a["lat"], b["lon"], b["lat"])
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "roadtrip-ml/edu"})
            with urllib.request.urlopen(req, timeout=OSRM_TIMEOUT) as resp:
                payload = json.load(resp)
            if payload.get("code") != "Ok" or not payload.get("routes"):
                return None                      # no road route exists (island, bad geocode)
            return payload["routes"][0]["distance"] / 1000.0
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError,
                json.JSONDecodeError, KeyError):
            if attempt < MAX_RETRIES - 1:
                time.sleep(1.5 * (attempt + 1))  # linear backoff
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=TARGET_PER_BAND * len(BANDS),
                    help="total pairs wanted (across all bands)")
    args = ap.parse_args()

    cities = load_cities()
    print(f"cities (pop >= {MIN_POPULATION:,}) : {len(cities)}")

    done, n_existing = load_done()
    if n_existing:
        print(f"already fetched            : {n_existing} pairs (resuming)")

    print("planning a stratified sample:")
    todo = plan_pairs(cities, done)
    want = max(0, args.target - n_existing)
    todo = todo[:want]
    print(f"to fetch now               : {len(todo)} pairs "
          f"(~{len(todo) * PAUSE / 60:.1f} min at {PAUSE}s spacing)\n")

    if not todo:
        print("nothing to do.")
        return

    new_file = not os.path.exists(OUT)
    fh = open(OUT, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()

    ok = failed = 0
    try:
        for i, (a, b, straight) in enumerate(todo, 1):
            road = fetch_road_km(a, b)
            if road is None:
                failed += 1
            else:
                writer.writerow({
                    "start_city": a["city"], "start_state": a["state"],
                    "start_lat": f"{a['lat']:.5f}", "start_lon": f"{a['lon']:.5f}",
                    "dest_city": b["city"], "dest_state": b["state"],
                    "dest_lat": f"{b['lat']:.5f}", "dest_lon": f"{b['lon']:.5f}",
                    "haversine_km": f"{straight:.3f}", "road_km": f"{road:.3f}",
                    "factor": f"{road / straight:.5f}", "source": "osrm",
                })
                fh.flush()          # crash-safe: every good row is on disk before the next call
                ok += 1
            if i % 25 == 0 or i == len(todo):
                print(f"  {i:>4}/{len(todo)}   ok {ok}   failed {failed}")
            time.sleep(PAUSE)
    except KeyboardInterrupt:
        print("\ninterrupted - progress is saved, re-run to continue")
    finally:
        fh.close()

    print(f"\nwrote {ok} new rows to {OUT}  (failed {failed})")
    if failed > ok * 0.3 and ok:
        print("NOTE: high failure rate - OSRM is probably rate-limiting. Re-run later to top up.")


if __name__ == "__main__":
    sys.exit(main())
