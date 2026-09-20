"""
Build data/india_cities.csv - every Indian city in GeoNames with population >= 15000,
with real latitude/longitude and state.

Source : https://download.geonames.org/export/dump/cities15000.zip  (CC BY 4.0)
Output : data/india_cities.csv  ->  city, state, lat, lon, population

Run:  python scripts/build_cities.py
"""
import csv
import io
import os
import urllib.request
import zipfile

HEADERS = {"User-Agent": "Mozilla/5.0"}
CITIES_URL = "https://download.geonames.org/export/dump/cities15000.zip"
ADMIN1_URL = "https://download.geonames.org/export/dump/admin1CodesASCII.txt"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "india_cities.csv")


def fetch(url):
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=HEADERS), timeout=180
    ).read()


def main():
    # admin1 code -> state name, e.g. "IN.16" -> "Maharashtra"
    states = {}
    for line in fetch(ADMIN1_URL).decode("utf-8").strip().split("\n"):
        parts = line.split("\t")
        if parts[0].startswith("IN."):
            states[parts[0].split(".")[1]] = parts[1]
    print(f"states mapped        : {len(states)}")

    blob = fetch(CITIES_URL)
    text = zipfile.ZipFile(io.BytesIO(blob)).read("cities15000.txt").decode("utf-8")
    print(f"world cities fetched : {len(text.strip().splitlines())}")

    cities = []
    for line in text.strip().split("\n"):
        f = line.split("\t")
        if f[8] != "IN":                       # country code
            continue
        state = states.get(f[10])              # admin1 code
        if not state:
            continue
        try:
            population = int(f[14])
        except ValueError:
            population = 0
        cities.append(
            {
                "city": f[2],                  # asciiname
                "state": state,
                "lat": round(float(f[4]), 5),
                "lon": round(float(f[5]), 5),
                "population": population,
            }
        )

    # Deduplicate on (city, state), keeping the largest by population.
    best = {}
    for c in cities:
        key = (c["city"].lower(), c["state"])
        if key not in best or c["population"] > best[key]["population"]:
            best[key] = c
    cities = sorted(best.values(), key=lambda c: -c["population"])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["city", "state", "lat", "lon", "population"])
        w.writeheader()
        w.writerows(cities)

    print(f"india cities written : {len(cities)}  -> {OUT}")
    print(f"distinct states      : {len(set(c['state'] for c in cities))}")
    print(f"population >= 100k   : {sum(1 for c in cities if c['population'] >= 100000)}")


if __name__ == "__main__":
    main()
