"""
Build data/road_trip_wide.csv - the 500+ city version of road_trip_data.csv.

WHY THIS EXISTS
---------------
The original road_trip_data.csv covers only 9 fixed city pairs (110-460 km). A model trained on
it cannot serve arbitrary Indian routes. This script regenerates the SAME cost physics over REAL
geography for 537 Indian cities (population >= 100k), giving ~25,000 trips across ~5,000 routes.

The cost formula was reverse-engineered from the original CSV at R2 = 0.999162 (mean abs error
Rs 18.44), so trips generated here are physically consistent with the original data:

    total_trip_cost = fuel_consumption_litres * fuel_price
                    + toll_cost + parking_cost
                    + distance_km * maintenance_rate(vehicle_type)

    fuel_consumption_litres = (distance_km / mileage) * traffic_multiplier

WHAT IS REAL vs. MODELLED
-------------------------
REAL      : city names, states, latitude/longitude (GeoNames, 3739 Indian cities)
REAL      : straight-line distance between cities (haversine on real coordinates)
CALIBRATED: road distance = haversine * 1.2355. That factor was measured against the project's
            own 9 routes (mean 1.2355, median 1.210, sd 0.082) and cross-checked against the OSRM
            routing engine, which agreed within 5-9%.
CALIBRATED: maintenance rate, traffic multipliers, toll rate, mileage ranges - all measured from
            the original CSV.
PARTLY REAL: fuel prices - see data/fuel_prices.csv, 6 states verified, 29 estimated.

DELIBERATE CHANGE FROM THE ORIGINAL DATA
----------------------------------------
In the original CSV, traffic_level is statistically INDEPENDENT of departure_hour (P(High) wanders
between 0.13 and 0.35 with no rush-hour pattern), which makes traffic unpredictable and any traffic
classifier no better than the 51% majority baseline. This generator instead gives traffic a real
rush-hour profile (morning 8-11, evening 17-21). That is more realistic AND it makes traffic
classification a genuine task rather than decoration. It is a modelling choice, stated here openly.

Run:  python scripts/build_wide_dataset.py
"""
import csv
import math
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CITIES = os.path.join(ROOT, "data", "india_cities.csv")
FUEL = os.path.join(ROOT, "data", "fuel_prices.csv")
OUT = os.path.join(ROOT, "data", "road_trip_wide.csv")

SEED = 42
MIN_POPULATION = 100_000
N_ROUTES = 5_000
TRIPS_PER_ROUTE = 5
MIN_KM, MAX_KM = 50.0, 1500.0
DISTANCE_DECAY_KM = 350.0        # short-trip preference; see the route sampling loop

WINDING_FACTOR = 1.2355          # measured on the project's own 9 routes
WINDING_SD = 0.082

# --- constants recovered from road_trip_data.csv -----------------------------------------
MAINTENANCE = {"Hatchback": 0.5694, "Sedan": 0.6943, "SUV": 0.8366}   # Rs/km
TRAFFIC_MULT = {"Low": 0.944, "Medium": 1.045, "High": 1.165}          # fuel burn multiplier
TRAFFIC_MIN_PER_KM = {"Low": 0.919, "Medium": 1.059, "High": 1.263}    # travel time
TOLL_RATE_MEAN, TOLL_RATE_SD = 1.301, 0.360                            # Rs/km
PARKING_CHOICES = [0.0, 40.0, 60.0, 80.0, 120.0, 150.0]
MILEAGE = {   # (vehicle, fuel) -> (min, max) km/l, from the original CSV
    ("Hatchback", "Petrol"): (14.41, 21.11), ("Hatchback", "Diesel"): (15.93, 23.12),
    ("Hatchback", "CNG"): (13.55, 19.74),
    ("Sedan", "Petrol"): (12.48, 18.24), ("Sedan", "Diesel"): (13.86, 20.04),
    ("Sedan", "CNG"): (11.72, 17.05),
    ("SUV", "Petrol"): (8.64, 14.38), ("SUV", "Diesel"): (9.58, 15.85),
    ("SUV", "CNG"): (8.18, 13.40),
}
VEHICLES = ["Hatchback", "Sedan", "SUV"]
VEHICLE_W = [0.38, 0.37, 0.25]
FUELS = ["Petrol", "Diesel", "CNG"]
FUEL_W = [0.52, 0.36, 0.12]

# Rush-hour traffic profile (see the note above). Weights are P(Low), P(Medium), P(High).
def traffic_weights(hour):
    if 8 <= hour <= 11 or 17 <= hour <= 21:
        return [0.10, 0.35, 0.55]          # peak
    if 6 <= hour <= 7 or 12 <= hour <= 16 or hour == 22:
        return [0.30, 0.50, 0.20]          # shoulder
    return [0.62, 0.31, 0.07]              # night / early morning

# Monsoon-aware weather: June-September is the Indian monsoon.
def weather_weights(month):
    if 6 <= month <= 9:
        return [0.22, 0.28, 0.05, 0.45]    # Clear, Cloudy, Fog, Rain
    if month in (12, 1):
        return [0.45, 0.24, 0.28, 0.03]    # winter fog
    return [0.63, 0.25, 0.04, 0.08]

WEATHERS = ["Clear", "Cloudy", "Fog", "Rain"]


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def main():
    rng = random.Random(SEED)

    with open(CITIES, encoding="utf-8") as fh:
        cities = [c for c in csv.DictReader(fh) if int(c["population"]) >= MIN_POPULATION]
    for c in cities:
        c["lat"], c["lon"] = float(c["lat"]), float(c["lon"])
    print(f"cities (pop >= {MIN_POPULATION:,}) : {len(cities)}")

    with open(FUEL, encoding="utf-8") as fh:
        prices = {r["state"]: r for r in csv.DictReader(fh)}

    # --- sample distinct routes within a realistic road-trip band ---
    routes, seen, attempts = [], set(), 0
    while len(routes) < N_ROUTES and attempts < N_ROUTES * 400:
        attempts += 1
        a, b = rng.sample(cities, 2)
        key = tuple(sorted((a["city"] + "|" + a["state"], b["city"] + "|" + b["state"])))
        if key in seen:
            continue
        straight = haversine(a["lat"], a["lon"], b["lat"], b["lon"])
        road = straight * WINDING_FACTOR
        if not (MIN_KM <= road <= MAX_KM):
            continue
        # Uniformly random city pairs are mostly LONG (median ~900 km across India), but real road
        # trips skew short. Accept short routes preferentially so the distance distribution looks
        # like actual travel demand and still covers the original CSV's 110-460 km band densely.
        if rng.random() > math.exp(-road / DISTANCE_DECAY_KM):
            continue
        seen.add(key)
        routes.append((a, b, straight))
    print(f"distinct routes sampled       : {len(routes)}  (from {attempts:,} attempts)")

    rows = []
    for origin, dest, straight in routes:
        for _ in range(TRIPS_PER_ROUTE):
            # road distance: calibrated winding factor with per-trip variation
            factor = max(1.05, rng.gauss(WINDING_FACTOR, WINDING_SD))
            distance = round(min(MAX_KM, max(MIN_KM, straight * factor)), 2)

            vehicle = rng.choices(VEHICLES, VEHICLE_W)[0]
            fuel = rng.choices(FUELS, FUEL_W)[0]
            lo, hi = MILEAGE[(vehicle, fuel)]
            mileage = round(rng.uniform(lo, hi), 2)

            hour = rng.randrange(24)
            month = rng.randrange(1, 13)
            traffic = rng.choices(["Low", "Medium", "High"], traffic_weights(hour))[0]
            weather = rng.choices(WEATHERS, weather_weights(month))[0]

            # fuel price: state of the ORIGIN city, plus small day-to-day variation
            row_price = prices.get(origin["state"])
            base = float(row_price[fuel.lower()]) if row_price else 107.5
            fuel_price = round(base * rng.uniform(0.97, 1.03), 2)

            litres = round((distance / mileage) * TRAFFIC_MULT[traffic], 3)
            toll = round(max(0.0, distance * max(0.0, rng.gauss(TOLL_RATE_MEAN, TOLL_RATE_SD))), 2)
            parking = rng.choice(PARKING_CHOICES)
            minutes = round(distance * TRAFFIC_MIN_PER_KM[traffic] * rng.uniform(0.95, 1.05), 2)

            maint = distance * MAINTENANCE[vehicle] * rng.uniform(0.80, 1.20)
            total = round(litres * fuel_price + toll + parking + maint, 2)

            rows.append({
                "start_city": origin["city"], "start_state": origin["state"],
                "destination_city": dest["city"], "destination_state": dest["state"],
                "distance_km": distance,
                "vehicle_type": vehicle, "fuel_type": fuel,
                "mileage": mileage, "fuel_price": fuel_price,
                "passengers": rng.randint(1, 5),
                "departure_hour": hour, "month": month,
                "traffic_level": traffic, "weather_condition": weather,
                "toll_cost": toll, "parking_cost": parking,
                "estimated_time_minutes": minutes,
                "fuel_consumption_litres": litres,
                "total_trip_cost": total,
            })

    # --- cost EFFICIENCY band (the classification target) ---
    # Deliberately NOT an absolute cost band: absolute cost is ~determined by distance alone, so
    # classifying it is trivial and tells the user nothing. Rs/km relative to the fleet is
    # informative - it isolates the effect of vehicle, mileage, fuel choice and traffic.
    per_km = sorted(r["total_trip_cost"] / r["distance_km"] for r in rows)
    q = [per_km[int(len(per_km) * f)] for f in (0.25, 0.50, 0.75)]
    for r in rows:
        v = r["total_trip_cost"] / r["distance_km"]
        r["cost_per_km"] = round(v, 3)
        r["cost_band"] = ("Excellent" if v < q[0] else "Good" if v < q[1]
                          else "Average" if v < q[2] else "Poor")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    used = {r["start_city"] for r in rows} | {r["destination_city"] for r in rows}
    dists = [r["distance_km"] for r in rows]
    costs = [r["total_trip_cost"] for r in rows]
    print(f"trips written                 : {len(rows):,} -> {OUT}")
    print(f"distinct cities used          : {len(used)}")
    print(f"distance km  min/mean/max     : {min(dists):.0f} / {sum(dists)/len(dists):.0f} / {max(dists):.0f}")
    print(f"total cost Rs min/mean/max    : {min(costs):.0f} / {sum(costs)/len(costs):.0f} / {max(costs):.0f}")
    print(f"cost_per_km quartile cuts     : {[round(x,2) for x in q]}")


if __name__ == "__main__":
    main()
